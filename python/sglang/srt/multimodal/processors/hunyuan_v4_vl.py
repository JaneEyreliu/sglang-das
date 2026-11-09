# Copyright 2025 The HunYuan team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HYV4 VL multimodal processor.

The chat template renders each image as ``<image_start><image_pad><image_end>``
and each video as ``<video_start><video_pad><video_end>``. ``process_mm_data``
resizes / patchifies the media, expands every triple to the full placeholder
layout (see ``configs/hy_v4_vl_processing.py``) and tokenizes the result; the
base class then derives per-image / per-frame offsets from the pad-token runs.
M-RoPE positions follow the reference ``get_mrope_input_positions``.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.configs.hy_v4_vl_processing import HYV4VLProcessor, sample_frame_indices
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.hunyuan_v4_vl import HYV4VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

logger = logging.getLogger(__name__)

# Fallback fps for videos handed over as raw, already-sampled frames.
_RAW_VIDEO_FPS = 24.0


def _triple_regex(start: str, pad: str, end: str) -> re.Pattern:
    # One pad (template output) or an already expanded run of pads.
    return re.compile(f"{re.escape(start)}(?:{re.escape(pad)})+{re.escape(end)}")


def _frames_to_tchw(frames) -> torch.Tensor:
    """Raw video frames (THWC / TCHW tensor, ndarray or list) -> uint8 TCHW."""
    if isinstance(frames, (list, tuple)):
        frames = torch.stack([torch.as_tensor(f) for f in frames])
    frames = torch.as_tensor(frames).detach().cpu()
    if frames.ndim != 4:
        raise ValueError(f"HYV4 expects 4D video frames, got {tuple(frames.shape)}")
    if frames.shape[1] not in (1, 3, 4) and frames.shape[-1] in (1, 3, 4):
        frames = frames.permute(0, 3, 1, 2)
    return frames.contiguous()


class HYV4VLMultimodalProcessor(BaseMultimodalProcessor):
    models = [HYV4VLForConditionalGeneration]
    # Decode on CPU; preprocessing (resize / patchify) runs on CPU anyway.
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        if not isinstance(_processor, HYV4VLProcessor):
            raise TypeError(
                "HYV4 VL needs the customized HYV4VLProcessor, got "
                f"{type(_processor).__name__}"
            )
        proc = _processor
        self.image_start_token_id = proc.image_start_token_id
        self.image_end_token_id = proc.image_end_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=proc.image_start_token + proc.image_token + proc.image_end_token,
            image_token_id=proc.image_token_id,
            image_token_regex=_triple_regex(
                proc.image_start_token, proc.image_token, proc.image_end_token
            ),
            video_token=proc.video_start_token + proc.video_token + proc.video_end_token,
            video_token_id=proc.video_token_id,
            video_token_regex=_triple_regex(
                proc.video_start_token, proc.video_token, proc.video_end_token
            ),
        ).build(_processor)

    def preprocess_fingerprint_payload(self) -> dict[str, Any]:
        payload = super().preprocess_fingerprint_payload()
        payload["hyv4"] = {
            "image": self._processor.image_config,
            "video": self._processor.video_config,
            "vision": self._processor.vision,
        }
        return payload

    def resolve_image_token_counts(self, images: List) -> List[int]:
        # Pad tokens per image, used to expand a pre-tokenized prompt's single
        # <image_pad> between its start / end markers.
        counts = []
        for image in images:
            if isinstance(image, torch.Tensor):
                height, width = image.shape[-2:]
            else:
                height, width = image.height, image.width
            counts.append(
                self._processor.image_token_count(height, width, self.image_config)
            )
        return counts

    def _sample_video(self, video):
        """-> (uint8 TCHW frames, frame indices, source fps)."""
        options = {**self._processor.video_config, **self.video_config}
        if isinstance(video, VideoDecoderWrapper):
            total, fps = len(video), video.avg_fps
            indices = sample_frame_indices(
                total,
                fps,
                fps=options.get("fps"),
                min_frames=options["min_frames"],
                max_frames=options["max_frames"],
                num_frames=options.get("num_frames"),
            )
            frames = video.get_frames_as_tensor(indices).cpu().permute(0, 3, 1, 2)
            return frames.contiguous(), indices, fps
        frames = _frames_to_tchw(video)
        return frames, list(range(frames.shape[0])), _RAW_VIDEO_FPS

    def process_mm_data(
        self,
        input_text,
        images=None,
        videos=None,
        audios=None,
        processor=None,
        **kwargs,
    ) -> dict:
        # pixel values stay fp32 like the reference; the vit casts to its dtype.
        processor, tokenizer = self._resolve_processor(processor)
        out: Dict[str, Any] = {}
        image_grids, video_grids, video_times = [], [], []
        if images:
            pixels = []
            for image in images:
                patches, grid = processor.preprocess_image(image, self.image_config)
                pixels.append(patches)
                image_grids.append(grid)
            out["pixel_values"] = torch.cat(pixels)
            out["image_grid_thw"] = torch.tensor(image_grids, dtype=torch.long)
        if videos:
            pixels = []
            for video in videos:
                frames, indices, fps = video
                patches, grid, times = processor.preprocess_video(
                    frames, indices, fps, self.video_config
                )
                pixels.append(patches)
                video_grids.append(grid)
                video_times.append(times)
            out["pixel_values_videos"] = torch.cat(pixels)
            out["video_grid_thw"] = torch.tensor(video_grids, dtype=torch.long)

        image_index = video_index = 0
        parts = []
        for part in re.split(self.mm_tokens.get_combined_regex(), input_text):
            modality = self.mm_tokens.get_modality_of_token(part)
            if modality == Modality.IMAGE:
                if image_index >= len(image_grids):
                    raise ValueError("More HYV4 image placeholders than images")
                parts.append(processor.image_replacement(image_grids[image_index]))
                image_index += 1
            elif modality == Modality.VIDEO:
                if video_index >= len(video_grids):
                    raise ValueError("More HYV4 video placeholders than videos")
                parts.append(
                    processor.video_replacement(
                        video_grids[video_index], video_times[video_index]
                    )
                )
                video_index += 1
            else:
                parts.append(part)
        if image_index != len(image_grids) or video_index != len(video_grids):
            raise ValueError(
                f"HYV4 media count mismatch: {image_index}/{len(image_grids)} images, "
                f"{video_index}/{len(video_grids)} videos placed in the prompt"
            )
        text = "".join(parts)
        # Same guard as the base processor against a double BOS.
        bos = getattr(tokenizer, "bos_token", None)
        add_special_tokens = not (
            self._tokenizer_auto_adds_specials and bos and text.startswith(bos)
        )
        out["input_ids"] = tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
        return out

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if request_obj.audio_data:
            raise ValueError("HYV4 VL does not accept audio inputs")
        entry_time = time.perf_counter()
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )
        if any(isinstance(v, dict) for v in base_output.images + base_output.videos):
            raise ValueError(
                "HYV4 VL does not support precomputed / processor-output media yet"
            )
        load_time = time.perf_counter()
        base_output.videos = [self._sample_video(v) for v in base_output.videos]
        sample_time = time.perf_counter()

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )
        process_time = time.perf_counter()

        input_ids = input_ids.flatten()
        input_ids_list = input_ids.tolist()
        image_grids = ret.get("image_grid_thw") if ret else None
        video_grids = ret.get("video_grid_thw") if ret else None
        mrope_positions, mrope_delta = self._processor.get_mrope_positions(
            input_ids_list,
            image_grids.tolist() if image_grids is not None else [],
            video_grids.tolist() if video_grids is not None else [],
        )
        logger.debug(
            "[HYV4VLProcessor Perf] rid=%s load %.2f ms, sample %.2f ms, "
            "process %.2f ms, mrope %.2f ms",
            getattr(request_obj, "rid", None),
            (load_time - entry_time) * 1000,
            (sample_time - load_time) * 1000,
            (process_time - sample_time) * 1000,
            (time.perf_counter() - process_time) * 1000,
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids_list,
            padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
                input_ids_list, mm_items
            ),
            mm_items=mm_items,
            im_start_id=self.image_start_token_id,
            im_end_id=self.image_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=torch.tensor([[mrope_delta]], dtype=torch.long),
        )
