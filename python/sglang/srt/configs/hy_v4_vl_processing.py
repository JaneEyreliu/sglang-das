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
"""HYV4 VL preprocessing, ported from the reference HYV4VLProcessor.

The checkpoint names ``HYV4VLProcessor`` / ``HYV4VLImageProcessor`` /
``HYV4VLVideoProcessor`` without shipping code, and transformers has no such
classes, so ``AutoProcessor`` cannot load it. This module is a dependency-light
stand-in registered for ``hy_v4_vl``: it carries the tokenizer, the processor
configs and the pure image/video/prompt/M-RoPE logic used by
``multimodal/processors/hunyuan_v4_vl.py``. It must not import model code (it
is imported from ``configs``).

Placeholder layout (same as the reference):
  image: <image_start> <image_pad> * vision_token_count <image_end>
  video: <video_start> ( "<{t:.1f} seconds>" <image_start> <video_pad> * n
         <image_end> ) * frames <video_end>
"""

import json
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

PREPROCESSOR_CONFIG_NAME = "preprocessor_config.json"
VIDEO_PREPROCESSOR_CONFIG_NAME = "video_preprocessor_config.json"

# Defaults of the reference HYV4VLConfig / HYV4VLVisionConfig.
_TOKEN_ID_DEFAULTS = {
    "image_start_token_id": 120118,
    "image_end_token_id": 120119,
    "image_token_id": 120120,
    "image_newline_token_id": 120121,
    "video_start_token_id": 120122,
    "video_end_token_id": 120123,
    "video_token_id": 120124,
}
_VISION_DEFAULTS = {
    "patch_size": 16,
    "spatial_merge_size": 2,
    "spatial_patch_size": 1,
    "temporal_patch_size": 1,
    "use_image_newline_token": True,
    "cat_extra_token": 0,
}
# Used only if preprocessor_config.json omits a key (the shipped one sets all).
_IMAGE_DEFAULTS = {
    "min_pixels": 4096,
    "max_pixels": 4194304,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}
# Reference HYV4VLVideoProcessor defaults; video_preprocessor_config.json
# overrides (the shipped one: max_pixels 122880000, 2..1024 frames, 6 / 1000).
_VIDEO_DEFAULTS = {
    "max_pixels": 33554432,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "fps": 2,
    "min_frames": 4,
    "max_frames": 100,
    "timestamp_token_per_frame": 4,
    "timestamp_token_reserve": 2000,
    "video_max_image_size": 1024,
}


def smart_resize(height, width, factor=32, min_pixels=4096, max_pixels=4194304):
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError(
            "Image dimensions and pixel limits must be positive and ordered"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must not exceed 200")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    h, w = max(h, factor), max(w, factor)
    if h * w > max_pixels:
        raise ValueError("Image aspect ratio and patch rounding exceed max_pixels")
    return h, w


def adaptive_resize(
    height,
    width,
    num_frames,
    patch_size=16,
    merge_size=2,
    max_pixels=33554432,
    video_max_image_size=1024,
    timestamp_token_per_frame=6,
    timestamp_token_reserve=2000,
):
    factor = patch_size * merge_size
    if min(height, width, num_frames, factor, max_pixels) <= 0:
        raise ValueError(
            "Video dimensions, frame count and pixel budget must be positive"
        )
    frame_tokens = height * width // factor**2 + max(height, width) // factor
    budget = max(
        1,
        max_pixels // factor**2
        - timestamp_token_reserve
        - timestamp_token_per_frame * num_frames,
    )
    if frame_tokens * num_frames > budget:
        scale = math.sqrt(budget / (frame_tokens * num_frames))
    elif height * width > video_max_image_size**2:
        scale = video_max_image_size / math.sqrt(height * width)
    else:
        scale = 1.0
    return (
        max(factor, int(height * scale) // factor * factor),
        max(factor, int(width * scale) // factor * factor),
    )


def patchify(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[T, C, H, W] -> [T * H/p * W/p, C * p * p], row-major per frame."""
    t, channels, height, width = frames.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    return (
        frames.reshape(
            t,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1, channels * patch_size**2)
    )


def sample_frame_indices(
    total: int, video_fps: float, fps=2, min_frames=4, max_frames=100, num_frames=None
) -> List[int]:
    """Reference HYV4VLVideoProcessor.sample_frames (segment midpoints)."""
    if total <= 0 or video_fps is None or video_fps <= 0:
        raise ValueError("Video metadata must contain positive frame count and fps")
    if num_frames is None:
        desired = math.floor(total / video_fps * fps) if fps and fps > 0 else max_frames
        num_frames = min(total, max(min_frames, min(max_frames, desired)))
    if not 0 < num_frames <= total:
        raise ValueError("num_frames must be positive and not exceed available frames")
    segment = (total - 1) / num_frames
    return [
        min(
            (int(np.round(segment * i)) + int(np.round(segment * (i + 1)))) // 2,
            total - 1,
        )
        for i in range(num_frames)
    ]


def vision_token_grid(grid_thw, vision, modality: str) -> Tuple[int, int, int]:
    """(frames, rows, cols) of placeholder tokens; cols include the newline."""
    t, h, w = (int(v) for v in grid_thw)
    if min(t, h, w) <= 0:
        raise ValueError("Vision grid dimensions must be positive")
    merge = vision["spatial_merge_size"]
    if modality == "video":
        merge *= vision["spatial_patch_size"]
        if t == 1 and vision["spatial_patch_size"] != 1:
            raise ValueError("Single-frame spatial video compression is unsupported")
    elif t != 1:
        raise ValueError("An image must have exactly one temporal grid entry")
    if h % merge or w % merge:
        raise ValueError("Vision grid must be divisible by its spatial merge factor")
    frames = math.ceil(t / vision["temporal_patch_size"]) if modality == "video" else 1
    rows = h // merge
    cols = w // merge + int(bool(vision["use_image_newline_token"]))
    return frames, rows, cols


def vision_token_count(grid_thw, vision, modality: str) -> int:
    frames, rows, cols = vision_token_grid(grid_thw, vision, modality)
    return frames * (rows * cols + 2 * int(bool(vision["cat_extra_token"])))


def _to_uint8_chw(image) -> torch.Tensor:
    """PIL / HWC ndarray / CHW or HWC tensor -> uint8 RGB [C, H, W] on CPU."""
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    else:
        from torchvision.transforms.functional import pil_to_tensor

        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = pil_to_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"HYV4 expects a single [C, H, W] image, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    elif tensor.shape[0] == 4:
        tensor = tensor[:3]
    return tensor


def _normalize(frames: torch.Tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    return (frames.to(torch.float32) / 255.0 - mean) / std


def _resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    # v2 like the reference's TorchvisionBackend; v1 differs by up to 22 levels on uint8.
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as TF

    if frames.shape[-2:] == (height, width):
        return frames
    return TF.resize(
        frames,
        [height, width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def _read_json(model_path: str, name: str, revision: Optional[str]) -> dict:
    local = os.path.join(model_path, name)
    if os.path.isfile(local):
        path = local
    else:
        from transformers.utils.hub import cached_file

        path = cached_file(
            model_path, name, revision=revision, _raise_exceptions_for_missing_entries=False
        )
        if path is None:
            return {}
    with open(path) as f:
        return json.load(f)


class HYV4VLProcessor:
    """Tokenizer + HYV4 VL preprocessing; stands in for the missing HF class."""

    def __init__(
        self,
        tokenizer,
        image_config: Optional[dict] = None,
        video_config: Optional[dict] = None,
        token_ids: Optional[dict] = None,
        vision_config: Optional[dict] = None,
        chat_template: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_config = {**_IMAGE_DEFAULTS, **(image_config or {})}
        self.video_config = {**_VIDEO_DEFAULTS, **(video_config or {})}
        self.vision = {**_VISION_DEFAULTS, **(vision_config or {})}
        token_ids = {**_TOKEN_ID_DEFAULTS, **(token_ids or {})}
        for name, value in token_ids.items():
            setattr(self, name, int(value))
        for name in _TOKEN_ID_DEFAULTS:
            token = tokenizer.convert_ids_to_tokens(getattr(self, name))
            if token is None or tokenizer.convert_tokens_to_ids(token) != getattr(
                self, name
            ):
                raise ValueError(
                    f"HYV4 tokenizer does not define {name}={getattr(self, name)}"
                )
            setattr(self, name.removesuffix("_id"), token)
        for key, vision_key in (
            ("patch_size", "patch_size"),
            ("merge_size", "spatial_merge_size"),
            ("spatial_patch_size", "spatial_patch_size"),
            ("temporal_patch_size", "temporal_patch_size"),
        ):
            for config in (self.image_config, self.video_config):
                if key in config and int(config[key]) != int(self.vision[vision_key]):
                    raise ValueError(
                        f"HYV4 processor {key}={config[key]} disagrees with "
                        f"vision_config {vision_key}={self.vision[vision_key]}"
                    )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        from transformers import AutoTokenizer

        revision = kwargs.pop("revision", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        path = pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=trust_remote_code, revision=revision
        )
        config = _read_json(path, "config.json", revision)
        text = config.get("text_config") or config
        token_ids = {
            name: config.get(name, text.get(name, default))
            for name, default in _TOKEN_ID_DEFAULTS.items()
        }
        return cls(
            tokenizer,
            image_config=_read_json(path, PREPROCESSOR_CONFIG_NAME, revision),
            video_config=_read_json(path, VIDEO_PREPROCESSOR_CONFIG_NAME, revision),
            token_ids=token_ids,
            vision_config=config.get("vision_config"),
            chat_template=getattr(tokenizer, "chat_template", None),
        )

    # ------------------------------------------------------------------ image
    def image_size(self, height: int, width: int, overrides: Optional[dict] = None):
        options = {**self.image_config, **(overrides or {})}
        factor = self.vision["patch_size"] * self.vision["spatial_merge_size"]
        return smart_resize(
            height, width, factor, options["min_pixels"], options["max_pixels"]
        )

    def image_token_count(self, height: int, width: int, overrides=None) -> int:
        h, w = self.image_size(height, width, overrides)
        patch = self.vision["patch_size"]
        return vision_token_count((1, h // patch, w // patch), self.vision, "image")

    def preprocess_image(self, image, overrides: Optional[dict] = None):
        """-> (pixel_values [h*w/p^2, 3*p*p] fp32, grid [1, H/p, W/p])."""
        options = {**self.image_config, **(overrides or {})}
        frames = _to_uint8_chw(image).unsqueeze(0)
        height, width = self.image_size(*frames.shape[-2:], overrides)
        frames = _resize(frames, height, width)
        frames = _normalize(frames, options["image_mean"], options["image_std"])
        patch = self.vision["patch_size"]
        return patchify(frames, patch), [1, height // patch, width // patch]

    # ------------------------------------------------------------------ video
    def preprocess_video(
        self,
        frames: torch.Tensor,
        frame_indices: Sequence[int],
        video_fps: float,
        overrides: Optional[dict] = None,
    ):
        """frames: uint8 [T, C, H, W] already sampled at ``frame_indices``.

        -> (pixel_values_videos [T*h*w, 3*p*p], grid [T, H/p, W/p], timestamps)
        """
        options = {**self.video_config, **(overrides or {})}
        t, channels, height, width = frames.shape
        if t == 0:
            raise ValueError("Video must contain at least one frame")
        if t > options["max_frames"]:
            raise ValueError("Video exceeds max_frames; sample frames before encoding")
        if channels == 4:
            frames = frames[:, :3]
        elif channels == 1:
            frames = frames.expand(-1, 3, -1, -1)
        patch = self.vision["patch_size"]
        merge = self.vision["spatial_merge_size"]
        height, width = adaptive_resize(
            height,
            width,
            t,
            patch,
            merge,
            options["max_pixels"],
            options["video_max_image_size"],
            options["timestamp_token_per_frame"],
            options["timestamp_token_reserve"],
        )
        if t * height * width > options["max_pixels"]:
            raise ValueError("Processed video exceeds its total max_pixels budget")
        frames = _resize(frames, height, width)
        frames = _normalize(frames, options["image_mean"], options["image_std"])
        indices = list(frame_indices)
        if len(indices) != t:
            raise ValueError("Video frame indices do not match the decoded frames")
        tau = self.vision["temporal_patch_size"]
        indices += [indices[-1]] * (-len(indices) % tau)
        timestamps = [
            sum(indices[i : i + tau]) / tau / video_fps
            for i in range(0, len(indices), tau)
        ]
        return patchify(frames, patch), [t, height // patch, width // patch], timestamps

    # ----------------------------------------------------------------- prompt
    def image_replacement(self, grid) -> str:
        return (
            self.image_start_token
            + self.image_token * vision_token_count(grid, self.vision, "image")
            + self.image_end_token
        )

    def video_replacement(self, grid, timestamps: Sequence[float]) -> str:
        frames, _, _ = vision_token_grid(grid, self.vision, "video")
        per_frame = vision_token_count(grid, self.vision, "video") // frames
        if len(timestamps) != frames:
            raise ValueError("Video timestamps must match merged frame count")
        parts = [self.video_start_token]
        for timestamp in timestamps:
            parts.append(f"<{float(timestamp):.1f} seconds>")
            parts.append(self.image_start_token)
            parts.append(self.video_token * per_frame)
            parts.append(self.image_end_token)
        parts.append(self.video_end_token)
        return "".join(parts)

    # ----------------------------------------------------------------- M-RoPE
    def get_mrope_positions(
        self,
        input_ids: Sequence[int],
        image_grids: List[List[int]],
        video_grids: List[List[int]],
    ) -> Tuple[torch.Tensor, int]:
        """Reference HYV4VLForConditionalGeneration.get_mrope_input_positions.

        ``input_ids`` is the expanded (unpadded) prompt. Returns ``([3, L], delta)``
        where ``delta = max_position + 1 - L``.
        """
        tokens = torch.as_tensor(list(input_ids), dtype=torch.long)
        length = len(tokens)
        positions = torch.arange(length).repeat(3, 1)
        if not length or (not image_grids and not video_grids):
            return positions, 0
        vision = self.vision
        extra = int(bool(vision["cat_extra_token"]))
        starts = torch.where(tokens == self.image_start_token_id)[0].tolist()
        ends = torch.where(tokens == self.image_end_token_id)[0].tolist()
        if len(starts) != len(ends):
            raise ValueError("HYV4 image/frame start and end tokens must be paired")
        token_list = tokens.tolist()
        start_pos = start_index = mm_index = scan = 0
        image_index = video_index = -1
        inside_video = False
        while mm_index < len(starts):
            marker = starts[mm_index]
            pos = marker + 1 + extra
            while scan < marker:
                if token_list[scan] == self.video_start_token_id:
                    inside_video = True
                elif token_list[scan] == self.video_end_token_id:
                    inside_video = False
                scan += 1
            is_video = inside_video or (
                image_index + 1 >= len(image_grids)
                and video_index + 1 < len(video_grids)
            )
            if start_pos < pos:
                positions[:, start_pos:pos] = torch.arange(pos - start_pos) + start_index
                start_index += pos - start_pos
            if is_video:
                video_index += 1
                if video_index >= len(video_grids):
                    raise ValueError("Video placeholders do not match video_grid_thw")
                frames, rows, cols = vision_token_grid(
                    video_grids[video_index], vision, "video"
                )
            else:
                image_index += 1
                if image_index >= len(image_grids):
                    raise ValueError("Image placeholders do not match image_grid_thw")
                frames, rows, cols = vision_token_grid(
                    image_grids[image_index], vision, "image"
                )
            if mm_index + frames > len(starts):
                raise ValueError("Missing HYV4 image/frame boundary tokens")
            count = rows * cols
            pad_id = self.video_token_id if is_video else self.image_token_id
            for frame in range(frames):
                begin = starts[mm_index + frame] + 1 + extra
                end = begin + count
                if end + extra != ends[mm_index + frame] or not torch.all(
                    tokens[begin - extra : end + extra] == pad_id
                ):
                    raise ValueError(
                        "HYV4 visual placeholders do not match frame boundaries and grid"
                    )
                positions[0, begin:end] = start_index
                positions[1, begin:end] = (
                    torch.arange(rows).repeat_interleave(cols) + start_index
                )
                positions[2, begin:end] = torch.arange(cols).repeat(rows) + start_index
                start_index += max(rows, cols)
                if is_video and frame + 1 < frames:
                    bridge_start = ends[mm_index + frame]
                    bridge_end = starts[mm_index + frame + 1] + 1
                    positions[:, bridge_start:bridge_end] = (
                        torch.arange(bridge_end - bridge_start) + start_index
                    )
                    start_index += bridge_end - bridge_start
            start_pos = ends[mm_index + frames - 1] if is_video else pos + count
            mm_index += frames
        if start_pos < length:
            positions[:, start_pos:] = torch.arange(length - start_pos) + start_index
        if image_index + 1 != len(image_grids) or video_index + 1 != len(video_grids):
            raise ValueError("Unused HYV4 media grids after position construction")
        return positions, int(positions.max()) + 1 - length
