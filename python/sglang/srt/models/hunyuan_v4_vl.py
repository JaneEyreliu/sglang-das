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
"""HYV4 VL: the HYV4 text model plus a learned-position ViT and conv projector.

The text stack is inherited unchanged so ``model.*`` / ``lm_head.*`` weight
names, quantization prefixes and every HYV4-specific code path keep working.
The vision tower lives at ``vit`` to match the checkpoint, stays in bf16 and is
replicated on every rank (16 heads x 96 dims, not worth sharding).
"""

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.attention.vision import (
    VisionAttention,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.hunyuan_v4 import HYV4ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class HYV4VisionMLP(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.dense_h_to_4h = ReplicatedLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=None,
            prefix=add_prefix("dense_h_to_4h", prefix),
        )
        self.dense_4h_to_h = ReplicatedLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=None,
            prefix=add_prefix("dense_4h_to_h", prefix),
        )
        if config.hidden_act not in ("gelu", "gelu_pytorch_tanh"):
            raise ValueError(f"Unsupported HYV4 vision act: {config.hidden_act}")
        self.approximate = "tanh" if config.hidden_act == "gelu_pytorch_tanh" else "none"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.dense_h_to_4h(x)
        x = F.gelu(x, approximate=self.approximate)
        x, _ = self.dense_4h_to_h(x)
        return x


class HYV4VisionBlock(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Checkpoint q/k/v_proj load into qkv_proj shards and o_proj into proj.
        self.self_attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            quant_config=None,
            prefix=add_prefix("self_attn", prefix),
            proj_bias=True,
            qkv_bias=True,
            use_data_parallel=True,
        )
        self.mlp = HYV4VisionMLP(config, prefix=add_prefix("mlp", prefix))

    def forward(self, x: torch.Tensor, forward_metadata) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x), forward_metadata=forward_metadata
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class HYV4VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position_edge = config.max_image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
            bias=config.add_patchemb_bias,
        )
        self.position_embedding = nn.Embedding(
            self.position_edge**2 + 1, config.hidden_size, dtype=torch.float32
        )

    def compute_pos_embeds(
        self, grid_thw: List[List[int]], dtype: torch.dtype
    ) -> torch.Tensor:
        """Bilinearly resized learned positions, ``(num_patches, hidden)``."""
        hidden = self.config.hidden_size
        # contiguous() matters: some vendor upsample kernels ignore strides.
        positions = (
            self.position_embedding.weight[1:]
            .float()
            .reshape(1, self.position_edge, self.position_edge, hidden)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        out = []
        for t, h, w in grid_thw:
            resized = F.interpolate(
                positions,
                scale_factor=(
                    (h + 0.1) / self.position_edge,
                    (w + 0.1) / self.position_edge,
                ),
                mode=self.config.interpolate_mode,
                align_corners=False,
            )
            out.append(resized.flatten(2).transpose(1, 2).to(dtype).repeat(1, t, 1))
        return torch.cat(out, dim=1).squeeze(0)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: List[List[int]]
    ) -> torch.Tensor:
        c, p = self.config.num_channels, self.config.patch_size
        weight = self.patch_embedding.weight
        patches = pixel_values.to(device=weight.device, dtype=weight.dtype)
        patches = self.patch_embedding(patches.reshape(-1, c, p, p)).flatten(1)
        return patches + self.compute_pos_embeds(grid_thw, patches.dtype)


class HYV4VisionPatchMerger(nn.Module):
    """rmsnorm_mlp_adapter: RMSNorm -> Conv(k=merge) -> GELU -> Conv1x1 + newline."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.projector_type != "rmsnorm_mlp_adapter":
            raise ValueError(
                f"Unsupported HYV4 projector_type: {config.projector_type}"
            )
        if config.cat_extra_token or config.perceive_post_norm:
            raise ValueError(
                "HYV4 VL: cat_extra_token / perceive_post_norm are not supported"
            )
        dim, merge = config.hidden_size, config.spatial_merge_size
        hidden = config.perceive_intermediate_size or dim * merge**2
        self.norm = RMSNorm(dim, eps=config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, hidden, merge, stride=merge),
            nn.GELU(),
            nn.Conv2d(hidden, config.out_hidden_size, 1),
        )
        if config.use_image_newline_token:
            self.image_newline = nn.Parameter(torch.empty(config.out_hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (frames, h, w, C)
        config = self.config
        dtype = x.dtype
        x = self.norm(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        x = self.proj(x.permute(0, 3, 1, 2).contiguous())
        if x.shape[0] > 1:
            # Pool frames as an explicit 5D batch: (1, C, T, H, W).
            x = (
                F.avg_pool3d(
                    x.permute(1, 0, 2, 3).unsqueeze(0),
                    (config.temporal_patch_size, 1, 1),
                    stride=(config.temporal_patch_size, 1, 1),
                )
                .squeeze(0)
                .permute(1, 0, 2, 3)
            )
            x = F.avg_pool2d(
                x, config.spatial_patch_size, stride=config.spatial_patch_size
            )
        b, c, h, _ = x.shape
        if config.use_image_newline_token:
            newline = self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1)
            x = torch.cat((x, newline.to(dtype)), dim=-1)
        x = x.flatten(2).transpose(1, 2)
        return x.reshape(-1, x.shape[-1])


class HYV4VisionTransformer(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        if config.use_2d_rope:
            raise ValueError("HYV4 VL: use_2d_rope vision towers are not supported")
        self.config = config
        self.embeddings = HYV4VisionPatchEmbed(config)
        self.layers = nn.ModuleList(
            [
                HYV4VisionBlock(config, prefix=add_prefix(f"layers.{i}", prefix))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.perceive = HYV4VisionPatchMerger(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: List[List[int]]
    ) -> torch.Tensor:
        """Packed patches ``(sum t*h*w, c*p*p)`` -> LM embeddings ``(tokens, H)``."""
        lengths = [h * w for t, h, w in grid_thw for _ in range(t)]
        if not lengths or min(lengths) <= 0 or sum(lengths) != pixel_values.shape[0]:
            raise ValueError(
                f"HYV4 pixel_values ({pixel_values.shape[0]}) do not match "
                f"grid_thw {grid_thw}"
            )
        hidden = self.embeddings(pixel_values, grid_thw).unsqueeze(0)
        cu_seqlens = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        metadata = prepare_vision_attention_metadata(cu_seqlens, self.device)
        for layer in self.layers:
            hidden = layer(hidden, metadata)
        hidden = hidden.squeeze(0)

        temporal = self.config.temporal_patch_size
        outputs, offset = [], 0
        for t, h, w in grid_thw:
            states = hidden[offset : offset + t * h * w].reshape(t, h, w, -1)
            offset += t * h * w
            if t > 1 and t % temporal:
                pad = temporal - t % temporal
                states = torch.cat((states, states[-1:].expand(pad, -1, -1, -1)))
            outputs.append(self.perceive(states))
        return torch.cat(outputs)


def _vision_token_count(grid, config, modality: str) -> int:
    t, h, w = (int(v) for v in grid)
    merge = config.spatial_merge_size
    frames = 1
    if modality == "video":
        merge *= config.spatial_patch_size
        frames = -(-t // config.temporal_patch_size)
    cols = w // merge + int(bool(config.use_image_newline_token))
    return frames * (h // merge) * cols


class HYV4VLForConditionalGeneration(HYV4ForCausalLM):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__(config.text_config, quant_config, prefix)
        self.config = config
        self.text_config = config.text_config
        self.vit = HYV4VisionTransformer(
            config.vision_config, prefix=add_prefix("vit", prefix)
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def _vision_feature(
        self, items: List[MultimodalDataItem], modality: str
    ) -> torch.Tensor:
        key = f"{modality}_grid_thw"
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        grids = torch.cat(
            [torch.as_tensor(getattr(item, key)).reshape(-1, 3) for item in items]
        ).tolist()
        embeds = self.vit(pixel_values, grids)
        vcfg = self.config.vision_config
        expected = sum(_vision_token_count(g, vcfg, modality) for g in grids)
        if embeds.shape[0] != expected:
            raise ValueError(
                f"HYV4 {modality} embeddings {tuple(embeds.shape)} do not match "
                f"{expected} placeholder tokens for grids {grids}"
            )
        return embeds

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return self._vision_feature(items, "image")

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return self._vision_feature(items, "video")

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        mode = forward_batch.forward_mode
        use_mm = (
            input_embeds is None
            and self.pp_group.is_first_rank
            and not mode.is_decode()
            and not mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        )
        if not use_mm:
            # Text-only and decode batches take the unchanged HYV4 path.
            return super().forward(
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.VIDEO: self.get_video_feature,
            },
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def _load_vit_weight(self, params, name: str, weight: torch.Tensor) -> str:
        for proj, shard in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
            if f".self_attn.{proj}." in name:
                name = name.replace(f".self_attn.{proj}.", ".self_attn.qkv_proj.")
                param = params[name]
                param.weight_loader(param, weight, shard)
                return name
        name = name.replace(".self_attn.o_proj.", ".self_attn.proj.")
        param = params[name]
        getattr(param, "weight_loader", default_weight_loader)(param, weight)
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        vit_params = dict(self.vit.named_parameters())
        vit_loaded = set()

        def text_weights():
            for name, weight in weights:
                if name.startswith("vit."):
                    inner = name[len("vit.") :]
                    if inner.startswith("vit."):
                        inner = inner[len("vit.") :]
                    vit_loaded.add(self._load_vit_weight(vit_params, inner, weight))
                    continue
                yield name, weight

        # One pass over the stream so the text loader sees every text tensor.
        super().load_weights(text_weights())
        missing = sorted(set(vit_params) - vit_loaded)
        if missing:
            raise ValueError(
                f"HYV4 VL checkpoint is missing {len(missing)} vit tensors, "
                f"e.g. {missing[:4]}"
            )


EntryClass = [HYV4VLForConditionalGeneration]
