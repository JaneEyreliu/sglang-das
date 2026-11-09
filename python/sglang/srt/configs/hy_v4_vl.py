# Hunyuan-V4 VL (Hy4-testing) model configuration.
#
# The checkpoint stores the HYV4 text fields flat at the top level and nests
# only ``vision_config``. The text half is built as a ``HYV4Config`` and the
# outer config forwards unknown attributes to it, because many call sites read
# text fields (``index_topk``, ``hc_mult`` ...) directly off ``hf_config``.
from typing import Any

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.hy_v4 import HYV4Config
from sglang.srt.configs.hy_v4_vl_processing import HYV4VLProcessor
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


class HYV4VLVisionConfig(PretrainedConfig):
    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        out_hidden_size=None,
        hidden_act="gelu",
        rms_norm_eps=1e-5,
        max_image_size=2048,
        interpolate_mode="bilinear",
        cat_extra_token=0,
        perceive_pre_norm=True,
        perceive_post_norm=True,
        projector_type=None,
        perceive_intermediate_size=None,
        use_image_newline_token=True,
        use_2d_rope=False,
        add_patchemb_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_image_size = max_image_size
        self.interpolate_mode = interpolate_mode
        self.cat_extra_token = cat_extra_token
        self.perceive_pre_norm = perceive_pre_norm
        self.perceive_post_norm = perceive_post_norm
        self.projector_type = projector_type
        self.perceive_intermediate_size = perceive_intermediate_size
        self.use_image_newline_token = use_image_newline_token
        self.use_2d_rope = use_2d_rope
        self.add_patchemb_bias = add_patchemb_bias
        if use_2d_rope:
            raise ValueError("HYV4 VL vision 2D RoPE is not supported")


# The checkpoint names HYV4VLProcessor without shipping code; AutoProcessor
# cannot load it, so get_processor picks this registered stand-in.
@register_customized_processor(HYV4VLProcessor)
class HYV4VLConfig(PretrainedConfig):
    model_type = "hy_v4_vl"
    sub_configs = {"text_config": HYV4Config, "vision_config": HYV4VLVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    _OUTER_KEYS = (
        "architectures",
        "auto_map",
        "processor_class",
        "transformers_version",
        "quantization_config",
    )

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_start_token_id=120118,
        image_end_token_id=120119,
        image_token_id=120120,
        image_newline_token_id=120121,
        video_start_token_id=120122,
        video_end_token_id=120123,
        video_token_id=120124,
        **kwargs,
    ):
        outer_kwargs = {k: kwargs.pop(k) for k in self._OUTER_KEYS if k in kwargs}
        kwargs.pop("model_type", None)
        if isinstance(text_config, HYV4Config):
            text = text_config
        else:
            # Nested text fields win; flat top-level fields fill the rest.
            text_kwargs = {**kwargs, **(text_config or {})}
            text_kwargs.pop("model_type", None)
            rope = dict(text_kwargs.pop("rope_scaling", None) or {})
            rope.update(text_kwargs.get("rope_parameters") or {})
            rope.setdefault("rope_theta", 10000000.0)
            rope.setdefault("rope_type", "default")
            rope.setdefault("mrope_interleaved", True)
            rope.setdefault("mrope_section", [12, 10, 10])
            text_kwargs["rope_parameters"] = rope
            text = HYV4Config(**text_kwargs)
        # The text model reads arch-gated flags (is_deepseek_dsa / is_hy_v4) off
        # this nested config, so it must carry the text architecture itself.
        if not getattr(text, "architectures", None):
            text.architectures = ["HYV4ForCausalLM"]
        for key in ("dtype", "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id"):
            if hasattr(text, key):
                outer_kwargs.setdefault(key, getattr(text, key))
        super().__init__(**outer_kwargs)
        self.text_config = text
        if isinstance(vision_config, HYV4VLVisionConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = HYV4VLVisionConfig(**(vision_config or {}))
        if self.vision_config.out_hidden_size is None:
            self.vision_config.out_hidden_size = text.hidden_size
        if self.vision_config.out_hidden_size != text.hidden_size:
            raise ValueError("HYV4 VL projector output must match text hidden_size")
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.image_newline_token_id = image_newline_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_token_id = video_token_id

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        text = self.__dict__.get("text_config")
        if text is not None and hasattr(text, name):
            return getattr(text, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


__all__ = ["HYV4VLConfig", "HYV4VLVisionConfig"]
