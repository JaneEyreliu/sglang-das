import functools
import logging
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_index_topk,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    maybe_configure_main_kv_page_plan,
    maybe_prefetch_full_attention_kv,
    maybe_prefetch_next_full_attention_kv,
)
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.layer_boundary import (
    IHCState,
    append_stages,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.hy4_dcp import validate_hyv4_launch
from sglang.srt.layers.hy4_ihc_tilelang import (
    try_tilelang_ihc_head, try_tilelang_ihc_post, try_tilelang_ihc_pre,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.hyv4_gate_cp_shard import (
    build_gate_cp_chain,
    get_gate_cp_shard_runtime,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.mem_cache.hyv4_dcp_write_locs import scoped_hyv4_dcp_write_locs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2MoE,
)
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    get_device_capability,
    is_cuda,
    is_hcu,
    make_pp_layers,
)

logger = logging.getLogger(__name__)

_is_hcu = is_hcu()


def enable_hyv4_gate_proj_parallel() -> bool:
    """Whether HYV4 overlaps the gated-MLA g_proj GEMM with the Q/KV projections.

    Ports the vLLM-HCU PCP+EP ``gate_proj`` overlap: the attention output-gate
    projection (``linear_gate``, a.k.a. "g_proj") runs on a dedicated side
    stream concurrently with the Q/KV projection GEMMs, and the join is deferred
    until after the attention core -- which is where sglang applies the gate
    anyway (``forward_absorb_core``). Gated on expert parallelism being active,
    the regime where the extra projection GEMM is large enough for the overlap
    to pay off (notably prefill); a no-op otherwise.
    """
    return (
        envs.SGLANG_ENABLE_HYV4_GATE_PROJ_PARALLEL.get()
        and get_parallel().moe_ep_size > 1
    )



def _hpc_ihc_available(op_name: str, hc_mult: int, hidden_size: int) -> bool:
    try:
        from sglang.kernels.ops.layernorm.hy4_ihc import _hpc_ihc_op
    except ImportError:
        return False
    return _hpc_ihc_op(op_name, hc_mult, hidden_size) is not None


def normalize_hyv4_weight_name(name: str) -> str:
    """Accept component-suffixed HYV4 exports alongside original weight names.

    This changes names only; packed INT4 bytes and scales are not converted.
    Direct Parameter attributes must not acquire a Linear-style weight suffix.
    """
    name = name.replace(".self_attn.g_proj.", ".self_attn.linear_gate.")
    for source, target in (
        (".weight.weight", ".weight"),
        (".bias.weight", ".bias"),
        (".weight.packed", ".weight"),
        (".weight.scale", ".weight_scale"),
    ):
        if name.endswith(source):
            return name.removesuffix(source) + target
    if name.endswith(
        (
            ".learnable_sink_param.weight",
            ".e_score_correction_bias.weight",
            ".hc_base.weight",
            ".hc_scale.weight",
            ".hc_head_base.weight",
            ".hc_head_scale.weight",
        )
    ):
        return name.removesuffix(".weight")
    return name


def adapt_hyv4_packed_weight(name, weight, quant_config):
    """Expose CT int32 storage as byte pairs; normalize values after EP slicing."""
    if (
        getattr(quant_config, "checkpoint_format", None)
        != "hy4_compressed_tensors_w4a8"
    ):
        return name, weight
    if name.endswith(".weight_shape"):
        return None, None
    if name.endswith(".weight_packed"):
        if (
            ".mlp.experts." not in name
            or weight.dtype != torch.int32
            or weight.ndim != 2
        ):
            raise ValueError(f"Expected a packed INT32 HYV4 expert matrix: {name}")
        return name.removesuffix("_packed"), weight.contiguous().view(torch.int8)
    return name, weight


def permute_hyv4_indexer_weight(name, loaded_weight, config):
    if ".self_attn.indexer.wq_b." in name:
        group_count = config.index_n_heads
    elif any(
        key in name
        for key in (
            ".self_attn.indexer.wk.",
            ".self_attn.indexer.k_norm.",
        )
    ):
        group_count = 1
    else:
        return loaded_weight

    shape = loaded_weight.shape
    loaded_weight = loaded_weight.reshape(
        group_count,
        config.index_head_dim,
        *shape[1:],
    )
    rope_dim = config.qk_rope_head_dim
    return torch.cat(
        (
            loaded_weight[:, -rope_dim:],
            loaded_weight[:, :-rope_dim],
        ),
        dim=1,
    ).reshape(shape)


def hyv4_linear_scale_suffix(model: nn.Module) -> str:
    """Suffix to append to a checkpoint ``*.weight_scale`` name.

    The HYV4 checkpoints always name linear weight scales ``weight_scale``, but
    fp8 block quantization registers them as ``weight_scale_inv`` while the
    per-channel int4/int8 methods keep ``weight_scale``.
    """
    for name, _ in model.named_parameters():
        if name.endswith(".weight_scale_inv"):
            return "_inv"
    return ""


class HYV4HCPreLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.magnitude = config.hc_magnitude
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_fn = ReplicatedLinear(
            config.hidden_size * config.hc_mult,
            2 * config.hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.hc_fn",
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(
            torch.empty(2 * config.hc_mult, dtype=torch.float32)
        )
        self._fused_ihc_pre_disabled = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        rms_weight: torch.Tensor | None = None,
        rms_eps: float = 0.0,
    ):
        if rms_weight is None and envs.SGLANG_OPT_HY4_IHC_TILELANG.get():
            # Opt-in HCU TileLang iHC (SGLANG_OPT_HY4_IHC_TILELANG); it does not
            # fuse the RMSNorm, so it only serves the unfused-norm call.
            fused = try_tilelang_ihc_pre(
                hidden_states,
                self.hc_fn.weight,
                self.hc_scale,
                self.hc_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.magnitude,
            )
            if fused is not None:
                return fused
        if hidden_states.is_cuda and not self._fused_ihc_pre_disabled:
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import fused_hy4_ihc_pre
            except ImportError:
                pass
            else:
                try:
                    return fused_hy4_ihc_pre(
                        hidden_states,
                        self.hc_fn.weight,
                        self.hc_scale,
                        self.hc_base,
                        self.magnitude,
                        self.rms_norm_eps,
                        self.hc_eps,
                        rms_weight,
                        rms_eps,
                    )
                except Exception:
                    self._fused_ihc_pre_disabled = True
                    logger.warning(
                        "fused_hy4_ihc_pre failed, disabling fused path",
                        exc_info=True,
                    )
        shape = hidden_states.shape
        flat = hidden_states.flatten(1).float()
        scale = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        gates = self.hc_fn(flat)[0] * scale
        pre = (
            torch.sigmoid(
                gates[..., : self.hc_mult] * self.hc_scale[0]
                + self.hc_base[: self.hc_mult]
            )
            + self.hc_eps
        )
        post = (
            self.magnitude
            * torch.sigmoid(
                gates[..., self.hc_mult :] * self.hc_scale[1]
                + self.hc_base[self.hc_mult :]
            )
            + self.hc_eps
        )
        reduced = torch.sum(pre.unsqueeze(-1) * hidden_states.reshape(shape), dim=1)
        reduced = reduced.to(hidden_states.dtype)
        if rms_weight is not None:
            reduced_float = reduced.float()
            reduced = (
                reduced_float
                * torch.rsqrt(
                    reduced_float.square().mean(dim=-1, keepdim=True) + rms_eps
                )
                * rms_weight.float()
            ).to(hidden_states.dtype)
        return reduced, post


class HYV4HCLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_pre = HYV4HCPreLayer(config, f"{prefix}.hc_pre")
        self._fused_ihc_post_disabled = False
        self._fused_ihc_post_pre_disabled = False

    def prepare_input(self, hidden_states: torch.Tensor):
        if hidden_states.ndim == 3:
            return hidden_states
        if hidden_states.ndim != 2:
            raise RuntimeError(
                f"iHC expects a 2D or 3D tensor, got {hidden_states.shape}"
            )
        if hidden_states.shape[-1] == self.hidden_size:
            return hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        if hidden_states.shape[-1] == self.hidden_size * self.hc_mult:
            return hidden_states.reshape(-1, self.hc_mult, self.hidden_size)
        raise RuntimeError(
            "iHC input width must equal hidden_size or hc_mult * hidden_size"
        )

    def pre(self, hidden_states: torch.Tensor, norm: RMSNorm | None = None):
        fuse_norm = (
            norm is not None
            and hidden_states.is_cuda
            and _hpc_ihc_available("fuse_ihc_pre", self.hc_mult, self.hidden_size)
        )
        reduced, post = self.hc_pre(
            hidden_states,
            norm.weight if fuse_norm else None,
            norm.variance_epsilon if fuse_norm else 0.0,
        )
        if norm is not None and not fuse_norm:
            reduced = norm(reduced)
        return reduced, post, hidden_states

    def post(self, output, residual, post):
        fused = try_tilelang_ihc_post(output, residual, post)
        if fused is not None:
            return fused
        if output.is_cuda and not self._fused_ihc_post_disabled:
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import fused_hy4_ihc_post
            except ImportError:
                pass
            else:
                try:
                    return fused_hy4_ihc_post(output, residual, post)
                except Exception:
                    self._fused_ihc_post_disabled = True
                    logger.warning(
                        "fused_hy4_ihc_post failed, disabling fused path",
                        exc_info=True,
                    )
        result = post.float().unsqueeze(-1) * output.float().unsqueeze(1)
        return (result + residual.float()).to(output.dtype)

    def post_pre(self, output, residual, post, next_layer, norm):
        if (
            output.is_cuda
            and not self._fused_ihc_post_pre_disabled
            and _hpc_ihc_available("fuse_ihc_post_pre", self.hc_mult, self.hidden_size)
        ):
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import (
                    fused_hy4_ihc_post_pre,
                )
            except ImportError:
                pass
            else:
                try:
                    next_residual, reduced, next_post = fused_hy4_ihc_post_pre(
                        output,
                        residual,
                        post,
                        next_layer.hc_pre.hc_fn.weight,
                        next_layer.hc_pre.hc_scale,
                        next_layer.hc_pre.hc_base,
                        next_layer.hc_pre.magnitude,
                        next_layer.hc_pre.rms_norm_eps,
                        next_layer.hc_pre.hc_eps,
                        norm.weight,
                        norm.variance_epsilon,
                    )
                    return reduced, next_post, next_residual
                except Exception:
                    self._fused_ihc_post_pre_disabled = True
                    logger.warning(
                        "fused_hy4_ihc_post_pre failed, disabling fused path",
                        exc_info=True,
                    )
        next_residual = self.post(output, residual, post)
        next_residual = next_layer.prepare_input(next_residual)
        return next_layer.pre(next_residual, norm)


class HYV4HCHeadLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.config = config
        self.hc_head_fn = ReplicatedLinear(
            config.hc_mult * config.hidden_size,
            config.hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.hc_head_fn",
        )
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32)
        )
        self._fused_ihc_head_disabled = False

    def forward(self, hidden_states: torch.Tensor, norm: RMSNorm | None = None):
        fused = try_tilelang_ihc_head(
            hidden_states,
            self.hc_head_fn.weight,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        if fused is not None:
            return fused if norm is None else norm(fused)
        if (
            hidden_states.is_cuda
            and not self._fused_ihc_head_disabled
            and _hpc_ihc_available(
                "fuse_ihc_head", self.config.hc_mult, self.config.hidden_size
            )
        ):
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import fused_hy4_ihc_head
            except ImportError:
                pass
            else:
                try:
                    return fused_hy4_ihc_head(
                        hidden_states,
                        self.hc_head_fn.weight,
                        self.hc_head_scale,
                        self.hc_head_base,
                        self.config.rms_norm_eps,
                        self.config.hc_eps,
                        None if norm is None else norm.weight,
                        0.0 if norm is None else norm.variance_epsilon,
                    )
                except Exception:
                    self._fused_ihc_head_disabled = True
                    logger.warning(
                        "fused_hy4_ihc_head failed, disabling fused path",
                        exc_info=True,
                    )
        shape = hidden_states.shape
        flat = hidden_states.flatten(1).float()
        scale = torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        gates = self.hc_head_fn(flat)[0] * scale
        gates = (
            torch.sigmoid(gates * self.hc_head_scale + self.hc_head_base)
            + self.config.hc_eps
        )
        output = torch.sum(gates.unsqueeze(-1) * flat.reshape(shape), dim=1)
        output = output.to(hidden_states.dtype)
        return output if norm is None else norm(output)


class HYV4Attention(DeepseekV2AttentionMLA):
    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        gate_proj_stream=None,
        is_nextn=False,
        gate_a2a_stream=None,
    ):
        rope_parameters = config.rope_parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_parameters["rope_theta"],
            rope_scaling=(
                None
                if rope_parameters.get("rope_type") == "default"
                else rope_parameters
            ),
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            # The FFN input boundary completes this stage's sum.
            reduce_results=False,
            layer_id=layer_id,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
        )
        parallel = get_parallel()
        attn_tp_size = parallel.attn_tp_size
        gate_kwargs = dict(
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_gate",
        )
        full_gate_width = config.num_attention_heads * config.v_head_dim
        # linear_gate CP sharding: output rows split across attention-CP ranks,
        # rebuilt per forward by GateCPShardChain. Decided on the layer's
        # resolved quant method rather than quant_config, since W4A8 configs
        # still leave this projection unquantized.
        self.gate_cp_shard_enabled = False
        if (
            envs.SGLANG_HY4_LINEAR_GATE_CP_SHARD.get()
            and parallel.attn_cp_size > 1
            and attn_tp_size == 1
            and not parallel.dcp_enabled
            and full_gate_width % parallel.attn_cp_size == 0
        ):
            self.linear_gate = ColumnParallelLinear(
                config.hidden_size,
                full_gate_width,
                parallel_group="attn_cp",
                **gate_kwargs,
            )
            if isinstance(self.linear_gate.quant_method, UnquantizedLinearMethod):
                self.gate_cp_shard_enabled = True
                self._gate_cp_runtime = get_gate_cp_shard_runtime(
                    self.linear_gate.weight
                )
                # Bound by the owning model's GateCPShardChain.
                self._gate_cp_chain = None
                self._gate_cp_index = None
            else:
                del self.linear_gate
        if not self.gate_cp_shard_enabled:
            self.linear_gate = ColumnParallelLinear(
                config.hidden_size,
                full_gate_width,
                parallel_group="attn_tp",
                **gate_kwargs,
            )
        self.local_gate_width = self.num_local_heads * config.v_head_dim
        gate_width = (
            full_gate_width
            if self.gate_cp_shard_enabled
            else self.linear_gate.output_size_per_partition
        )
        if gate_width != self.local_gate_width:
            raise ValueError(
                "HYV4 attention gate shard width must match the local attention "
                f"output width: {gate_width} != {self.local_gate_width}"
            )
        # Replicating these small logits avoids a per-layer DCP all-gather.
        # The much larger output gate remains attention-TP column-sharded.
        self.learnable_sink_param = nn.Parameter(
            torch.empty(config.num_attention_heads, dtype=torch.float32)
        )
        self.learnable_sink_param.weight_loader = self._sink_weight_loader
        self._gate_fallback_backend = self._resolve_gate_fallback_backend()
        self._gate_backend = self._resolve_gate_backend(
            getattr(config, "gating_type", None), self._gate_fallback_backend
        )
        if prefix.endswith(".0.self_attn"):
            logger.info("HYV4 MLA output gate backend: %s", self._gate_backend)
        self.attn_mqa.hyv4_sink_getter = self.get_local_attention_sink
        if self.dcp_size > 1:
            self.attn_mqa_for_dcp_decode.hyv4_sink_getter = self.get_dcp_attention_sink

        # Attention DCP A2A/gate overlap is independent of MoE SBO.
        self._hyv4_gate_a2a_enabled = False
        self._hyv4_gate_a2a_max_tokens = 0
        self._hyv4_gate_a2a_state = None
        if envs.SGLANG_HY4_DCP2_GATE_A2A_OVERLAP.get():
            if envs.SGLANG_ENABLE_HYV4_GATE_PROJ_PARALLEL.get():
                raise ValueError(
                    "HYV4 gate/A2A overlap and gate/QKV overlap are mutually exclusive"
                )
            max_tokens = envs.SGLANG_HY4_DCP2_GATE_A2A_MAX_TOKENS.get()
            if max_tokens < 0:
                raise ValueError(
                    "HYV4 gate/A2A max tokens must be nonnegative (0 = unlimited)"
                )
            if (
                is_hcu()
                and get_parallel().dcp_enabled
                and get_parallel().attn_dcp_size in (2, 4)
                and get_parallel().dcp_comm_backend == "a2a"
                and envs.SGLANG_HY4_DCP2_FUSED_POST_ATTN.get()
                and not self.gate_cp_shard_enabled
                and not is_nextn
                and gate_a2a_stream is not None
            ):
                from sglang.srt.layers.hyv4_gate_a2a_overlap import GateA2AState

                # Events belong to this layer; the dedicated stream is shared
                # by the target model. NEXTN retains its original schedule.
                self._hyv4_gate_a2a_state = GateA2AState(gate_a2a_stream)
                self._hyv4_gate_a2a_enabled = True
                self._hyv4_gate_a2a_max_tokens = max_tokens

        # Optional g_proj/projection overlap stream (see
        # ``enable_hyv4_gate_proj_parallel``). Created once at the model level
        # and shared by every layer (which run sequentially), so the overlap
        # uses exactly one side stream -- not one per layer. Kept distinct from
        # the parent's ``alt_stream`` so it never contends with the qk-norm /
        # q_b_proj + indexer overlaps already running there.
        # ``forward_absorb_prepare`` launches ``linear_gate`` on this stream
        # before the Q/KV projections; ``forward_absorb_core`` joins it after
        # attention, before the gate is applied.
        self.gate_proj_stream = gate_proj_stream
        self.gate_proj_start_event = (
            torch.cuda.Event() if gate_proj_stream is not None else None
        )
        self.gate_proj_done_event = (
            torch.cuda.Event() if gate_proj_stream is not None else None
        )

    @staticmethod
    def _sink_weight_loader(param, loaded_weight):
        if param.shape != loaded_weight.shape:
            raise ValueError("HYV4 sink weights must contain one logit per head")
        param.data.copy_(loaded_weight.float())

    def get_local_attention_sink(self):
        start = get_parallel().attn_tp_rank * self.num_local_heads
        return self.learnable_sink_param[start : start + self.num_local_heads]

    def get_dcp_attention_sink(self):
        parallel = get_parallel()
        heads = self.num_local_heads * parallel.attn_dcp_size
        start = (parallel.attn_tp_rank // parallel.attn_dcp_size) * heads
        return self.learnable_sink_param[start : start + heads]

    @staticmethod
    def _resolve_gate_fallback_backend() -> str:
        try:
            from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (  # noqa: F401
                sigmoid_gate_mul,
            )
        except ImportError:
            return "eager"
        return "triton"

    def _resolve_gate_backend(
        self, gating_type: str | None, fallback_backend: str
    ) -> str:
        weight = getattr(self.linear_gate, "weight", None)
        if self._hpc_gated_mla_supported(
            gating_type,
            None if weight is None else weight.dtype,
            None if weight is None else tuple(weight.shape),
            self.local_gate_width,
            self.hidden_size,
        ):
            return "hpc"
        return fallback_backend

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _hpc_gated_mla_supported(
        gating_type: str | None,
        weight_dtype: torch.dtype | None,
        weight_shape: tuple[int, ...] | None,
        local_gate_width: int,
        hidden_size: int,
    ) -> bool:
        try:
            import hpc
        except ImportError:
            return False

        if not hasattr(getattr(hpc, "gemm", None), "gated_mla_gemm"):
            logger.info(
                "HY4 gated MLA: the installed hpc build (%s) has no "
                "gemm.gated_mla_gemm; falling back.",
                getattr(hpc, "__version__", "unknown"),
            )
            return False

        major, minor = get_device_capability()
        if (major, minor) not in ((10, 0), (10, 3)):
            logger.warning(
                "HY4 gated MLA: hpc.gated_mla_gemm is built for sm100/sm103, "
                "got sm%d%d; falling back.",
                major,
                minor,
            )
            return False

        # Headwise gating broadcasts one scalar per head over v_head_dim; the
        # kernel only implements the elementwise product.
        if gating_type != "elementwise":
            return False

        if weight_dtype != torch.bfloat16:
            logger.warning(
                "HY4 gated MLA: gate weight dtype is %s, the kernel needs "
                "bfloat16 (keep linear_gate out of quantization).",
                weight_dtype,
            )
            return False
        expected_shape = (local_gate_width, hidden_size)
        if weight_shape != expected_shape:
            logger.warning(
                "HYV4 gated MLA: local gate weight shape is %s, expected %s.",
                weight_shape,
                expected_shape,
            )
            return False
        if local_gate_width % 256 != 0:
            logger.warning(
                "HY4 gated MLA: local gate width %s is not a multiple of 256.",
                local_gate_width,
            )
            return False
        return True

    def _hpc_gated_mla_inputs_supported(self, hidden_states, attn_out) -> bool:
        weight = self.linear_gate.weight
        return (
            hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and attn_out.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and weight.ndim == 2
            and attn_out.ndim == 2
            and hidden_states.shape[0] == attn_out.shape[0]
            and hidden_states.shape[1] == weight.shape[1]
            and weight.shape[0] == self.local_gate_width
            and attn_out.shape[1] == self.local_gate_width
            and hidden_states.device == weight.device == attn_out.device
        )

    @staticmethod
    def _apply_attention_output_gate_fallback(attn_out, gate, backend):
        if gate.shape != attn_out.shape:
            raise ValueError(
                "HYV4 projected attention gate shape must match the local attention "
                f"output shape: {tuple(gate.shape)} != {tuple(attn_out.shape)}"
            )
        if backend == "triton" and attn_out.is_cuda:
            from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
                sigmoid_gate_mul,
            )

            return sigmoid_gate_mul(attn_out, gate)
        return attn_out * torch.sigmoid(gate)

    def prepare_attention_output_gate(self, hidden_states):
        if self.gate_cp_shard_enabled:
            if self._gate_cp_chain is None:
                raise RuntimeError("HYV4 CP-sharded linear_gate is not bound to a model")
            return self._gate_cp_chain.gate(self._gate_cp_index, hidden_states)
        # The hpc kernel fuses the projection in, so hand it the activations and
        # let apply_attention_output_gate do the GEMM. The other two tiers want
        # the projected gate.
        if self._gate_backend == "hpc":
            return hidden_states

        from sglang.srt.layers.hyv4_tuned_gate import try_tuned_gate

        tuned = try_tuned_gate(self.linear_gate, hidden_states)
        if tuned is not None:
            return tuned
        return self.linear_gate(hidden_states)[0]

    def apply_attention_output_gate(self, attn_out, gate):
        if self._gate_backend == "hpc" and self._hpc_gated_mla_inputs_supported(
            gate, attn_out
        ):
            from hpc.gemm import gated_mla_gemm

            # linear_gate is column-parallel and unbiased, so the local weight
            # shard lines up with the local attn_out columns.
            return gated_mla_gemm(
                gate.contiguous(), self.linear_gate.weight, attn_out.contiguous()
            )
        if self._gate_backend == "hpc":
            gate = self.linear_gate(gate)[0]
            backend = self._gate_fallback_backend
        else:
            backend = self._gate_backend
        return self._apply_attention_output_gate_fallback(attn_out, gate, backend)

    def dispatch_attn_forward_method(self, forward_batch):
        backend = get_attn_backend()
        backend = getattr(backend, "primary", backend)
        if hasattr(backend, "use_mha") and backend.use_mha is not False:
            backend.use_mha = False
        method = super().dispatch_attn_forward_method(forward_batch)
        # HIP (HCU) maps MLA onto its ROCm implementation of the same path.
        if method not in (AttnForwardMethod.MLA, AttnForwardMethod.MLA_ROCM):
            raise RuntimeError(
                f"HYV4 requires the sparse MLA attention path, got {method}"
            )
        return method


class _HeadNorm:
    """The layer stack's terminal read: the learned head mixes the streams down
    and fuses the final norm into the same kernel."""

    def __init__(self, hc_head, norm):
        self.hc_head = hc_head
        self.norm = norm

    def __call__(self, hidden_states):
        return self.hc_head(hidden_states, self.norm)


def _build_stages(
    hc_attn_layer,
    hc_mlp_layer,
    input_layernorm,
    post_attention_layernorm,
    *,
    qkv_latent_func,
    layer_id,
    config,
):
    """The attention and FFN stage boundaries of one iHC decoder layer."""

    def post_pre(hidden_states, residual, post_gate, out_norm):
        return hc_attn_layer.post_pre(
            hidden_states, residual, post_gate, hc_mlp_layer, out_norm
        )

    residual = IHCState(
        expand=hc_attn_layer.prepare_input,
        attn_pre=hc_attn_layer.pre,
        ffn_pre=hc_mlp_layer.pre,
        attn_post=hc_attn_layer.post,
        ffn_post=hc_mlp_layer.post,
        post_pre=post_pre,
    ).residual_ops()

    attn, ffn = _declare_stages(config, layer_id, residual)
    return append_stages(
        (attn, input_layernorm, {"qkv_latent_func": qkv_latent_func}),
        (ffn, post_attention_layernorm),
    )


def _declare_stages(config, layer_id, residual):
    layer_types = config.mlp_layer_types
    next_id = layer_id + 1
    return (
        declare_attn(read=residual.attn_readout, update=residual.attn_update),
        declare_ffn(
            sparse=layer_types[layer_id] != "dense",
            next_layer_sparse=(
                next_id < len(layer_types) and layer_types[next_id] != "dense"
            ),
            read=residual.ffn_readout,
            update=residual.ffn_update,
        ),
    )


def _stage_facts(config, layer_id):
    """The stages the layer at ``layer_id`` declares, from the config alone:
    the model's shared declaration function (see make_layers). The iHC
    callables are never run; only the read/update declarations are compared."""
    residual = IHCState(
        expand=None, attn_pre=None, ffn_pre=None, attn_post=None, ffn_post=None
    ).residual_ops()
    return _declare_stages(config, layer_id, residual)


class HYV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        gate_proj_stream=None,
        gate_a2a_stream=None,
    ):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            layer_id,
            quant_config,
            f"{prefix}.self_attn",
            alt_stream,
            gate_proj_stream,
            gate_a2a_stream=gate_a2a_stream,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[layer_id] == "dense":
            self.mlp = DeepseekV2MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=False,
                allow_fused_down=False,
            )
        else:
            # DCP4 without SBO keeps both MoE branches on the caller stream.
            # Passing alt_stream otherwise enables shared-expert overlap even
            # with SBO disabled. Attention retains its independent streams,
            # including the dedicated gate/A2A fork above.
            moe_alt_stream = alt_stream
            if (
                is_hcu()
                and get_parallel().dcp_enabled
                and get_parallel().attn_dcp_size == 4
                and not get_global_server_args().enable_single_batch_overlap
            ):
                moe_alt_stream = None
            self.mlp = DeepseekV2MoE(
                config,
                layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                alt_stream=moe_alt_stream,
                reduce_results=False,
            )
            if hasattr(self.mlp, "shared_experts"):
                self.mlp.shared_experts.swiglu_limit = None
        self.hc_attn_layer = HYV4HCLayer(config, f"{prefix}.hc_attn_layer")
        self.hc_mlp_layer = HYV4HCLayer(config, f"{prefix}.hc_mlp_layer")
        self.attn_boundary, self.ffn_boundary = _build_stages(
            self.hc_attn_layer,
            self.hc_mlp_layer,
            self.input_layernorm,
            self.post_attention_layernorm,
            qkv_latent_func=self.self_attn.prepare_qkv_latent,
            layer_id=layer_id,
            config=config,
        )

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
        next_full_attention_layer_id: Optional[int] = None,
    ):
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        try:
            hidden_states = self.self_attn(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=prev_topk_indices,
            )
        finally:
            get_attn_tp_context().clear_attn_inputs()
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        # Enqueue the next layer's owner-broadcast (Index-K and Main-KV) on the
        # side stream while this layer's reduce-scatter and MLP run.
        maybe_prefetch_next_full_attention_kv(
            forward_batch, next_full_attention_layer_id
        )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        if isinstance(self.mlp, DeepseekV2MoE):
            hidden_states = self.mlp(hidden_states, forward_batch)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)
        return hidden_states, topk_indices


class HYV4Model(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.pp_group = get_parallel().pp_group
        self.config = config
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
                **get_embedding_tp_kwargs(),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.alt_stream = (
            get_stream("alt")
            if is_cuda() or envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
            else None
        )
        # One shared g_proj side stream for the whole stack (see
        # ``enable_hyv4_gate_proj_parallel``); created here, not per layer, so
        # the overlap does not spawn num_layers streams.
        self.gate_proj_stream = (
            get_stream("hyv4_gate_proj") if enable_hyv4_gate_proj_parallel() else None
        )
        if self.gate_proj_stream is not None:
            logger.info(
                "HYV4 gated-MLA g_proj/projection overlap enabled "
                "(shared side stream across all layers)"
            )
        # Separate from alt_stream (QK norm / MoE SBO) and from the older
        # early gate/QKV schedule. Allocate once, before graph capture.
        parallel = get_parallel()
        self.gate_a2a_stream = (
            get_stream("hyv4_gate_a2a")
            if (
                envs.SGLANG_HY4_DCP2_GATE_A2A_OVERLAP.get()
                and _is_hcu
                and parallel.dcp_enabled
                and parallel.attn_dcp_size in (2, 4)
                and parallel.dcp_comm_backend == "a2a"
                and envs.SGLANG_HY4_DCP2_FUSED_POST_ATTN.get()
            )
            else None
        )
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: HYV4DecoderLayer(
                config,
                idx,
                quant_config,
                prefix,
                self.alt_stream,
                self.gate_proj_stream,
                gate_a2a_stream=self.gate_a2a_stream,
            ),
            prefix=f"{prefix}.layers",
            stage_facts=functools.partial(_stage_facts, config),
        )

        if self.gate_a2a_stream is not None:
            configured = sum(
                getattr(
                    getattr(layer, "self_attn", None), "_hyv4_gate_a2a_enabled", False
                )
                for layer in self.layers[self.start_layer : self.end_layer]
            )
            logger.info(
                "HYV4 gate/A2A side-stream schedule: configured_layers=%d "
                "max_local_rows=%d (runtime eligibility still applies)",
                configured,
                envs.SGLANG_HY4_DCP2_GATE_A2A_MAX_TOKENS.get(),
            )

        self.gate_cp_chain = build_gate_cp_chain(
            [
                getattr(layer, "self_attn", None)
                for layer in self.layers[self.start_layer : self.end_layer]
            ]
        )
        # hc_head collapses the 3D iHC stream back to (T, H); only the last
        # stage produces a model output, so only it needs the head and norm.
        if self.pp_group.is_last_rank:
            self.hc_head = HYV4HCHeadLayer(config, f"{prefix}.hc_head")
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            # The head mixes the streams down and fuses the final norm into one
            # kernel.
            self._head_norm = _HeadNorm(self.hc_head, self.norm)
        else:
            self.hc_head = PPMissingLayer()
            self.norm = PPMissingLayer()
            self._head_norm = None
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.cp_rank = parallel.attn_cp_rank
        self.cp_size = parallel.attn_cp_size
        # The final local layer maps to no next layer; a later PP stage
        # prefetches its own first layer on model entry.
        local_layer_ids = list(range(self.start_layer, self.end_layer))
        self.next_full_attention_layer_id = dict(
            zip(local_layer_ids, local_layer_ids[1:])
        )

    def _dsa_forward_uses_topk(self) -> bool:
        """Does this batch's attention backend consume DSA top-k indices?

        Mirrors ``DeepseekV2Model._dsa_forward_uses_topk``: an MHA backend runs
        dense attention and never asks the indexer for top-k, so there is
        nothing to carry across a PP boundary.
        """
        backend = get_attn_backend()
        backend = getattr(backend, "primary", backend)
        return not getattr(backend, "use_mha", False)

    def _maybe_prepare_prefill_cp(self, num_tokens, forward_batch):
        """Set ``attn_cp_metadata`` so the CP-v1 prefill data split can run.

        ``num_tokens`` is the pre-split token count. Under PP the later stages
        have no ``input_ids`` and their incoming hidden state is already
        CP-split, so the caller derives it from ``extend_seq_lens_cpu``, which
        is the full length on every rank.

        The model runner prepares and gathers only for CP v2 and clears
        ``attn_cp_metadata`` otherwise, so the CP-v1 boundary is owned here.
        ``dsa_use_prefill_cp`` returns False while ``attn_cp_metadata`` is None,
        so ``forward``'s ``cp_split_and_rebuild_data`` is gated on this.

        Attention/indexer metadata is not reindexed here: the DSA backend's
        ``init_forward_metadata`` already builds rank-local metadata for the
        round-robin split. The backend owns the metadata split; the model owns
        only the data split.
        """
        if not (
            self.dsa_enable_prefill_cp and forward_batch.extend_seq_lens_cpu is not None
        ):
            return False
        if is_cp_v2_active(forward_batch):
            return False
        if not can_dsa_cp_split(num_tokens, self.cp_size, True, forward_batch):
            return False
        forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
            num_tokens,
            self.cp_rank,
            self.cp_size,
            forward_batch.seq_lens_cpu.tolist(),
            extend_seqs_len=forward_batch.extend_seq_lens_cpu,
        )
        return True

    @scoped_hyv4_dcp_write_locs
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        dsa_forward_uses_topk = self._dsa_forward_uses_topk()
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual_batch.start(forward_batch)
            initial_topk_indices = None
        else:
            assert pp_proxy_tensors is not None
            received = pp_proxy_tensors["hidden_states"]
            # The iHC stream crosses PP flattened to 2D (see the send side), so
            # restore the (T, hc_mult, H) layout the layers expect.
            if received.ndim == 2:
                received = received.view(
                    received.shape[0], self.config.hc_mult, self.config.hidden_size
                )
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                PPProxyTensors({"hidden_states": received}), forward_batch
            )
            initial_topk_indices = pp_proxy_tensors.tensors.get("topk_indices")
        # Shared-indexer layers reuse the previous layer's topk. Across a PP
        # boundary the previous stage's last topk arrives via the proxy.
        index_topk_share = IndexTopKShareState(forward_batch, initial_topk_indices)
        if not self.pp_group.is_first_rank:
            assert not (
                not forward_batch.forward_mode.is_idle()
                and hidden_states.shape[0] != 0
                and dsa_forward_uses_topk
                and dsa_layer_skips_topk(self.config, self.start_layer)
                and index_topk_share.topk_indices is None
            ), (
                f"PP stage starting at layer {self.start_layer} requires DSA "
                "topk_indices from the previous stage."
            )
        # The CP-v1 prefill boundary sits at the model level on purpose. iHC
        # carries a 3D (T, hc_mult, H) residual between layers, which the 2D CP
        # helpers cannot describe, but both the embedding output and the hc_head
        # output are flat (T, H). Splitting here and gathering after hc_head
        # keeps the iHC stream entirely CP-local. CP v2 shards and gathers at
        # the eager-runner boundary instead.
        #
        # Under PP only the first stage holds input_ids, and later stages
        # receive an already-split hidden state, so take the pre-split length
        # from extend_seq_lens_cpu -- it is the full length on every rank.
        if input_ids is not None:
            num_tokens = input_ids.shape[0]
        elif forward_batch.extend_seq_lens_cpu is not None:
            num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            num_tokens = hidden_states.shape[0]
        self._maybe_prepare_prefill_cp(num_tokens, forward_batch)
        use_cp = (
            self.dsa_enable_prefill_cp
            and dsa_use_prefill_cp(forward_batch)
            and not is_cp_v2_active(forward_batch)
        )
        if use_cp:
            if not getattr(self, "_cp_logged", False):
                self._cp_logged = True
                logger.info(
                    "HYV4 prefill CP engaged: tokens=%d cp_size=%d",
                    num_tokens,
                    self.cp_size,
                )
            # The first stage splits and every later stage inherits that split,
            # but positions arrive full-length on all of them.
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
        zero_allocator = BumpAllocator(
            buffer_size=2 * (self.end_layer - self.start_layer),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        if self.gate_cp_chain is not None:
            self.gate_cp_chain.begin(use_cp, hidden_states.shape[0])
        if _is_hcu and self.start_layer < self.end_layer:
            # A non-zero PP stage may start on a skip-topk layer, so no local
            # indexer would trigger the first Main-KV prefetch. Install this
            # batch's compact page plan and prefetch the first local layer.
            maybe_prefetch_full_attention_kv(forward_batch, self.start_layer)
        else:
            maybe_configure_main_kv_page_plan(forward_batch)
        try:
            for i in range(self.start_layer, self.end_layer):
                # NOTE: torch dynamo does not support graph break in context manager
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    hidden_states, topk_indices = self.layers[i](
                        positions,
                        hidden_states,
                        forward_batch,
                        zero_allocator,
                        prev_topk_indices=index_topk_share.topk_indices,
                        next_full_attention_layer_id=self.next_full_attention_layer_id.get(
                            i
                        ),
                    )
                index_topk_share.update(topk_indices)
        finally:
            if self.gate_cp_chain is not None:
                self.gate_cp_chain.end()
        index_topk_share.publish()
        if not self.pp_group.is_last_rank:
            proxy_tensors = residual_batch.to_pp(hidden_states, forward_batch)
            # Flatten the 3D iHC stream for PP IPC, as DeepseekV4Model does for
            # its mHC stream: the transport buffers are 2D.
            sent = proxy_tensors["hidden_states"]
            if sent.ndim == 3:
                proxy_tensors.tensors["hidden_states"] = sent.flatten(1)
            if (
                dsa_forward_uses_topk
                and self.end_layer < self.config.num_hidden_layers
                and dsa_layer_skips_topk(self.config, self.end_layer)
            ):
                topk_indices = index_topk_share.topk_indices
                if not forward_batch.forward_mode.is_idle() and sent.shape[0] != 0:
                    assert topk_indices is not None, (
                        f"PP stage ending at layer {self.end_layer} must forward "
                        "DSA topk_indices because the next stage starts on a "
                        "skip-topk layer."
                    )
                if topk_indices is None:
                    topk_indices = sent.new_empty(
                        (0, get_dsa_index_topk(self.config)), dtype=torch.int32
                    )
                proxy_tensors.tensors["topk_indices"] = topk_indices
            return proxy_tensors
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if forward_batch.forward_mode.is_idle():
            # Collapse the 3D iHC stream to the 2D model output without running
            # hc_head / norm on a zero-token batch.
            residual_batch.clear(forward_batch)
            return hidden_states[:, 0]
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self._head_norm
        )
        if use_cp:
            gather = cp_all_gather_rerange_output
            if _is_hcu:
                gather = functools.partial(gather, chunk_large_output=True)
            hidden_states = gather(
                hidden_states.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return hidden_states


class HYV4ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        return "HYV4 applies the SwiGLU limit only to routed experts."

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        if (
            quant_config is not None
            and quant_config.get_name() == "slimquant_w4a8_marlin"
            and getattr(quant_config, "checkpoint_format", None)
            not in ("hy4_w4a8_v1", "hy4_compressed_tensors_w4a8")
        ):
            raise ValueError(
                "HY4 W4A8 requires quantization_config.checkpoint_format="
                "hy4_w4a8_v1 or the supported HYV4 compressed-tensors W4A8 scheme"
            )
        validate_hyv4_launch(get_global_server_args(), get_parallel())
        get_attn_tp_context().init_context(config.q_lora_rank, True)
        self.model = HYV4Model(config, quant_config, f"{prefix}.model")
        self.num_fused_shared_experts = max(
            (
                layer.mlp.num_fused_shared_experts
                for layer in self.model.layers
                # Under PP, make_layers() pads self.layers with PPMissingLayer
                # placeholders for the other stages' layers; those have no .mlp.
                if isinstance(layer, HYV4DecoderLayer)
                and isinstance(layer.mlp, DeepseekV2MoE)
            ),
            default=0,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            # Keep checkpoint weights in bf16; LogitsProcessor emits fp32
            # logits when config.enable_lm_head_fp32 is set.
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.logits_processor.use_fp32_lm_head = config.enable_lm_head_fp32

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            # Already a PPProxyTensors; the scheduler sends it to the next stage.
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    prepare_context_parallel_metadata_for_dcp = (
        DeepseekV2ForCausalLM.prepare_context_parallel_metadata_for_dcp
    )
    get_model_config_for_expert_location = classmethod(
        DeepseekV2ForCausalLM.get_model_config_for_expert_location.__func__
    )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        scale_suffix = hyv4_linear_scale_suffix(self)

        def mapped_weights():
            for name, loaded_weight in weights:
                name, loaded_weight = adapt_hyv4_packed_weight(
                    name, loaded_weight, self.quant_config
                )
                if name is None:
                    continue
                if (
                    name.endswith(".weight.packed")
                    and getattr(self.quant_config, "checkpoint_format", None)
                    != "hy4_w4a8_v1"
                ):
                    raise ValueError(
                        "Packed HY4 weights require quantization_config."
                        "checkpoint_format=hy4_w4a8_v1; do not infer nibble order"
                    )
                name = normalize_hyv4_weight_name(name)
                if name.startswith(("model.mtp_layers.", "model.mtp.layers.")):
                    continue
                loaded_weight = permute_hyv4_indexer_weight(
                    name, loaded_weight, self.config
                )
                if name.endswith((".hc_fn", ".hc_head_fn")):
                    name += ".weight"
                if name.endswith(".weight_scale"):
                    name += scale_suffix
                yield name, loaded_weight

        self.do_load_weights(mapped_weights())
        from sglang.srt.layers.hyv4_tuned_gate import prepare_model_gates

        prepare_model_gates(self)


EntryClass = [HYV4ForCausalLM]
