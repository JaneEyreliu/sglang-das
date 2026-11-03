import copy
from typing import Iterable, Tuple

import torch
from torch import nn

from sglang.srt.configs.hy_v4_mtp import (
    normalize_hyv4_mtp_weight_name,
    validate_hyv4_mtp_metadata,
)
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.hy4_dcp import validate_hyv4_launch
from sglang.srt.layers.hyv4_gate_cp_shard import build_gate_cp_chain
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.cp.utils import is_cp_v2_active
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
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.models.hunyuan_v4 import (
    HYV4Attention,
    HYV4ForCausalLM,
    adapt_hyv4_packed_weight,
    hyv4_linear_scale_suffix,
    permute_hyv4_indexer_weight,
)
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator, is_cuda


def _gather_dsa_topk_indices_for_cp(
    topk_indices, local_num_tokens, cp_size, forward_batch, stream
):
    """Gather CP-local DSA top-k rows back to the original token order."""
    if (
        is_dsa_prefill_cp_round_robin_split()
        and topk_indices.shape[0] < local_num_tokens
    ):
        pad_rows = local_num_tokens - topk_indices.shape[0]
        topk_indices = torch.cat(
            [
                topk_indices,
                topk_indices.new_full((pad_rows, topk_indices.shape[1]), -1),
            ],
            dim=0,
        )
    return cp_all_gather_rerange_output(
        topk_indices, cp_size, forward_batch, stream
    )


def _split_hyv4_nextn_cp_inputs(
    forward_batch, hidden_states, target_hidden_states, positions
):
    """Apply one CP token layout to both MTP input streams and positions."""
    hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
    target_hidden_states = cp_split_and_rebuild_data(
        forward_batch, target_hidden_states
    )
    positions = cp_split_and_rebuild_position(forward_batch, positions)
    return hidden_states, target_hidden_states, positions


def _restore_hyv4_nextn_cp_output(hidden_states, cp_size, forward_batch):
    """Gather and restore original token order before draft logits."""
    return cp_all_gather_rerange_output(
        hidden_states.contiguous(),
        cp_size,
        forward_batch,
        torch.cuda.current_stream(),
    )


def _mtp_quant_config(quant_config):
    if quant_config is None:
        return None
    quant_config = copy.deepcopy(quant_config)
    decoder_prefix = "model.decoder"

    def normalize_name(name):
        for mtp_prefix in (
            "model.mtp.layers.0",
            "model.mtp_layers.0",
            "mtp.layers.0",
            "mtp_layers.0",
        ):
            for special in ("enorm", "hnorm", "eh_proj", "shared_head.norm"):
                name = name.replace(mtp_prefix + "." + special, "model." + special)
            name = name.replace(mtp_prefix, decoder_prefix)
        name = name.replace(".self_attn.g_proj", ".self_attn.linear_gate")
        return name

    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if ignored_layers is not None:
        quant_config.ignored_layers = list(
            dict.fromkeys(normalize_name(name) for name in ignored_layers)
        )

    ignored_modules = getattr(quant_config, "ignore", None)
    if ignored_modules is not None:
        quant_config.ignore = list(
            dict.fromkeys(normalize_name(name) for name in ignored_modules)
        )

    # Compressed-tensors applies this override before consulting its ignore list.
    if hasattr(quant_config, "linear_fp8_config"):
        quant_config.linear_fp8_config = None

    return quant_config


class HYV4MTPDecoderLayer(nn.Module):
    def __init__(self, config, quant_config=None, prefix="", alt_stream=None):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            0,
            quant_config,
            f"{prefix}.self_attn",
            alt_stream,
            is_nextn=True,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DeepseekV2MoE(
            config,
            0,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            alt_stream=alt_stream,
            is_nextn=True,
            reduce_results=False,
        )
        if hasattr(self.mlp, "shared_experts"):
            self.mlp.shared_experts.swiglu_limit = None
        self.attn_boundary, self.ffn_boundary = append_stages(
            (
                declare_attn(),
                self.input_layernorm,
                {"qkv_latent_func": self.self_attn.prepare_qkv_latent},
            ),
            (declare_ffn(sparse=True), self.post_attention_layernorm),
        )

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
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
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)
        return hidden_states, topk_indices


class HYV4ModelNextN(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
            **get_embedding_tp_kwargs(),
        )
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.alt_stream = get_stream("alt") if is_cuda() else None
        with layer_stack():
            self.decoder = HYV4MTPDecoderLayer(
                config,
                quant_config,
                f"{prefix}.decoder",
                self.alt_stream,
            )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gate_cp_chain = build_gate_cp_chain([self.decoder.self_attn])
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_size = None

    @scoped_hyv4_dcp_write_locs
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        previous_hidden_states = forward_batch.spec_info.hidden_states
        cp_v2_active = is_cp_v2_active(forward_batch)
        use_cp_v1 = (
            dsa_use_prefill_cp(forward_batch, self.dsa_enable_prefill_cp)
            and not cp_v2_active
        )
        if use_cp_v1:
            (
                hidden_states,
                previous_hidden_states,
                positions,
            ) = _split_hyv4_nextn_cp_inputs(
                forward_batch, hidden_states, previous_hidden_states, positions
            )
        if hidden_states.shape[0] > 0:
            if previous_hidden_states.shape != hidden_states.shape:
                raise ValueError(
                    "HYV4 MTP requires token-aligned D-dimensional hidden states"
                )
            mtp_input = torch.cat(
                (
                    self.enorm(hidden_states),
                    self.hnorm(previous_hidden_states),
                ),
                dim=-1,
            )
            from sglang.srt.layers.hyv4_tuned_mtp import try_tuned_mtp_input

            hidden_states = try_tuned_mtp_input(self.eh_proj, mtp_input)
            if hidden_states is None:
                hidden_states = self.eh_proj(mtp_input)
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        residual_batch.start(forward_batch)
        index_topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
        if get_parallel().enable_dsa_cache_layer_split:
            # The draft layer has a single owner rank, and HiCache load-back
            # restores (and invalidates) the draft pool only on that rank. The
            # Index-K fallback broadcast is gated on rank-local cache state, so
            # the owner alone would issue an extra collective and deadlock the
            # CP group. Reset on every rank so all ranks rebroadcast together.
            reset_layer_broadcast_cache = getattr(
                get_token_to_kv_pool(), "reset_layer_broadcast_cache", None
            )
            if reset_layer_broadcast_cache is not None:
                reset_layer_broadcast_cache()
        # CP-split tokens differ across ranks, so the gate shard must gather
        # full gate weights before computing its local activation slice.
        if self.gate_cp_chain is not None:
            self.gate_cp_chain.begin(use_cp_v1, hidden_states.shape[0])
        try:
            hidden_states, topk_indices = self.decoder(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=index_topk_share.topk_indices,
            )
        finally:
            if self.gate_cp_chain is not None:
                self.gate_cp_chain.end()
        # Join alt_stream before returning: leftover work there can create a
        # cross-stream dependency that deadlocks the next target-model graph
        # replay. Same guard as DeepseekModelNextN.
        if self.alt_stream is not None:
            torch.cuda.current_stream().wait_stream(self.alt_stream)
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if forward_batch.forward_mode.is_idle():
            index_topk_share.update(topk_indices)
            index_topk_share.publish()
            return hidden_states
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self.shared_head.norm
        )
        if use_cp_v1:
            local_num_tokens = hidden_states.shape[0]
            hidden_states = _restore_hyv4_nextn_cp_output(
                hidden_states, self.cp_size, forward_batch
            )
            if index_topk_share.should_publish and topk_indices is not None:
                topk_indices = _gather_dsa_topk_indices_for_cp(
                    topk_indices,
                    local_num_tokens,
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
        index_topk_share.update(topk_indices)
        index_topk_share.publish()
        return hidden_states


class HYV4ForCausalLMNextN(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return HYV4ForCausalLM.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = _mtp_quant_config(quant_config)
        quant_config = self.quant_config
        if (
            quant_config is not None
            and quant_config.get_name() == "slimquant_w4a8_marlin"
            and getattr(quant_config, "checkpoint_format", None)
            not in ("hy4_w4a8_v1", "hy4_compressed_tensors_w4a8")
        ):
            raise ValueError("HYV4 MTP W4A8 requires a supported HYV4 checkpoint format")
        self.pp_group = get_parallel().pp_group
        validate_hyv4_launch(get_global_server_args(), get_parallel())
        # Both index-share modes use EagleDraftWorker's existing lifecycle:
        # false recomputes per step; true carries draft-extend top-k through
        # the draft chain (and recomputes if no seed is available).
        get_attn_tp_context().init_context(config.q_lora_rank, True)
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_rank = None
            self.cp_size = None
        self.model = HYV4ModelNextN(
            config, quant_config, prefix=f"{prefix}.model"
        )
        self.num_fused_shared_experts = self.model.decoder.mlp.num_fused_shared_experts
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch):
        # The draft runner creates a fresh ForwardBatch for EAGLE prefill.
        # Prepare the same CP metadata as the target before the model applies
        # the token split. CP-v2 owns sharding at the runner boundary and must
        # not be split a second time here.
        if (
            self.dsa_enable_prefill_cp
            and not is_cp_v2_active(forward_batch)
            and can_dsa_cp_split(
                len(input_ids), self.cp_size, True, forward_batch
            )
        ):
            forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                len(input_ids),
                self.cp_rank,
                self.cp_size,
                forward_batch.seq_lens_cpu.tolist(),
                extend_seqs_len=forward_batch.extend_seq_lens_cpu,
            )
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    prepare_context_parallel_metadata_for_dcp = (
        HYV4ForCausalLM.prepare_context_parallel_metadata_for_dcp
    )
    get_model_config_for_expert_location = classmethod(
        HYV4ForCausalLM.get_model_config_for_expert_location.__func__
    )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The HYV4 checkpoint names the draft layer "model.mtp_layers.0", while
        # do_load_weights maps a "model.layers.<nextn_layer_id>" prefix onto
        # "model.decoder". Ask the mixin for that prefix instead of recomputing
        # it.
        layer_prefix = self._initialize_nextn_conf(True).nextn_layer_prefix
        scale_suffix = hyv4_linear_scale_suffix(self)
        metadata = {}

        def mapped_weights():
            for name, loaded_weight in weights:
                name, loaded_weight = adapt_hyv4_packed_weight(
                    name, loaded_weight, self.quant_config
                )
                if name is None:
                    continue
                relative = normalize_hyv4_mtp_weight_name(name)
                if relative is None:
                    continue
                if relative in metadata:
                    raise ValueError(f"Duplicate HYV4 MTP weight: {relative}")
                metadata[relative] = {
                    "shape": list(loaded_weight.shape),
                    "dtype": str(loaded_weight.dtype),
                }
                name = layer_prefix + "." + relative
                loaded_weight = permute_hyv4_indexer_weight(
                    name, loaded_weight, self.config
                )
                if name.endswith(".weight_scale"):
                    name += scale_suffix
                yield name, loaded_weight
            validate_hyv4_mtp_metadata(
                vars(self.config),
                metadata,
                packed_experts=getattr(self.quant_config, "checkpoint_format", None)
                in ("hy4_w4a8_v1", "hy4_compressed_tensors_w4a8"),
                mixed_int8=getattr(self.quant_config, "checkpoint_format", None)
                == "hy4_compressed_tensors_w4a8",
            )

        self.do_load_weights(mapped_weights(), is_nextn=True)
        from sglang.srt.layers.hyv4_tuned_mtp import prepare_mtp_input

        prepare_mtp_input(self)

    def post_load_weights(self, is_nextn=True, weight_names=None):
        super().post_load_weights(is_nextn=True, weight_names=weight_names)


EntryClass = [HYV4ForCausalLMNextN]
