from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Optional

import torch


def get_flashmla_op(name: str, *, is_hcu: bool) -> Callable[..., Any]:
    module_name = "flash_mla.flash_mla_interface" if is_hcu else "sgl_kernel.flash_mla"
    return getattr(import_module(module_name), name)


@dataclass(frozen=True)
class DSAFlashMLAMetadata:
    """Metadata needed only by FlashMLA.

    CUDA FlashMLA stores scheduler metadata in tensors. HCU FlashMLA stores it
    in a mutable backend object and initializes the object's tensor fields on
    the first kernel invocation.
    """

    flashmla_metadata: Any
    num_splits: Optional[torch.Tensor]

    def slice(self, sli) -> DSAFlashMLAMetadata:
        return DSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=(None if self.num_splits is None else self.num_splits[sli]),
        )

    def copy_(self, other: DSAFlashMLAMetadata) -> None:
        if isinstance(self.flashmla_metadata, torch.Tensor) and isinstance(
            other.flashmla_metadata, torch.Tensor
        ):
            self.flashmla_metadata.copy_(other.flashmla_metadata)
        else:
            object.__setattr__(self, "flashmla_metadata", other.flashmla_metadata)

        if isinstance(self.num_splits, torch.Tensor) and isinstance(
            other.num_splits, torch.Tensor
        ):
            self.num_splits.copy_(other.num_splits)
        else:
            object.__setattr__(self, "num_splits", other.num_splits)


@dataclass
class FlashMLACompactIndexCache:
    """One producer's local indices and scheduler, scoped to one forward.

    The backend must clear its entry at every forward-metadata initialization
    and replace it unconditionally at every index-producing layer. In
    particular, tensor or metadata identity is not a forward epoch: CUDA graph
    input buffers are updated in place. A producer reached during capture must
    start with a fresh scheduler so its update kernel is recorded in the graph.

    Only consecutive, explicitly sharing target layers may reuse an entry.
    Logical TopK tensors remain untouched; ``indices`` contains the separate
    rank-local compact representation. KV data and attention outputs are never
    cached here, because each layer has its own KV contents and queries.
    """

    original_topk: torch.Tensor
    forward_batch: Any
    metadata: Any
    last_layer_id: int
    num_rows: int
    dcp_size: int
    dcp_rank: int
    indices: torch.Tensor
    topk_length: torch.Tensor
    valid_chunks: Optional[torch.Tensor]
    layout_key: Optional[tuple] = None
    query_key: Optional[tuple] = None
    scheduler: Optional[DSAFlashMLAMetadata] = None
    scheduler_num_rows: Optional[int] = None
    scheduler_num_heads: Optional[int] = None
    scheduler_query_key: Optional[tuple] = None

    def can_reuse(
        self,
        *,
        layer: Any,
        original_topk: torch.Tensor,
        forward_batch: Any,
        metadata: Any,
        num_rows: int,
        dcp_size: int,
        dcp_rank: int,
        layout_key: Optional[tuple] = None,
        query_key: Optional[tuple] = None,
    ) -> bool:
        """Validate reuse using only host metadata, without reading GPU values.

        ``layout_key`` describes the conversion inputs and tensor layouts;
        ``query_key`` includes query trimming, padded heads and KV layout. The
        caller advances ``last_layer_id`` only after accepting a cache hit.
        Exact tensor identity intentionally excludes merely aliasing views.
        """
        return (
            bool(getattr(layer, "reuse_topk_indices", False))
            and not bool(getattr(layer, "is_nextn", False))
            and layer.layer_id == self.last_layer_id + 1
            and original_topk is self.original_topk
            and forward_batch is self.forward_batch
            and metadata is self.metadata
            and num_rows == self.num_rows
            and dcp_size == self.dcp_size
            and dcp_rank == self.dcp_rank
            and layout_key is not None
            and layout_key == self.layout_key
            and query_key is not None
            and query_key == self.query_key
        )

    def get_scheduler(
        self,
        *,
        num_rows: int,
        num_heads: int,
        query_key: Optional[tuple] = None,
    ) -> Optional[DSAFlashMLAMetadata]:
        """Return scheduling only for this entry's exact MLA input geometry."""
        if (
            query_key is not None
            and query_key == self.query_key
            and query_key == self.scheduler_query_key
            and num_rows == self.scheduler_num_rows
            and num_heads == self.scheduler_num_heads
        ):
            return self.scheduler
        return None

    def set_scheduler(
        self,
        scheduler: DSAFlashMLAMetadata,
        *,
        num_rows: int,
        num_heads: int,
        query_key: Optional[tuple] = None,
    ) -> None:
        if query_key is None or query_key != self.query_key:
            raise ValueError("FlashMLA scheduler geometry differs from compact indices")
        self.scheduler = scheduler
        self.scheduler_num_rows = num_rows
        self.scheduler_num_heads = num_heads
        self.scheduler_query_key = query_key


def can_fuse_flashmla_metadata(
    *metadatas: Optional[DSAFlashMLAMetadata],
) -> bool:
    return all(
        metadata is not None
        and isinstance(metadata.flashmla_metadata, torch.Tensor)
        and isinstance(metadata.num_splits, torch.Tensor)
        for metadata in metadatas
    )


def refresh_flashmla_metadata(
    destination: DSAFlashMLAMetadata,
    source: DSAFlashMLAMetadata,
    sli,
    *,
    is_hcu: bool,
) -> DSAFlashMLAMetadata:
    """Return the metadata wrapper the next forward must retain.

    Tensor metadata is refreshed in-place so captured data pointers stay
    stable. HCU scheduler objects carry shape-bound mutable state, so callers
    must retain the fresh source object instead of mutating a sliced temporary
    wrapper that the forward never reads.
    """

    if is_hcu:
        return source
    destination_view = destination.slice(sli)
    destination_view.copy_(source)
    return destination_view


def wrap_flashmla_metadata_result(
    result: tuple[Any, Optional[torch.Tensor]], *, is_hcu: bool
) -> DSAFlashMLAMetadata:
    flashmla_metadata, num_splits = result
    if is_hcu:
        num_splits = getattr(flashmla_metadata, "num_splits", num_splits)
    return DSAFlashMLAMetadata(
        flashmla_metadata=flashmla_metadata,
        num_splits=num_splits,
    )
