"""HY4 DCP retraction: local target KV and replicated index/draft state."""

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.platforms import current_platform


class HYV4DCPAllocator(PagedTokenToKVPoolAllocator):
    def __init__(self, *args, dcp_rank, **kwargs):
        super().__init__(*args, **kwargs)
        self.dcp_rank = dcp_rank
        self.draft_pool = None

    def register_draft_pool(self, pool):
        if self.draft_pool is not None and self.draft_pool is not pool:
            raise ValueError("HY4 supports one replicated MTP cache pool")
        if pool.page_size != self.page_size or pool.size != self.size:
            raise ValueError("HY4 draft cache must span the allocator virtual space")
        self.draft_pool = pool

    def _local_target_indices(self, indices):
        physical_page = self._kvcache.page_size
        # Restoring into different page IDs preserves ownership only when the
        # request starts on a virtual-page boundary and each page is complete
        # except its final tail. Decode retraction allocates exactly this layout.
        offsets = torch.arange(indices.numel(), device=indices.device)
        if not torch.equal(indices % self.page_size, offsets % self.page_size):
            raise ValueError("HY4 retraction requires page-aligned request indices")
        owned = indices % self.page_size // physical_page == self.dcp_rank
        loc = indices[owned]
        return loc // self.page_size * physical_page + loc % physical_page

    @staticmethod
    def _index_buffers(pool):
        return (
            pool.index_key_cache.buffer
            if pool.index_key_cache is not None
            else pool.index_k_buffer
        )

    @classmethod
    def _copy_index(cls, pool, indices):
        page = pool.index_page_size
        pages = indices[::page] // page
        chunk = max(1, pool.cpu_offloading_chunk_size // page)
        # Blocking D2H owns the snapshot before request pages are released.
        return [
            [
                buf[pages[i : i + chunk]].to("cpu", copy=True)
                for i in range(0, len(pages), chunk)
            ]
            for buf in cls._index_buffers(pool)
        ]

    @classmethod
    def _restore_index(cls, pool, saved, indices):
        page = pool.index_page_size
        pages = indices[::page] // page
        chunk = max(1, pool.cpu_offloading_chunk_size // page)
        for buf, chunks in zip(cls._index_buffers(pool), saved, strict=True):
            for i, cpu in zip(range(0, len(pages), chunk), chunks, strict=True):
                buf[pages[i : i + chunk]] = cpu.to(buf.device)

    @classmethod
    def _copy_pool(cls, pool, kv_indices, index_indices):
        return {
            "kv": MLATokenToKVPool.get_cpu_copy(pool, kv_indices),
            "index": cls._copy_index(pool, index_indices),
        }

    @classmethod
    def _restore_pool(cls, pool, saved, kv_indices, index_indices):
        MLATokenToKVPool.load_cpu_copy(pool, saved["kv"], kv_indices)
        cls._restore_index(pool, saved["index"], index_indices)

    def get_cpu_copy(self, indices, mamba_indices=None):
        local = self._local_target_indices(indices)
        return {
            "tokens": indices.numel(),
            "target": self._copy_pool(self._kvcache, local, indices),
            "draft": (
                self._copy_pool(self.draft_pool, indices, indices)
                if self.draft_pool is not None
                else None
            ),
        }

    def load_cpu_copy(self, saved, indices, mamba_indices=None):
        if saved["tokens"] != indices.numel():
            raise ValueError("HY4 retraction restore length differs from backup")
        if (saved["draft"] is None) != (self.draft_pool is None):
            raise ValueError("HY4 draft pool changed during retraction")
        local = self._local_target_indices(indices)
        self._restore_pool(self._kvcache, saved["target"], local, indices)
        if self.draft_pool is not None:
            self._restore_pool(self.draft_pool, saved["draft"], indices, indices)
        # Match the pool offload contract: all index writes finish before a
        # resumed request can enter a forward on another stream.
        current_platform.synchronize()

    def snapshot_mtp_retraction_state(self, batch, future_map):
        """Save the latest draft seed before filtering/freeing request slots.

        The scheduler drains pending results and synchronizes before this call.
        Read relay buffers directly without consuming/invalidation: requests
        which remain running still need their next forward to consume them.
        """
        if self.draft_pool is None or batch.spec_info is None:
            return
        spec = batch.spec_info
        indices = spec.future_indices
        if indices is not None:
            topk_p = future_map.topk_p_buf[indices]
            topk_index = future_map.topk_index_buf[indices]
            hidden = future_map.hidden_states_buf[indices]
        else:
            topk_p, topk_index, hidden = (
                spec.topk_p,
                spec.topk_index,
                spec.hidden_states,
            )
        for i, req in enumerate(batch.reqs):
            req.output_topk_p = topk_p[i].to("cpu", copy=True)
            req.output_topk_index = topk_index[i].to("cpu", copy=True)
            req.hidden_states_tensor = hidden[i].to("cpu", copy=True)
            # Recompute DSA top-k after relocation; old slot IDs are invalid.
            req.output_dsa_topk_indices = None
