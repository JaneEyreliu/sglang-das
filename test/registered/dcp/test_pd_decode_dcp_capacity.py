from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.decode import DecodePreallocQueue


def test_pd_decode_admits_global_prompt_against_dcp_virtual_capacity():
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(enable_hisparse=False)
    queue.max_total_num_tokens = 312_320
    queue.token_to_kv_pool_allocator = SimpleNamespace(size=312_320 * 8)
    queue._uses_swa_tail_prealloc = lambda: False
    req = SimpleNamespace(
        rid="1m-prompt",
        origin_input_ids=range(1_048_000),
        output_ids=[],
        pd_rebootstrap_in_progress=False,
    )

    with patch(
        "sglang.srt.disaggregation.decode.get_parallel",
        return_value=SimpleNamespace(dcp_enabled=True),
    ):
        assert not queue._check_if_req_exceed_kv_capacity(req)
