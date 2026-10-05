import ast
import unittest
from collections import Counter
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MOONCAKE_CONN = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/disaggregation/mooncake/conn.py"
)


class TestMooncakeTransferDispatchSource(unittest.TestCase):
    def test_worker_has_one_kv_transfer_dispatch_chain(self):
        tree = ast.parse(MOONCAKE_CONN.read_text())
        worker = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "transfer_worker"
        )
        calls = Counter(
            node.func.attr
            for node in ast.walk(worker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {
                "send_kvcache",
                "send_kvcache_dcp",
                "send_kvcache_slice",
                "_do_staging_transfer",
            }
        )

        self.assertEqual(
            calls,
            Counter(
                {
                    "send_kvcache": 1,
                    "send_kvcache_dcp": 1,
                    "send_kvcache_slice": 1,
                    "_do_staging_transfer": 1,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
