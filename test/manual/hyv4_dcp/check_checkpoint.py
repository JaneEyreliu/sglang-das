"""Read-only HY4 DCP2 checkpoint metadata check; no torch/GPU/model load."""

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

_helper_path = (
    Path(__file__).resolve().parents[3] / "python/sglang/srt/configs/hy_v4_mtp.py"
)
_spec = importlib.util.spec_from_file_location("hy_v4_mtp_metadata", _helper_path)
_mtp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mtp)


def read_header(path):
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 32 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large safetensors header: {path}")
        return json.loads(stream.read(size))


def check(model_path, *, mtp=False):
    config = json.loads((model_path / "config.json").read_text())
    if config.get("architectures") != ["HYV4ForCausalLM"]:
        raise ValueError("Expected architectures=[HYV4ForCausalLM]")
    if config.get("model_type") != "hy_v4":
        raise ValueError("Expected model_type=hy_v4")
    quant = config.get("quantization_config", {})
    checkpoint_format = quant.get("checkpoint_format")
    manifest = model_path / "hy4-assets.json"
    if checkpoint_format is None and manifest.is_file():
        checkpoint_format = json.loads(manifest.read_text()).get("format")
    if checkpoint_format != "hy4_w4a8_v1":
        raise ValueError(
            "Expected hy4_w4a8_v1 in quantization_config or hy4-assets.json"
        )
    heads, value_dim = config["num_attention_heads"], config["v_head_dim"]
    hidden, layers = config["hidden_size"], config["num_hidden_layers"]
    if heads % 2:
        raise ValueError("Attention heads must be divisible by attention TP2")
    if config["n_routed_experts"] % 32:
        raise ValueError("Routed expert count must be divisible by EP32")
    pattern = config.get("indexer_types")
    if pattern is not None and (
        len(pattern) != layers
        or pattern[0] != "full"
        or set(pattern) - {"full", "shared"}
    ):
        raise ValueError("Invalid HY4 indexer_types")
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise ValueError("No safetensors weight files found")
    found = set()
    mtp_weights = {}
    for file in files:
        for name, metadata in read_header(file).items():
            if mtp:
                relative = _mtp.normalize_hyv4_mtp_weight_name(name)
                if relative is not None:
                    if relative in mtp_weights:
                        raise ValueError(f"Duplicate HY4 MTP weight: {relative}")
                    mtp_weights[relative] = metadata
            parts = name.split(".")
            if len(parts) < 6 or parts[:2] != ["model", "layers"]:
                continue
            if parts[3:5] not in (
                ["self_attn", "linear_gate"],
                ["self_attn", "g_proj"],
            ):
                continue
            if parts[5:] not in (["weight"], ["weight", "weight"]):
                continue
            layer_id = int(parts[2])
            if layer_id >= layers:  # MTP is not part of the target stack.
                continue
            if metadata["shape"] != [heads * value_dim, hidden]:
                raise ValueError(f"Wrong gate shape for {name}: {metadata['shape']}")
            if metadata["dtype"] not in ("BF16", "F16", "F32"):
                raise ValueError(f"Expected unquantized gate weights: {name}")
            if layer_id in found:
                raise ValueError(f"Duplicate attention gate for layer {layer_id}")
            found.add(layer_id)
    if found != set(range(layers)):
        raise ValueError(
            f"Missing attention gates for layers: {sorted(set(range(layers)) - found)}"
        )
    result = {
        "layers": layers,
        "attention_tp": 2,
        "dcp": 2,
        "dp": 16,
        "ep": 32,
        "gate_global_shape": [heads * value_dim, hidden],
        "gate_per_rank_shape": [heads * value_dim // 2, hidden],
        "checkpoint_format": checkpoint_format,
    }
    if mtp:
        result.update(
            _mtp.validate_hyv4_mtp_metadata(config, mtp_weights, packed_experts=True)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument(
        "--mtp", action="store_true", help="Also validate the bundled W4A8 MTP layer"
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                check(args.model_path, mtp=args.mtp), ensure_ascii=False, indent=2
            )
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"HY4 checkpoint check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
