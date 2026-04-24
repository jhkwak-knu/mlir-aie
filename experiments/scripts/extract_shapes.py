"""Extract GEMM shapes from an MLP experiment config.

Given a config.json with `model.layer_sizes` and `dataset.batch_size`,
derive the per-layer GEMM shape (M=BS, K=prev, N=next) and emit a
shapes.json consumable by generate_configs.py.

Output schema (Notion step 21, Section "Step 1"):
    {
      "shapes": [
        {"layer": "fc1", "M": <bs>, "K": <prev>, "N": <next>, "calls_per_inference": 1},
        ...
      ],
      "unique_shapes": <int>,
      "total_calls_per_inference": <int>
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def derive_shapes(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-layer GEMM shapes from an MLP experiment config.

    Args:
        config: parsed JSON dict with `model.layer_sizes`, `model.activations`,
            `dataset.batch_size`.

    Returns:
        A dict matching the Notion Step 1 schema. Each GEMM is labeled fc1,
        fc2, ... in forward order, with calls_per_inference=1 because plain
        MLP inference calls each FC exactly once per sample.

    Raises:
        ValueError: if the config is structurally invalid (missing keys,
            too few layers, or activations length mismatch).
    """
    model = config.get("model")
    dataset = config.get("dataset")
    if not isinstance(model, dict) or not isinstance(dataset, dict):
        raise ValueError("config must contain 'model' and 'dataset' objects")

    if model.get("type") != "mlp":
        raise ValueError(
            f"unsupported model.type: {model.get('type')!r} (expected 'mlp')"
        )

    layer_sizes = model.get("layer_sizes")
    if not isinstance(layer_sizes, list) or len(layer_sizes) < 2:
        raise ValueError(
            "model.layer_sizes must be a list of at least 2 ints "
            "(input dim, [hidden ...,] output dim)"
        )
    if not all(isinstance(s, int) and s > 0 for s in layer_sizes):
        raise ValueError("all entries in model.layer_sizes must be positive ints")

    activations = model.get("activations", [])
    expected_activ = len(layer_sizes) - 2
    if not isinstance(activations, list) or len(activations) != expected_activ:
        raise ValueError(
            f"model.activations length must be {expected_activ} (got "
            f"{len(activations) if isinstance(activations, list) else type(activations).__name__}); "
            "one activation per hidden gap"
        )

    batch_size = dataset.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("dataset.batch_size must be a positive int")

    shapes: List[Dict[str, Any]] = []
    for i in range(len(layer_sizes) - 1):
        shapes.append(
            {
                "layer": f"fc{i + 1}",
                "M": batch_size,
                "K": layer_sizes[i],
                "N": layer_sizes[i + 1],
                "calls_per_inference": 1,
            }
        )

    return {
        "shapes": shapes,
        "unique_shapes": len(shapes),
        "total_calls_per_inference": sum(s["calls_per_inference"] for s in shapes),
    }


def _default_output_path(config_path: Path, config: Dict[str, Any]) -> Path:
    """Resolve the default shapes.json output location.

    Priority:
      1. config.output.results_dir (+ shapes.json)
      2. experiments/results/<config_stem>/shapes.json (fallback)
    """
    results_dir = config.get("output", {}).get("results_dir")
    if results_dir:
        return Path(results_dir).expanduser() / "shapes.json"
    return (
        Path(__file__).resolve().parents[1] / "results" / config_path.stem / "shapes.json"
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive per-layer GEMM shapes from an MLP experiment config.",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to experiment config JSON (e.g. "
             "experiments/configs/mlp_512_512_bs32.json).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output shapes.json path. Defaults to "
             "<config.output.results_dir>/shapes.json.",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print shapes JSON to stdout instead of writing a file.",
    )
    args = parser.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    shapes = derive_shapes(config)

    if args.stdout:
        json.dump(shapes, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    out_path = args.output or _default_output_path(args.config, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(shapes, f, indent=2)
        f.write("\n")

    print(f"wrote {out_path} ({shapes['unique_shapes']} shapes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
