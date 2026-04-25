"""Extract GEMM shapes from an experiment config.

Supports two model types via `model.type`:

  * "mlp" — derives one fc<i> entry per (batch_size, layer_sizes[i],
    layer_sizes[i+1]) tuple in forward order.
  * "distilbert" — derives the 6 GEMM types of a DistilBERT encoder layer
    (attention_qkv, attention_output, attention_score, attention_context,
    ffn_expand, ffn_compress) from `num_layers / hidden_dim / ffn_dim /
    num_heads / sequence_length`. Uses static formulas — call counts and
    head-batched matmul shapes match Hugging Face `distilbert-base-uncased`
    forward semantics.

Output schema (Notion task #21 §1 + #22 §1-3):

    {
      "shapes": [
        {"layer": <id>, "M": ..., "K": ..., "N": ...,
         "calls_per_inference": <int>,
         # DistilBERT-only:
         "gemm_type": <id>, "layers": "0-N", "head_batched": bool,
         "sub_type": [...]},
        ...
      ],
      "unique_shapes": <int>,
      "total_calls_per_inference": <int>,
      # DistilBERT-only:
      "model": <pretrained_source>,
      "sequence_length": ..., "batch_size": ...,
      "multi_head_strategy": ...,
      "c5_violations": [...],
    }

DistilBERT C5 (XDNA2 mmul<4,8,8> 2x2 expansion: M%8 == 0, K%8 == 0,
N%16 == 0) is checked op-level. Violations land in `c5_violations`
without aborting the call so the user / Claude.ai can decide policy
(Notion #22 §Step 1-3 says: do NOT auto-fix, surface for review).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# XDNA2 op-level C5 thresholds (M%8 == 0, K%8 == 0, N%16 == 0). Mirrors
# xdna_search/hw_constants.py (MMUL_R=4, MMUL_S=8, MMUL_T=8) but is
# inlined here to keep extract_shapes.py runnable without sourcing
# setup_env.sh — only generate_configs.py / compile_kernels.py truly
# need xdna_search on sys.path.
_C5_M_MOD = 8
_C5_K_MOD = 8
_C5_N_MOD = 16


# ----- MLP -------------------------------------------------------------------


def _derive_shapes_mlp(config: Dict[str, Any]) -> Dict[str, Any]:
    """Per-layer GEMM shapes for a plain MLP forward pass."""
    model = config["model"]
    dataset = config["dataset"]

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


# ----- DistilBERT ------------------------------------------------------------


def _require_positive_int(d: Dict[str, Any], key: str, where: str) -> int:
    """Pull a positive-int field out of a config block or raise ValueError."""
    v = d.get(key)
    if not isinstance(v, int) or v <= 0:
        raise ValueError(
            f"{where}.{key} must be a positive int (got {v!r})"
        )
    return v


def _check_c5_op(M: int, K: int, N: int) -> bool:
    """Op-level XDNA2 C5: M%8 == 0, K%8 == 0, N%16 == 0."""
    return (M % _C5_M_MOD == 0) and (K % _C5_K_MOD == 0) and (N % _C5_N_MOD == 0)


def _derive_shapes_distilbert(config: Dict[str, Any]) -> Dict[str, Any]:
    """6 DistilBERT-encoder GEMM types derived statically from config.

    The 5 unique (M, K, N) tuples and their per-inference call counts
    follow the canonical Hugging Face `distilbert-base-uncased` forward
    pass: 4 nn.Linear ops per attention block (Q/K/V/out) + 2 ops per
    FFN block (expand/compress) + the Q·K^T and attn·V matmuls inside
    multi-head attention. The Q·K^T / attn·V pair is `head_batched`,
    i.e. one batched matmul per layer (heads stacked along the batch
    dim) — `(M, K, N)` here records the per-head shape, NOT the batched
    shape. The physical NPU dispatch can fold heads into M (M' = h*M)
    later in Step 4-3 without changing the unique-shape count.
    """
    model = config["model"]
    dataset = config["dataset"]

    num_layers = _require_positive_int(model, "num_layers", "model")
    hidden_dim = _require_positive_int(model, "hidden_dim", "model")
    ffn_dim = _require_positive_int(model, "ffn_dim", "model")
    num_heads = _require_positive_int(model, "num_heads", "model")
    seq_len = _require_positive_int(model, "sequence_length", "model")
    batch_size = _require_positive_int(dataset, "batch_size", "dataset")

    if hidden_dim % num_heads != 0:
        raise ValueError(
            f"model.hidden_dim ({hidden_dim}) must be divisible by "
            f"model.num_heads ({num_heads})"
        )
    head_dim = hidden_dim // num_heads

    multi_head_strategy = model.get("multi_head_strategy", "batched")
    pretrained_source = model.get("pretrained_source", "")

    layers_label = f"0-{num_layers - 1}"

    # 6 GEMM-type entries. attention_qkv keeps Q/K/V split as `sub_type`
    # so Step 6 per-type analysis can drill down further if needed; the
    # tuple (M, K, N) is identical for the three so the kernel binary
    # is shared.
    entries: List[Tuple[Dict[str, Any], int, int, int]] = [
        ({
            "layer": "attention_qkv",
            "gemm_type": "attention_qkv",
            "calls_per_inference": 3 * num_layers,
            "layers": layers_label,
            "sub_type": ["Q", "K", "V"],
        }, seq_len, hidden_dim, hidden_dim),
        ({
            "layer": "attention_output",
            "gemm_type": "attention_output",
            "calls_per_inference": num_layers,
            "layers": layers_label,
        }, seq_len, hidden_dim, hidden_dim),
        ({
            "layer": "attention_score",
            "gemm_type": "attention_score",
            "calls_per_inference": num_layers,
            "layers": layers_label,
            "head_batched": True,
        }, seq_len, head_dim, seq_len),
        ({
            "layer": "attention_context",
            "gemm_type": "attention_context",
            "calls_per_inference": num_layers,
            "layers": layers_label,
            "head_batched": True,
        }, seq_len, seq_len, head_dim),
        ({
            "layer": "ffn_expand",
            "gemm_type": "ffn_expand",
            "calls_per_inference": num_layers,
            "layers": layers_label,
        }, seq_len, hidden_dim, ffn_dim),
        ({
            "layer": "ffn_compress",
            "gemm_type": "ffn_compress",
            "calls_per_inference": num_layers,
            "layers": layers_label,
        }, seq_len, ffn_dim, hidden_dim),
    ]

    shapes: List[Dict[str, Any]] = []
    c5_violations: List[Dict[str, Any]] = []
    seen_mkn: set = set()
    for body, M, K, N in entries:
        body["M"] = M
        body["K"] = K
        body["N"] = N
        shapes.append(body)
        seen_mkn.add((M, K, N))
        if not _check_c5_op(M, K, N):
            c5_violations.append({
                "gemm_type": body["gemm_type"],
                "M": M, "K": K, "N": N,
                "rule": (f"M%{_C5_M_MOD}=={M % _C5_M_MOD}, "
                         f"K%{_C5_K_MOD}=={K % _C5_K_MOD}, "
                         f"N%{_C5_N_MOD}=={N % _C5_N_MOD}; "
                         f"need M%{_C5_M_MOD}==0, K%{_C5_K_MOD}==0, "
                         f"N%{_C5_N_MOD}==0"),
            })

    return {
        "model": pretrained_source,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "multi_head_strategy": multi_head_strategy,
        "shapes": shapes,
        "unique_shapes": len(seen_mkn),
        "total_calls_per_inference": sum(s["calls_per_inference"] for s in shapes),
        "c5_violations": c5_violations,
    }


# ----- Top-level dispatch ----------------------------------------------------


_DERIVERS = {
    "mlp": _derive_shapes_mlp,
    "distilbert": _derive_shapes_distilbert,
}


def derive_shapes(config: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch shape derivation by `model.type`.

    Raises:
        ValueError: if model/dataset are missing or model.type is not
            one of the supported derivers.
    """
    model = config.get("model")
    dataset = config.get("dataset")
    if not isinstance(model, dict) or not isinstance(dataset, dict):
        raise ValueError("config must contain 'model' and 'dataset' objects")

    mtype = model.get("type")
    deriver = _DERIVERS.get(mtype)
    if deriver is None:
        supported = ", ".join(sorted(_DERIVERS))
        raise ValueError(
            f"unsupported model.type: {mtype!r} (supported: {supported})"
        )
    return deriver(config)


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
        description="Derive per-GEMM shapes from an experiment config.",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to experiment config JSON.",
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

    # Surface C5 violations to stderr so a CLI run draws attention to
    # the issue without failing.
    for v in shapes.get("c5_violations", []) or []:
        print(
            f"warning: C5 violation: {v['gemm_type']} "
            f"({v['M']}x{v['K']}x{v['N']}) — {v['rule']}",
            file=sys.stderr,
        )

    if args.stdout:
        json.dump(shapes, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    out_path = args.output or _default_output_path(args.config, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(shapes, f, indent=2)
        f.write("\n")

    print(f"wrote {out_path} ({shapes['unique_shapes']} unique shapes, "
          f"{shapes['total_calls_per_inference']} calls/inference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
