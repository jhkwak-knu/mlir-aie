"""Generate per-(shape, setter) tiling configurations for an MLP experiment.

Pipeline:
  1. Load the experiment config and its derived shapes.json.
  2. Run Step 2-0 preflight: instantiate each requested setter with a dummy
     shape (32, 784, 512) and record ok / empty / failure status.
  3. For each (shape, setter) pair, call xdna_search.search.factory and
     convert the best CostResult into a tc.json-compatible dict via
     scripts/generate/cost_model.py::cost_result_to_tc.
  4. Emit configurations.json at Notion Step 2 schema, with preflight on top.

Error policy (Notion Section 7): setter exceptions or empty results are
captured per-entry and do NOT abort the remaining combinations.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "test" / "onnx-mlir" / "scripts"
GENERATE_DIR = SCRIPTS_DIR / "generate"

# xdna_search and cost_model live under test/onnx-mlir/scripts/*; add both.
for _p in (SCRIPTS_DIR, GENERATE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cost_model import cost_result_to_tc  # noqa: E402
from xdna_search.hw_constants import (  # noqa: E402
    DEFAULT_CALIB_PATH,
    DEFAULT_SYS_PATH,
)
from xdna_search.io import load_calibration, load_system_info  # noqa: E402
from xdna_search.search.factory import get_searcher_factory  # noqa: E402
from xdna_search.types import (  # noqa: E402
    CalibCoeffs,
    DEFAULT_COEFFS,
    OpCase,
    SystemInfo,
)


# Notion uses snake_case setter identifiers; xdna_search registers hyphenated
# factory keys. Keep the mapping in one place so adding a setter later only
# touches this dict.
SETTER_FACTORY_KEY: Dict[str, str] = {
    "star_map": "star-map",
    "max_p": "max-p",
    "charm_cdse": "charm",
    "timeloop": "timeloop",
}

# Dummy shape for Step 2-0 preflight. Matches MLP fc1 so issues hit the real
# workload shape first.
PREFLIGHT_SHAPE: Tuple[int, int, int] = (32, 784, 512)

AXIS_LABEL = ("M", "N", "K")


# ----- Preflight -----------------------------------------------------------


def run_preflight(
    sys_info: SystemInfo,
    coeffs: CalibCoeffs,
    setters: List[str],
    shape: Tuple[int, int, int] = PREFLIGHT_SHAPE,
) -> Dict[str, Any]:
    """Step 2-0: verify each setter can produce a config for a dummy shape."""
    report: Dict[str, Any] = {
        "dummy_shape": {"M": shape[0], "K": shape[1], "N": shape[2]},
        "setters": {},
    }
    for setter in setters:
        entry: Dict[str, Any] = {"factory_key": SETTER_FACTORY_KEY.get(setter)}
        if entry["factory_key"] is None:
            entry["status"] = "unknown_setter"
            entry["error"] = f"{setter!r} is not in SETTER_FACTORY_KEY"
            report["setters"][setter] = entry
            continue

        try:
            factory = get_searcher_factory(entry["factory_key"])
            searcher = factory(sys_info, coeffs)
            op = OpCase(M=shape[0], K=shape[1], N=shape[2], elem_type="bf16")
            out = searcher.search(op)
            if not out.ranked:
                entry["status"] = "empty"
                entry["error"] = (
                    "searcher returned empty ranked list "
                    f"(valid={len(out.valid)}, enumerated={out.total_enumerated})"
                )
            else:
                entry["status"] = "ok"
                entry["valid_candidates"] = len(out.valid)
                entry["ranked_count"] = len(out.ranked)
        except Exception as exc:  # noqa: BLE001 — we want to capture anything
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()
        report["setters"][setter] = entry
    return report


# ----- Configuration generation -------------------------------------------


def _config_view(cr_candidate, cr_tp_order: int, tc_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return the Notion-schema `config` block for a CostResult."""
    return {
        "P": cr_candidate.num_cores,
        "SP_m": cr_candidate.SPm,
        "SP_n": cr_candidate.SPn,
        "TP_m": cr_candidate.TPm,
        "TP_n": cr_candidate.TPn,
        "TP_k": cr_candidate.TPk,
        "d_inner": AXIS_LABEL[cr_tp_order],
        "tc_entry": tc_entry,
    }


def generate_configurations(
    shapes: List[Dict[str, Any]],
    setters: List[str],
    sys_info: SystemInfo,
    coeffs: CalibCoeffs,
    elem_type: str = "bf16",
) -> List[Dict[str, Any]]:
    """Produce one entry per (shape, setter) with best-rank tc.json dict.

    Per Notion Section 7, exceptions and empty rankings are captured in an
    `error` field; the loop never aborts.
    """
    configurations: List[Dict[str, Any]] = []
    for shape in shapes:
        shape_view = {
            "layer": shape["layer"],
            "M": shape["M"], "K": shape["K"], "N": shape["N"],
        }
        for setter in setters:
            entry: Dict[str, Any] = {"shape": shape_view, "setter": setter}
            factory_key = SETTER_FACTORY_KEY.get(setter)
            if factory_key is None:
                entry["error"] = f"unknown setter {setter!r}"
                configurations.append(entry)
                continue

            try:
                searcher = get_searcher_factory(factory_key)(sys_info, coeffs)
                op = OpCase(
                    M=shape["M"], K=shape["K"], N=shape["N"],
                    elem_type=elem_type,
                )
                out = searcher.search(op)
                if not out.ranked:
                    entry["error"] = (
                        "no valid configuration (empty ranked list); "
                        f"valid={len(out.valid)}, enumerated={out.total_enumerated}"
                    )
                else:
                    best = out.ranked[0]
                    tc_entry = cost_result_to_tc(op, best)
                    entry["config"] = _config_view(best.candidate, best.tp_order, tc_entry)
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()
            configurations.append(entry)
    return configurations


# ----- Loading helpers -----------------------------------------------------


def load_shapes(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    shapes = doc.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError(f"{path}: 'shapes' array is missing or empty")
    return shapes


def _resolve_paths(
    config_path: Path, config: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Return (shapes_path, default_output_path) derived from a config.

    Uses config.output.results_dir if set; otherwise falls back to
    experiments/results/<config_stem>/.
    """
    results_dir_raw = config.get("output", {}).get("results_dir")
    if results_dir_raw:
        results_dir = Path(results_dir_raw).expanduser()
    else:
        results_dir = (
            Path(__file__).resolve().parents[1] / "results" / config_path.stem
        )
    return results_dir / "shapes.json", results_dir / "configurations.json"


# ----- CLI -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-(shape, setter) tiling configurations.",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Experiment config JSON (consumes model/, setters/, output/).",
    )
    parser.add_argument(
        "--shapes", type=Path, default=None,
        help="shapes.json path. Defaults to "
             "<config.output.results_dir>/shapes.json.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output configurations.json path. Defaults to "
             "<config.output.results_dir>/configurations.json.",
    )
    parser.add_argument(
        "--system-info", type=Path, default=Path(DEFAULT_SYS_PATH),
        help="xdna2_info.json path.",
    )
    parser.add_argument(
        "--calibration", type=Path, default=Path(DEFAULT_CALIB_PATH),
        help="Calibration JSON path. Falls back to DEFAULT_COEFFS if absent.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run preflight only; do not emit configurations.json.",
    )
    parser.add_argument(
        "--elem-type", default="bf16",
        help="Element type passed into OpCase (default: bf16).",
    )
    args = parser.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    setters = config.get("setters") or []
    if not setters:
        parser.error(f"{args.config}: 'setters' must be a non-empty list")

    shapes_path_default, output_path_default = _resolve_paths(args.config, config)
    shapes_path = args.shapes or shapes_path_default
    output_path = args.output or output_path_default

    shapes = load_shapes(shapes_path)
    sys_info = load_system_info(args.system_info)
    try:
        coeffs = load_calibration(args.calibration)
    except FileNotFoundError:
        print(
            f"warning: calibration file {args.calibration} missing; "
            "using DEFAULT_COEFFS",
            file=sys.stderr,
        )
        coeffs = DEFAULT_COEFFS

    preflight = run_preflight(sys_info, coeffs, setters)

    print("preflight:")
    for setter, info in preflight["setters"].items():
        status = info["status"]
        extra = info.get("error", "") if status != "ok" else f"valid={info.get('valid_candidates')}"
        print(f"  {setter:12s} [{status:8s}] {extra}")

    if args.dry_run:
        return 0

    configurations = generate_configurations(shapes, setters, sys_info, coeffs, args.elem_type)

    doc = {
        "source_config": str(args.config.resolve()),
        "source_shapes": str(shapes_path.resolve()),
        "preflight": preflight,
        "configurations": configurations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    n_err = sum(1 for e in configurations if "error" in e)
    print(
        f"wrote {output_path} "
        f"({len(configurations)} entries, {n_err} errors)"
    )
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
