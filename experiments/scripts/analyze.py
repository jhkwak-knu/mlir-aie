"""Aggregate measurements.csv / .json and produce figures + report.

Notion step 21 Step 6-4 / 6-6 outputs:
  * report/mlp_experiment_report.md
  * figures/fig1_kernel_edp.png
  * figures/fig2_model_edp.png
  * figures/fig3_level1_vs_level2.png
  * tables/table1_edp_decomp.csv

Inputs:
  * measurements.csv (runner append-output, one row per (batch, setter, layer))
  * measurements.json (runner append-output, runs[] array; used for metadata)

Metrics:
  * Per setter: min model time, min model energy -> min model EDP.
  * Per (setter, layer): min layer time, min layer energy.
  * Dilution factor (vs STAR-Map baseline by default):
        level1_gain = EDP_star / EDP_setter  (kernel-level, summed over layers)
        level2_gain = EDP_star / EDP_setter  (model-level)
        dilution   = level2_gain / level1_gain
    Values near 1.0 mean kernel-level wins transfer cleanly; smaller
    values mean host overhead / non-GEMM layers dilute the advantage.
  * GEMM time fraction = sum(layer_time_us_min) / model_time_us_min.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal Markdown table renderer (avoids the tabulate dependency)."""
    cols = list(df.columns)
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:{floatfmt[1:]}}"
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(fmt(row[c]) for c in cols) + " |"
            for _, row in df.iterrows()]
    return "\n".join([header, sep, *rows])


# ----- Core aggregation helpers -------------------------------------------


def aggregate_per_setter(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate measurements.csv into per-(setter, measurement_type, layer) mins.

    The runner's `energy_uj_min_package` column is a *per-batch* total over
    `n_inner` inference replays, whereas `time_us_min` is already the per-inference
    minimum observed within that batch. Normalize energy to per-inference before
    computing EDP so the two axes share the same unit.
    """
    df = df.copy()
    # Guard against rows with n_inner == 0 (shouldn't happen, but be explicit).
    df["energy_per_inference_uj"] = df.apply(
        lambda r: (r["energy_uj_min_package"] / r["n_inner"]) if r["n_inner"] else 0.0,
        axis=1,
    )
    g = df.groupby(["setter", "measurement_type", "layer"], as_index=False).agg(
        time_us_min=("time_us_min", "min"),
        time_us_mean=("time_us_mean", "mean"),
        energy_per_inference_uj_min=("energy_per_inference_uj", "min"),
        n_batches=("batch_idx", "nunique"),
    )
    g["edp_min"] = g["time_us_min"] * g["energy_per_inference_uj_min"]
    return g


def dilution_factors(per_setter: pd.DataFrame,
                     baseline_setter: str) -> pd.DataFrame:
    """Compute dilution factor per setter against `baseline_setter`.

    Returns a frame with columns: setter, level1_gain, level2_gain, dilution.
    Gain ratios are baseline / setter (larger is better for the baseline).
    """
    kernels = per_setter[per_setter["measurement_type"] == "kernel"]
    model = per_setter[per_setter["measurement_type"] == "model"]

    level1_edp = kernels.groupby("setter")["edp_min"].sum().rename("level1_edp")
    level2_edp = model.groupby("setter")["edp_min"].sum().rename("level2_edp")
    out = pd.concat([level1_edp, level2_edp], axis=1).reset_index()

    base = out[out["setter"] == baseline_setter].iloc[0]
    out["level1_gain"] = base["level1_edp"] / out["level1_edp"]
    out["level2_gain"] = base["level2_edp"] / out["level2_edp"]
    out["dilution"] = out["level2_gain"] / out["level1_gain"]
    return out.sort_values("setter").reset_index(drop=True)


def gemm_time_fraction(per_setter: pd.DataFrame) -> pd.DataFrame:
    """For each setter: sum of layer times / model time."""
    kernels = per_setter[per_setter["measurement_type"] == "kernel"]
    model = per_setter[per_setter["measurement_type"] == "model"]
    layers_time = kernels.groupby("setter")["time_us_min"].sum().rename("layers_time_us_sum")
    model_time = model.groupby("setter")["time_us_min"].min().rename("model_time_us_min")
    out = pd.concat([layers_time, model_time], axis=1).reset_index()
    out["gemm_time_fraction"] = out["layers_time_us_sum"] / out["model_time_us_min"]
    return out


# ----- Figures -------------------------------------------------------------


def fig1_kernel_edp(per_setter: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # local import so tests can stub

    kernels = per_setter[per_setter["measurement_type"] == "kernel"].copy()
    if kernels.empty:
        return
    pivot = kernels.pivot(index="layer", columns="setter", values="edp_min")
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_title("Kernel-level EDP per layer (min across batches)")
    ax.set_ylabel("EDP (us * uJ)")
    ax.set_xlabel("layer")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig2_model_edp(per_setter: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    model = per_setter[per_setter["measurement_type"] == "model"].copy()
    if model.empty:
        return
    ax = model.plot(
        x="setter", y="edp_min", kind="bar", legend=False, figsize=(6, 4),
    )
    ax.set_title("Model-level EDP per setter (min across batches)")
    ax.set_ylabel("EDP (us * uJ)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig3_level1_vs_level2(dilution: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if dilution.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dilution["level1_gain"], dilution["level2_gain"])
    for _, row in dilution.iterrows():
        ax.annotate(row["setter"],
                    (row["level1_gain"], row["level2_gain"]),
                    textcoords="offset points", xytext=(5, 5))
    lim = max(float(dilution[["level1_gain", "level2_gain"]].max().max()), 1.0) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="perfect transfer")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Level-1 EDP gain (baseline / setter)")
    ax.set_ylabel("Level-2 EDP gain (baseline / setter)")
    ax.set_title("Kernel-level vs model-level EDP gain")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----- Report writer -------------------------------------------------------


def write_report(report_path: Path,
                 per_setter: pd.DataFrame,
                 dilution: pd.DataFrame,
                 gemm_frac: pd.DataFrame,
                 metadata: Dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model = per_setter[per_setter["measurement_type"] == "model"].sort_values("setter")
    kernels = per_setter[per_setter["measurement_type"] == "kernel"].copy()
    kernels = kernels.sort_values(["setter", "layer"])

    lines: List[str] = []
    lines.append("# MLP E2E EDP Experiment Report\n")
    lines.append(f"Generated by analyze.py from measurements.csv.\n")
    lines.append(f"Measurement config: `{metadata.get('config_path', 'unknown')}`\n")
    lines.append(f"Backend: **{metadata.get('backend', 'unknown')}**, "
                 f"Runs: **{metadata.get('num_runs', '?')}**, "
                 f"Baseline setter: **{metadata.get('baseline_setter')}**.\n")

    lines.append("\n## Model-level aggregates (per inference, min across batches)\n")
    lines.append(_df_to_markdown(
        model[["setter", "time_us_min", "energy_per_inference_uj_min", "edp_min"]],
        floatfmt=".2f"))

    lines.append("\n\n## Per-layer kernel aggregates (per inference, min across batches)\n")
    lines.append(_df_to_markdown(
        kernels[["setter", "layer", "time_us_min",
                 "energy_per_inference_uj_min", "edp_min"]],
        floatfmt=".2f"))

    lines.append("\n\n## Dilution factor (vs baseline)\n")
    lines.append(_df_to_markdown(dilution, floatfmt=".3f"))

    lines.append("\n\n## GEMM time fraction per setter\n")
    lines.append(_df_to_markdown(gemm_frac, floatfmt=".3f"))

    lines.append("\n\n## Interpretation\n")
    lines.append("- `level1_gain` > 1 means the setter's summed per-layer EDP "
                 "is lower than the baseline (higher is better for that setter).\n")
    lines.append("- `level2_gain` > 1 means the setter's whole-inference EDP "
                 "is lower than the baseline.\n")
    lines.append("- `dilution` = level2_gain / level1_gain. Values near 1.0 mean "
                 "kernel-level gains survive into model-level; below 1.0 means "
                 "host overhead / non-GEMM stages (ReLU, softmax, dispatch "
                 "latency) are eroding the advantage.\n")

    report_path.write_text("\n".join(lines) + "\n")


# ----- CLI -----------------------------------------------------------------


def _load_metadata(measurements_json: Path, baseline_setter: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"baseline_setter": baseline_setter}
    if measurements_json.is_file():
        doc = json.loads(measurements_json.read_text())
        runs = doc.get("runs", [])
        meta["num_runs"] = len(runs)
        if runs:
            meta["config_path"] = runs[0].get("config_path")
            meta["backend"] = runs[0].get("backend")
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate measurements + figures.")
    parser.add_argument("--measurements-csv", type=Path, required=True)
    parser.add_argument("--measurements-json", type=Path, default=None,
                        help="for metadata; defaults to CSV sibling measurements.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="will write report/, figures/, tables/ under this dir")
    parser.add_argument("--baseline-setter", default="star_map",
                        help="setter to use as the 1.0x reference for gains")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.measurements_csv)
    if df.empty:
        print("analyze: measurements.csv is empty", file=sys.stderr)
        return 2

    measurements_json = args.measurements_json or args.measurements_csv.with_name(
        "measurements.json"
    )
    meta = _load_metadata(measurements_json, args.baseline_setter)

    per_setter = aggregate_per_setter(df)
    dilution = dilution_factors(per_setter, args.baseline_setter)
    gemm_frac = gemm_time_fraction(per_setter)

    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report").mkdir(parents=True, exist_ok=True)

    fig1_kernel_edp(per_setter, args.output_dir / "figures" / "fig1_kernel_edp.png")
    fig2_model_edp(per_setter, args.output_dir / "figures" / "fig2_model_edp.png")
    fig3_level1_vs_level2(dilution,
                          args.output_dir / "figures" / "fig3_level1_vs_level2.png")

    per_setter.to_csv(args.output_dir / "tables" / "per_setter.csv", index=False)
    dilution.to_csv(args.output_dir / "tables" / "table1_edp_decomp.csv", index=False)
    gemm_frac.to_csv(args.output_dir / "tables" / "gemm_time_fraction.csv", index=False)

    report_path = args.output_dir / "report" / "mlp_experiment_report.md"
    write_report(report_path, per_setter, dilution, gemm_frac, meta)

    print(f"analyze: wrote {report_path}")
    print(f"analyze: figures under {args.output_dir / 'figures'}")
    print(f"analyze: tables  under {args.output_dir / 'tables'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
