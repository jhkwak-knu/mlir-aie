"""Publication-quality figures and tables for the E2E EDP experiment.

Generates the artifacts used in the dissertation chapter for both the MLP
and DistilBERT runs. Designed to be model-agnostic: pass in
`--measurements-csv`, `--configurations-json`, and `--output-dir`, and the
script writes vector + raster figures plus CSV tables.

Outputs (under `<output-dir>`):

    figures/fig1_setter_breakdown.{pdf,png}
    figures/fig2_te_decomposition.{pdf,png}
    tables/table1_main.csv
    tables/table2_configs.csv

Style conventions (kept consistent across MLP and DistilBERT):

    * Setter palette (4 colorblind-safe hues)
    * Serif font, 10pt body / 9pt ticks
    * 300 DPI for PNG, native vector for PDF
    * Baseline reference (1.0) drawn explicitly on gain plots
    * Bar value labels for direct readability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Reuse aggregation primitives so plots stay consistent with analyze.py.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import analyze  # noqa: E402

# ---------- Style constants -------------------------------------------------

SETTER_ORDER: List[str] = ["star_map", "max_p", "charm_cdse", "timeloop"]
SETTER_DISPLAY: Dict[str, str] = {
    "star_map":   "STAR-Map",
    "max_p":      "Max-P",
    "charm_cdse": "CHARM-CDSE",
    "timeloop":   "Timeloop",
}
SETTER_PALETTE: Dict[str, str] = {
    "star_map":   "#3E5972",  # dark slate blue (proposed method, emphasized)
    "max_p":      "#E89B7B",  # coral (matches reference figure)
    "charm_cdse": "#B5A07A",  # warm khaki
    "timeloop":   "#6A8E7F",  # sage green
}
LAYER_PALETTE: Dict[str, str] = {
    "fc1": "#3E5972",  # dark slate blue (matches reference)
    "fc2": "#E89B7B",  # coral (matches reference)
    "fc3": "#B0B0B0",  # neutral grey
}
PNG_DPI = 300
RC_PARAMS: Dict[str, Any] = {
    # Font is left at matplotlib default (DejaVu Sans) for consistency with
    # earlier figures already pulled into the dissertation chapter.
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.alpha":        0.5,
    "axes.axisbelow":    True,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,  # embed Type-42 fonts so reviewers can copy text
}

# ---------- Figure helpers --------------------------------------------------


def _save(fig, base_path: Path) -> None:
    """Save both PDF (vector) and PNG (raster, 300dpi) next to each other."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".pdf"))
    fig.savefig(base_path.with_suffix(".png"), dpi=PNG_DPI)


def _setters_present(per_setter: pd.DataFrame) -> List[str]:
    seen = set(per_setter["setter"].unique())
    return [s for s in SETTER_ORDER if s in seen]


def figure_setter_breakdown(per_setter: pd.DataFrame, base_path: Path,
                             baseline_setter: str) -> None:
    """X-axis = setter; per setter, three layer EDP-gain bars plus a model
    EDP-gain marker.

    Bars and the marker are normalized to the baseline setter at matching
    granularity (per-layer for the bars, per-inference for the marker), so
    a value above 1.0 means the setter beats the baseline. The vertical
    distance between the bars and the diamond visualizes the per-setter
    kernel->model dilution. The diamond carries the model gain as a text
    label so the figure is readable without the table.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC_PARAMS)

    kernels = per_setter[per_setter["measurement_type"] == "kernel"]
    models = per_setter[per_setter["measurement_type"] == "model"]
    if kernels.empty or models.empty:
        return

    setters = [s for s in SETTER_ORDER if s in per_setter["setter"].unique()]
    layers = ["fc1", "fc2", "fc3"]
    layers = [l for l in layers if l in kernels["layer"].unique()]
    if not setters or not layers:
        return

    base_kernel: Dict[str, float] = {}
    for l in layers:
        r = kernels[(kernels["setter"] == baseline_setter)
                    & (kernels["layer"] == l)]
        base_kernel[l] = float(r.iloc[0]["edp_min"]) if not r.empty else 1.0
    rb = models[models["setter"] == baseline_setter]
    base_model = float(rb.iloc[0]["edp_min"]) if not rb.empty else 1.0

    width = 0.22
    x = np.arange(len(setters))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    layer_ratios: Dict[str, List[float]] = {}
    for j, l in enumerate(layers):
        vals: List[float] = []
        for s in setters:
            r = kernels[(kernels["setter"] == s) & (kernels["layer"] == l)]
            v = float(r.iloc[0]["edp_min"]) if not r.empty else 0.0
            vals.append((base_kernel[l] / v) if v > 0 else 0.0)
        layer_ratios[l] = vals
        offset = (j - (len(layers) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=l,
               color=LAYER_PALETTE.get(l, "gray"),
               edgecolor="black", linewidth=0.4)

    model_ratios: List[float] = []
    for s in setters:
        r = models[models["setter"] == s]
        v = float(r.iloc[0]["edp_min"]) if not r.empty else 0.0
        model_ratios.append((base_model / v) if v > 0 else 0.0)
    ax.scatter(x, model_ratios, marker="D", s=70, color="black",
               zorder=10, label="Model")
    for i in range(len(setters)):
        ax.text(x[i], model_ratios[i],
                f"  {model_ratios[i]:.2f}",
                ha="left", va="center", fontsize=8)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SETTER_DISPLAY.get(s, s) for s in setters])
    ax.set_ylabel(f"EDP gain (baseline = "
                  f"{SETTER_DISPLAY.get(baseline_setter, baseline_setter)})")
    all_vals = [v for vs in layer_ratios.values() for v in vs] + model_ratios
    ymax = max(max(all_vals), 1.0) * 1.15
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=4, frameon=True)
    fig.tight_layout()
    _save(fig, base_path)
    plt.close(fig)


def figure_te_decomposition(per_setter: pd.DataFrame, base_path: Path) -> None:
    """Two-panel decomposition: model time and model energy split into

        host overhead | kernel (sum of per-layer dispatches)

    Time decomposition uses the directly measured per-layer dispatch times
    (their sum equals the kernel time, the residual is the host time).
    Energy is decomposed by time-proportional attribution
    (E_host = E_model x T_host / T_model) because per-layer RAPL deltas are
    too coarse-grained relative to fc3's ~680us dispatch to attribute
    energy reliably layer by layer.

    Putting host overhead at the bottom and the colored kernel segment on
    top makes the host band's roughly uniform thickness across setters
    visible at a glance, which is what the dilution-factor argument
    relies on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(RC_PARAMS)
    setters = [s for s in SETTER_ORDER
               if s in per_setter["setter"].unique()]
    if not setters:
        return

    # Pull model & summed-kernel time/energy per setter.
    rows = []
    for s in setters:
        model_row = per_setter[(per_setter["setter"] == s)
                               & (per_setter["measurement_type"] == "model")]
        kernel_rows = per_setter[(per_setter["setter"] == s)
                                 & (per_setter["measurement_type"] == "kernel")]
        if model_row.empty or kernel_rows.empty:
            continue
        m = model_row.iloc[0]
        t_model = float(m["time_us_min"])
        e_model = float(m["energy_per_inference_uj_min"])
        t_kernel = float(kernel_rows["time_us_min"].sum())
        t_host = max(0.0, t_model - t_kernel)
        e_host = e_model * (t_host / t_model) if t_model > 0 else 0.0
        e_kernel = max(0.0, e_model - e_host)
        rows.append({
            "setter": s,
            "T_model_ms":  t_model / 1e3,
            "T_kernel_ms": t_kernel / 1e3,
            "T_host_ms":   t_host / 1e3,
            "E_model_mJ":  e_model / 1e3,
            "E_kernel_mJ": e_kernel / 1e3,
            "E_host_mJ":   e_host / 1e3,
        })
    sd = pd.DataFrame(rows)
    if sd.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    OVERHEAD_COLOR = "#BDBDBD"
    x = np.arange(len(setters))
    width = 0.6

    # ----- (a) Time -----
    ax1.bar(x, sd["T_host_ms"], width,
            color=OVERHEAD_COLOR, edgecolor="black", linewidth=0.5, hatch="...")
    for i, s in enumerate(setters):
        ax1.bar(x[i], sd["T_kernel_ms"].iloc[i], width,
                bottom=sd["T_host_ms"].iloc[i],
                color=SETTER_PALETTE.get(s, "gray"),
                edgecolor="black", linewidth=0.5)
    for i in range(len(setters)):
        ax1.text(x[i], sd["T_model_ms"].iloc[i],
                 f"{sd['T_model_ms'].iloc[i]:.1f}",
                 ha="center", va="bottom", fontsize=8)
        ax1.text(x[i], sd["T_host_ms"].iloc[i] + sd["T_kernel_ms"].iloc[i] / 2,
                 f"{sd['T_kernel_ms'].iloc[i]:.1f}",
                 ha="center", va="center", fontsize=8, color="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([SETTER_DISPLAY.get(s, s) for s in setters])
    ax1.set_ylabel("Time per inference (ms)")
    ax1.set_title("(a) Time")

    # ----- (b) Energy -----
    ax2.bar(x, sd["E_host_mJ"], width,
            color=OVERHEAD_COLOR, edgecolor="black", linewidth=0.5, hatch="...")
    for i, s in enumerate(setters):
        ax2.bar(x[i], sd["E_kernel_mJ"].iloc[i], width,
                bottom=sd["E_host_mJ"].iloc[i],
                color=SETTER_PALETTE.get(s, "gray"),
                edgecolor="black", linewidth=0.5)
    for i in range(len(setters)):
        ax2.text(x[i], sd["E_model_mJ"].iloc[i],
                 f"{sd['E_model_mJ'].iloc[i]:.0f}",
                 ha="center", va="bottom", fontsize=8)
        ax2.text(x[i], sd["E_host_mJ"].iloc[i] + sd["E_kernel_mJ"].iloc[i] / 2,
                 f"{sd['E_kernel_mJ'].iloc[i]:.0f}",
                 ha="center", va="center", fontsize=8, color="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([SETTER_DISPLAY.get(s, s) for s in setters])
    ax2.set_ylabel("Energy per inference (mJ)")
    ax2.set_title("(b) Energy")

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="Kernel"),
        Patch(facecolor=OVERHEAD_COLOR, edgecolor="black", hatch="...",
              label="Host overhead"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, base_path)
    plt.close(fig)


def figure_dilution_gain(dilution: pd.DataFrame, base_path: Path,
                         baseline_label: str) -> None:
    """Per-setter Level-1 (kernel) vs Level-2 (model) EDP gain."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC_PARAMS)
    if dilution.empty:
        return

    setters = [s for s in SETTER_ORDER if s in dilution["setter"].values]
    width = 0.35
    x = np.arange(len(setters))

    l1 = [float(dilution[dilution["setter"] == s].iloc[0]["level1_gain"])
          for s in setters]
    l2 = [float(dilution[dilution["setter"] == s].iloc[0]["level2_gain"])
          for s in setters]

    fig, ax = plt.subplots(figsize=(6, 4))
    legend_drawn = False
    for i, s in enumerate(setters):
        c = SETTER_PALETTE.get(s, "gray")
        b1 = ax.bar(x[i] - width / 2, l1[i], width, color=c,
                    edgecolor="black", linewidth=0.5,
                    label="Level-1 (kernel sum)" if not legend_drawn else None)
        b2 = ax.bar(x[i] + width / 2, l2[i], width, color=c,
                    edgecolor="black", linewidth=0.5, hatch="///",
                    label="Level-2 (model)" if not legend_drawn else None)
        legend_drawn = True
        ax.text(x[i] - width / 2, l1[i], f"{l1[i]:.2f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, l2[i], f"{l2[i]:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SETTER_DISPLAY.get(s, s) for s in setters])
    ax.set_ylabel(f"EDP gain (baseline = {SETTER_DISPLAY.get(baseline_label, baseline_label)})")
    ymax = max(max(l1), max(l2), 1.0) * 1.18
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _save(fig, base_path)
    plt.close(fig)


# ---------- Table builders --------------------------------------------------


def _model_per_inference_energy(df: pd.DataFrame) -> pd.DataFrame:
    """For model rows, derive per-inference energy = npu_uj_package / n_inner.

    The runner stores per-batch totals; here we collapse to setter-level mins
    consistent with analyze.aggregate_per_setter.
    """
    model = df[df["measurement_type"] == "model"].copy()
    model["energy_per_inference_uj"] = model.apply(
        lambda r: (r["energy_uj_min_package"] / r["n_inner"]) if r["n_inner"] else 0.0,
        axis=1,
    )
    return model.groupby("setter", as_index=False).agg(
        time_us_min=("time_us_min", "min"),
        energy_per_inference_uj_min=("energy_per_inference_uj", "min"),
    )


def table1_main(df: pd.DataFrame, per_setter: pd.DataFrame,
                dilution: pd.DataFrame,
                gemm_frac: pd.DataFrame) -> pd.DataFrame:
    """Per-setter consolidated table for the dissertation main results.

    Columns:
        setter, time_us, energy_uj, edp, level1_gain, level2_gain, dilution,
        gemm_time_fraction
    """
    model = _model_per_inference_energy(df)
    model["edp"] = model["time_us_min"] * model["energy_per_inference_uj_min"]

    out = model.merge(dilution[["setter", "level1_gain", "level2_gain", "dilution"]],
                      on="setter", how="left")
    out = out.merge(gemm_frac[["setter", "gemm_time_fraction"]], on="setter", how="left")
    out = out.rename(columns={
        "time_us_min": "time_us",
        "energy_per_inference_uj_min": "energy_uj",
    })

    out["setter_display"] = out["setter"].map(lambda s: SETTER_DISPLAY.get(s, s))
    out["__order"] = out["setter"].map(lambda s: SETTER_ORDER.index(s)
                                       if s in SETTER_ORDER else 99)
    out = out.sort_values("__order").drop(columns=["__order"])
    return out[["setter", "setter_display", "time_us", "energy_uj", "edp",
                "level1_gain", "level2_gain", "dilution", "gemm_time_fraction"]]


def table2_configs(configurations_json: Path) -> pd.DataFrame:
    """Selected (P, SP, TP, d_inner) per (layer, setter) from configurations.json."""
    doc = json.loads(configurations_json.read_text())
    rows = []
    for entry in doc.get("configurations", []):
        if "error" in entry:
            continue
        shape = entry.get("shape", {})
        cfg = entry.get("config", {})
        rows.append({
            "layer":   shape.get("layer"),
            "setter":  entry.get("setter"),
            "M":       shape.get("M"),
            "K":       shape.get("K"),
            "N":       shape.get("N"),
            "P":       cfg.get("P"),
            "SP_m":    cfg.get("SP_m"),
            "SP_n":    cfg.get("SP_n"),
            "TP_m":    cfg.get("TP_m"),
            "TP_k":    cfg.get("TP_k"),
            "TP_n":    cfg.get("TP_n"),
            "d_inner": cfg.get("d_inner"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["setter_display"] = df["setter"].map(lambda s: SETTER_DISPLAY.get(s, s))
    df["__layer_order"] = df["layer"]
    df["__setter_order"] = df["setter"].map(lambda s: SETTER_ORDER.index(s)
                                            if s in SETTER_ORDER else 99)
    df = df.sort_values(["__layer_order", "__setter_order"])
    return df.drop(columns=["__layer_order", "__setter_order"]).reset_index(drop=True)


# ---------- CLI -------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures and tables.")
    parser.add_argument("--measurements-csv", type=Path, required=True)
    parser.add_argument("--configurations-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="will write figures/, tables/ under this dir")
    parser.add_argument("--baseline-setter", default="max_p")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.measurements_csv)
    if df.empty:
        print("publication_plots: measurements.csv is empty", file=sys.stderr)
        return 2

    per_setter = analyze.aggregate_per_setter(df)
    dilution = analyze.dilution_factors(per_setter, args.baseline_setter)
    gemm_frac = analyze.gemm_time_fraction(per_setter)

    fig_dir = args.output_dir / "figures"
    tab_dir = args.output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    figure_setter_breakdown(per_setter, fig_dir / "fig1_setter_breakdown",
                            baseline_setter=args.baseline_setter)
    figure_te_decomposition(per_setter, fig_dir / "fig2_te_decomposition")

    t1 = table1_main(df, per_setter, dilution, gemm_frac)
    t2 = table2_configs(args.configurations_json)
    t1.to_csv(tab_dir / "table1_main.csv", index=False)
    t2.to_csv(tab_dir / "table2_configs.csv", index=False)

    print(f"publication_plots: figures -> {fig_dir}")
    print(f"publication_plots: tables  -> {tab_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
