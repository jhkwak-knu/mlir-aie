#!/usr/bin/env python3
"""
diag_f7_alt.py — Alternative F7 layout experiment.

Same data as F7, but P_SM / P_CH|TL annotation strip moved from ABOVE the
plot area (current production layout) to BELOW the workload labels (under
the x-axis). Frees up top space for title+legend and clusters P-info near
the workload it describes, at the cost of taller bottom margin.

Outputs: output/F7_alt_t_vs_e_decomposition.png (and .pdf)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import load_config, DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON
from data_loader import load_all
from baselines import compute_all_baselines
from figures import COLOR_FW, COLOR_CHARM, COLOR_TIMELOOP


def _meas_or_pred(res, field_meas, field_pred):
    v = getattr(res, field_meas)
    if res.in_measurements and not np.isnan(v):
        return v
    return getattr(res, field_pred)


def collect_panel_rows(baselines, cfg, baseline_fn):
    rows = []
    for wl in cfg.WORKLOADS:
        bl = baselines.get(wl)
        if not bl or not bl.framework:
            continue
        base = baseline_fn(bl)
        if base is None:
            continue
        fw = bl.framework

        fw_t   = _meas_or_pred(fw,   "measured_time_us",   "pred_time")
        fw_e   = _meas_or_pred(fw,   "measured_energy_uj", "pred_energy")
        fw_edp = _meas_or_pred(fw,   "measured_edp",       "pred_edp")
        b_t    = _meas_or_pred(base, "measured_time_us",   "pred_time")
        b_e    = _meas_or_pred(base, "measured_energy_uj", "pred_energy")
        b_edp  = _meas_or_pred(base, "measured_edp",       "pred_edp")

        vals = (fw_t, fw_e, fw_edp, b_t, b_e, b_edp)
        if any(np.isnan(v) or v <= 0 for v in vals):
            continue

        rows.append({
            "wl":      wl,
            "label":   f"{wl[0]}×{wl[1]}×{wl[2]}",
            "p_sm":    fw.P,
            "p_base":  base.P,
            "t_red":   (1.0 - fw_t   / b_t)   * 100.0,
            "e_red":   (1.0 - fw_e   / b_e)   * 100.0,
            "edp_red": (1.0 - fw_edp / b_edp) * 100.0,
        })
    return rows


def main():
    cfg = load_config(DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON)
    _, groups = load_all(cfg)
    baselines = compute_all_baselines(groups, cfg)

    output_dir = Path(__file__).parent / "output"

    # Style constants (match figures.py F7)
    COL_T_BAR  = "#E76F51"
    COL_E_BAR  = "#264653"
    COL_EDP_MK = "black"

    panels_spec = [
        ("(a) STAR-Map vs CHARM-CDSE",
         lambda bl: bl.charm_spk1, "CH", COLOR_CHARM),
        ("(b) STAR-Map vs Timeloop",
         lambda bl: bl.timeloop,  "TL", COLOR_TIMELOOP),
    ]
    panel_data = [
        (title, abbr, color, collect_panel_rows(baselines, cfg, fn))
        for title, fn, abbr, color in panels_spec
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9.5), sharex=True)
    bar_w = 0.38

    for panel_idx, (title, p_abbr, p_color, rows) in enumerate(panel_data):
        ax = axes[panel_idx]
        x = np.arange(len(rows))
        t_vals   = [r["t_red"]   for r in rows]
        e_vals   = [r["e_red"]   for r in rows]
        edp_vals = [r["edp_red"] for r in rows]
        labels   = [r["label"]   for r in rows]

        ax.bar(x - bar_w / 2, t_vals, width=bar_w,
               color=COL_T_BAR, alpha=0.9,
               label="T Reduction (%)", zorder=3)
        ax.bar(x + bar_w / 2, e_vals, width=bar_w,
               color=COL_E_BAR, alpha=0.9,
               label="E Reduction (%)", zorder=3)
        ax.scatter(x, edp_vals, marker="D", s=60,
                   color=COL_EDP_MK, edgecolors="white", linewidths=1.2,
                   zorder=6, label="EDP Reduction (%)")

        for xi, yi in zip(x, edp_vals):
            ax.annotate(
                f"{yi:.0f}",
                xy=(xi, yi),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=6.5, color=COL_EDP_MK,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.75),
                zorder=7,
            )

        ax.axhline(0, color="black", lw=0.8, zorder=2)

        # ── P-annotation strip BELOW the x-axis ──
        # Because sharex=True, only the bottom panel shows rotated workload
        # labels. The P strip must clear those labels in panel (b), so we
        # use panel-specific Y coordinates.
        if panel_idx == 0:
            Y_SM, Y_BASE = -0.08, -0.14     # just below axis (no labels here)
        else:
            Y_SM, Y_BASE = -0.58, -0.64     # below workload labels

        ax.text(-0.01, Y_SM,
                r"$P_{\mathrm{SM}}=$",
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=7, color=COLOR_FW)
        ax.text(-0.01, Y_BASE,
                rf"$P_{{\mathrm{{{p_abbr}}}}}=$",
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=7, color=p_color)

        for i, r in enumerate(rows):
            ax.text(i, Y_SM, f"{r['p_sm']}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="center",
                    fontsize=7, color=COLOR_FW)
            ax.text(i, Y_BASE, f"{r['p_base']}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="center",
                    fontsize=7, color=p_color)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        # Title moved closer to plot area now (no overhead P-strip)
        ax.set_title(title, fontsize=10, pad=6)
        ax.grid(axis="y", alpha=0.25, zorder=1)
        ax.set_ylim(-10, 105)
        ax.set_ylabel("Reduction (%)", fontsize=9)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    preferred_order = ["EDP Reduction (%)", "T Reduction (%)", "E Reduction (%)"]
    ordered = [(h, l) for name in preferred_order
               for h, l in zip(handles, leg_labels) if l == name]
    if ordered:
        oh, ol = zip(*ordered)
        fig.legend(oh, ol,
                   loc="upper center", bbox_to_anchor=(0.5, 0.97),
                   ncol=len(ol), fontsize=9,
                   framealpha=0.9, edgecolor="gray")

    # Adjust layout: bottom panel needs extra room for workload-label + P-strip
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    fig.subplots_adjust(hspace=0.55)

    out_pdf = output_dir / "F7_alt_t_vs_e_decomposition.pdf"
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
