"""
figures.py — Generate paper figures from measurement data and cost model.

F5: Predicted vs Measured scatter plots (T and E, log-log)
F6: EDP Landscape Small Multiples (main result figure)
F7: T vs E decomposition diverging bar chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import Config
from baselines import WorkloadBaselines
from data_loader import WorkloadKey

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_GT = "#E63946"       # red (optimal floor — reference bound)
COLOR_FW = "#1F6FA5"       # saturated blue (pruned framework — competitive baseline)
COLOR_FW_EXH = "#6FC5CA"   # saturated cyan (exhaustive framework — family sibling of FW)
COLOR_STATIC = "#D87659"   # terracotta (Max-P GT — warm reference-bound family, distinct hue from GT's red and Timeloop's purple)
COLOR_NAIVE = COLOR_STATIC # backwards-compat alias (legacy name)
COLOR_CHARM = "#0FA086"    # vivid teal (CHARM-CDSE 1-level, SP_k=1 — competitive baseline)
COLOR_TIMELOOP = "#7E2EA0"  # vivid purple (Timeloop EDP, Cat B representative — competitive baseline)
COLOR_SCATTER = "#888888"  # darker gray (improved visibility for measurement cloud)


def _workload_label(wl: WorkloadKey) -> str:
    m, k, n = wl
    if m == k == n:
        return f"{m}³"
    return f"{m}×{k}×{n}"


def _macs(wl: WorkloadKey) -> int:
    return wl[0] * wl[1] * wl[2]


def _macs_label(wl: WorkloadKey) -> str:
    macs = _macs(wl)
    if macs >= 1e9:
        return f"{macs/1e9:.1f}G"
    elif macs >= 1e6:
        return f"{macs/1e6:.0f}M"
    elif macs >= 1e3:
        return f"{macs/1e3:.0f}K"
    return str(macs)


# ═════════════════════════════════════════════════════════════════════════════
# F5: Predicted vs Measured scatter (T and E)
# ═════════════════════════════════════════════════════════════════════════════
def figure_f5(
    df: pd.DataFrame,
    cfg: Config,
    output_dir: Path,
) -> Path:
    """F5: Predicted vs Measured scatter plots (log-log).
    
    Uses pre-computed predictions from CSV/JSON.
    T predictions (cycles) are converted to µs via clock_mhz.
    Includes ±MAPE error band and accuracy metrics annotation.
    """
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # ── Helper: draw error band + metrics ──
    def _draw_scatter(ax, meas, pred, color, color_light, label_mape, label_rho):
        """Draw scatter with diagonal, ±MAPE band, and metrics text."""
        meas = meas.values.astype(float)
        pred = pred.values.astype(float)

        # Axis limits (tight)
        all_vals = np.concatenate([meas, pred])
        lo, hi = all_vals.min() * 0.7, all_vals.max() * 1.5
        diag = np.array([lo, hi])

        # P90 error band (90% of predictions fall within this range)
        ape = np.abs(pred - meas) / meas
        mape = np.mean(ape) * 100
        p90 = np.percentile(ape, 90)
        ax.fill_between(diag, diag / (1 + p90), diag * (1 + p90),
                         alpha=0.10, color=color)

        # Diagonal
        ax.plot(diag, diag, "k-", lw=0.8, alpha=0.4)

        # Scatter
        ax.scatter(meas, pred, s=8, alpha=0.35, color=color, edgecolors="none")

        # Metrics
        rho, _ = sp_stats.spearmanr(meas, pred)
        p90_str = f"{p90*100:.1f}%"
        ax.text(0.05, 0.95,
                f"MAPE = {mape:.1f}%\n"
                f"MAPE P90 = {p90_str} (shaded)\n"
                f"ρ = {rho:.3f}\n"
                f"n = {len(meas)}",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # ── (a) Performance Model ──
    ax = axes[0]
    valid_t = df[df["t_total_pred"] > 0].copy()
    t_pred_us = valid_t["t_total_pred"] / cfg.hw.clock_mhz
    _draw_scatter(ax, valid_t["time_us"], t_pred_us,
                  COLOR_FW, "#A8DADC", "MAPE", "ρ")
    ax.set_xlabel("Measured T (µs)")
    ax.set_ylabel("Predicted T (µs)")
    ax.set_title("(a) Performance Model")

    # ── (b) Energy Model ──
    ax = axes[1]
    valid_e = df[df["energy_valid"] & (df["e_total_pred"] > 0)].copy()
    if not valid_e.empty:
        _draw_scatter(ax, valid_e["energy_uj"], valid_e["e_total_pred"],
                      COLOR_GT, "#F4A0A0", "MAPE", "ρ")
    ax.set_xlabel("Measured E (µJ)")
    ax.set_ylabel("Predicted E (µJ)")
    ax.set_title("(b) Energy Model")

    fig.tight_layout()
    out = output_dir / "F5_pred_vs_measured.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# F6: EDP Landscape Small Multiples (★ main figure)
# ═════════════════════════════════════════════════════════════════════════════
def figure_f6(
    groups: Dict[WorkloadKey, pd.DataFrame],
    baselines: Dict[WorkloadKey, WorkloadBaselines],
    cfg: Config,
    output_dir: Path,
) -> Path:
    """F6: EDP Landscape Small Multiples.

    3×7 grid, MACs ascending. Each cell: strip plot of measured EDP with:
      - Reference bounds (horizontal lines, contextual, BACK layer):
          · Ground Truth (★ + dashed, red)         → unconstrained optimum
          · Max-P Ground Truth (★ + dash-dot, terracotta)
                                                   → P-restricted optimum
            (defined as GT | P = P_max, where P_max = max feasible PE
            count per workload; may be < hardware P_max for workloads
            with restrictive granularity)
            Both reference bounds share the ★ shape to signal
            "empirically grounded / oracle" at a glance; color + line
            style distinguish the two oracle variants.
      - Competitive baselines (markers only, MIDDLE layer):
          · CHARM-CDSE (▲), Timeloop (✚), STAR-Map-exh (●)
      - Proposed method (FRONT layer, focal):
          · STAR-Map (●, smallest)

    Visual hierarchy: back-large → front-small, so small front markers
    sit visually nested inside larger back markers when overlapping
    (e.g., STAR-Map ≡ GT cases show a small dot inside a large star).

    The gap between GT and Max-P GT quantifies the pure contribution of
    adaptive P selection (resource elasticity). Quantitative contribution
    claims against external DSE use CHARM/Timeloop; Max-P GT is retained
    as a contextual reference for the value of P adaptation itself.
    """
    workloads = cfg.WORKLOADS
    n_wl = len(workloads)
    ncols, nrows = 3, 7

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 16), squeeze=False)
    p_set = cfg.hw.P_set

    # Track if legend entries have been added (only once).
    legend_added = {
        "gt": False, "fw": False, "fw_exh": False,
        "charm": False, "timeloop": False, "static_pmax": False,
    }

    for idx, wl in enumerate(workloads):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        wl_df = groups.get(wl)
        bl = baselines.get(wl)

        # Title: "M×K×N (MACs)"
        title = f"{wl[0]}×{wl[1]}×{wl[2]}\n({_macs_label(wl)} MACs)"

        if wl_df is None or wl_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
            ax.set_title(title, fontsize=7, pad=3)
            ax.set_xticks(range(len(p_set)))
            ax.set_xticklabels(p_set, fontsize=5)
            continue

        # Plot all measured EDP points as strip plot
        valid = wl_df[wl_df["energy_valid"]].copy()
        if not valid.empty:
            p_to_x = {p: i for i, p in enumerate(p_set)}
            valid["x"] = valid["numSpm"].map(p_to_x)
            valid_plot = valid.dropna(subset=["x"])

            rng = np.random.RandomState(42)
            jitter = rng.uniform(-0.2, 0.2, len(valid_plot))
            ax.scatter(
                valid_plot["x"] + jitter,
                valid_plot["edp_measured"],
                s=10, alpha=0.55, color=COLOR_SCATTER, edgecolors="none",
                zorder=2,
            )

        # ── Markers ──
        # Fixed x-offsets per baseline type to reduce occlusion when multiple
        # baselines land on the same P. GT and STAR-Map share offset=0 so the
        # "small STAR-Map ● nested inside large GT ★" motif is preserved.
        X_OFF_GT = 0.0
        X_OFF_FW = 0.0
        X_OFF_PMAX = +0.14
        X_OFF_CHARM = -0.14
        X_OFF_TL = +0.28
        X_OFF_FW_EXH = -0.28

        if bl:
            # ── Precompute reference-bound EDP values for adaptation-gap shading ──
            gt_edp = None
            if bl.gt and not np.isnan(bl.gt.measured_edp):
                gt_edp = float(bl.gt.measured_edp)

            pmax_edp = None
            if bl.naive_max:
                cand = (
                    bl.naive_max.measured_edp
                    if bl.naive_max.in_measurements and not np.isnan(bl.naive_max.measured_edp)
                    else bl.naive_max.pred_edp
                )
                if not np.isnan(cand) and cand > 0:
                    pmax_edp = float(cand)

            # Adaptation-gap shading (BACK-most layer, zorder=1):
            # shaded band between GT and Max-P GT visualizes the headroom
            # contributed by P adaptation. Skipped when the two coincide.
            if (gt_edp is not None and pmax_edp is not None
                    and not np.isclose(gt_edp, pmax_edp, rtol=0.02)):
                lo = min(gt_edp, pmax_edp)
                hi = max(gt_edp, pmax_edp)
                ax.axhspan(lo, hi, facecolor=COLOR_GT, alpha=0.06,
                           edgecolor="none", zorder=1)

            # Ground Truth (unconstrained optimum — reference bound, BACK layer):
            # ★ marker + dashed line. Largest size so small front markers
            # nest visibly inside when they coincide.
            if gt_edp is not None:
                x_gt = p_set.index(bl.gt.P) if bl.gt.P in p_set else 0
                lbl_gt = "Ground Truth" if not legend_added["gt"] else None
                ax.scatter([x_gt + X_OFF_GT], [gt_edp],
                           marker="*", s=180, color=COLOR_GT, zorder=4,
                           edgecolors="darkred", linewidths=0.8, label=lbl_gt)
                ax.axhline(gt_edp,
                           color=COLOR_GT, ls="--", lw=0.8, alpha=0.5, zorder=3)
                legend_added["gt"] = True

            # Max-P Ground Truth (P-restricted optimum — reference bound, BACK layer):
            # ★ marker (shared with GT to signal "oracle / empirically grounded")
            # + dash-dot line + terracotta color (distinct from GT's red).
            # Defined as GT | P = P_max, where P_max = max feasible PE count
            # per workload.
            # The GT ↔ Max-P GT gap = pure contribution of adaptive P selection.
            if pmax_edp is not None:
                x_pmax = (
                    p_set.index(bl.naive_max.P) if bl.naive_max.P in p_set else 0
                )
                lbl_pmax = "Max-P Ground Truth" if not legend_added["static_pmax"] else None
                # zorder=3.5 places Max-P GT ★ BEHIND GT ★ when they coincide,
                # so the unconstrained optimum (GT) is always visually dominant.
                # Size matches GT's 180 so the two oracle references carry
                # equal visual weight in the legend and the cells; color
                # (terracotta vs red) and line style (dash-dot vs dashed)
                # carry the semantic distinction between the two variants.
                ax.scatter([x_pmax + X_OFF_PMAX], [pmax_edp],
                           marker="*", s=180, color=COLOR_STATIC, zorder=3.5,
                           edgecolors="#7A3020", linewidths=0.8, label=lbl_pmax)
                ax.axhline(pmax_edp,
                           color=COLOR_STATIC, ls="-.", lw=0.8, alpha=0.5, zorder=3)
                legend_added["static_pmax"] = True

            # CHARM-CDSE (1-level, SP_k=1): marker only (competitive baseline)
            if bl.charm_spk1:
                charm_edp = (
                    bl.charm_spk1.measured_edp
                    if bl.charm_spk1.in_measurements and not np.isnan(bl.charm_spk1.measured_edp)
                    else bl.charm_spk1.pred_edp
                )
                if not np.isnan(charm_edp) and charm_edp > 0:
                    x_charm = (
                        p_set.index(bl.charm_spk1.P) if bl.charm_spk1.P in p_set else 0
                    )
                    marker_charm = "^"
                    lbl_charm = "CHARM-CDSE" if not legend_added["charm"] else None
                    ax.scatter([x_charm + X_OFF_CHARM], [charm_edp],
                               marker=marker_charm, s=60, color=COLOR_CHARM, zorder=5,
                               edgecolors="#1D6B62", linewidths=0.6, label=lbl_charm)
                    legend_added["charm"] = True

            # Timeloop (Cat B representative, EDP objective): marker only
            if bl.timeloop:
                tl_edp = (
                    bl.timeloop.measured_edp
                    if bl.timeloop.in_measurements and not np.isnan(bl.timeloop.measured_edp)
                    else bl.timeloop.pred_edp
                )
                if not np.isnan(tl_edp) and tl_edp > 0:
                    x_tl = (
                        p_set.index(bl.timeloop.P) if bl.timeloop.P in p_set else 0
                    )
                    marker_tl = "P"   # filled plus
                    lbl_tl = "Timeloop" if not legend_added["timeloop"] else None
                    ax.scatter([x_tl + X_OFF_TL], [tl_edp],
                               marker=marker_tl, s=70, color=COLOR_TIMELOOP, zorder=5,
                               edgecolors="#5B2C6F", linewidths=0.6, label=lbl_tl)
                    legend_added["timeloop"] = True

            # STAR-Map w/o pruning (MIDDLE-FRONT layer, ablation)
            if bl.framework_exhaustive:
                fwe_edp = bl.framework_exhaustive.measured_edp if bl.framework_exhaustive.in_measurements and not np.isnan(bl.framework_exhaustive.measured_edp) else bl.framework_exhaustive.pred_edp
                if not np.isnan(fwe_edp) and fwe_edp > 0:
                    x_fwe = p_set.index(bl.framework_exhaustive.P) if bl.framework_exhaustive.P in p_set else 0
                    marker_fwe = "o"
                    lbl_fwe = "STAR-Map w/o pruning" if not legend_added["fw_exh"] else None
                    ax.scatter([x_fwe + X_OFF_FW_EXH], [fwe_edp],
                               marker=marker_fwe, s=30, color=COLOR_FW_EXH, zorder=6,
                               edgecolors=COLOR_FW, linewidths=0.6, label=lbl_fwe)
                    legend_added["fw_exh"] = True

            # STAR-Map (FRONT layer, focal — proposed method, smallest).
            # offset=0 preserves the "nested inside GT" motif when they coincide.
            if bl.framework:
                fw_edp = bl.framework.measured_edp if bl.framework.in_measurements and not np.isnan(bl.framework.measured_edp) else bl.framework.pred_edp
                if not np.isnan(fw_edp) and fw_edp > 0:
                    x_fw = p_set.index(bl.framework.P) if bl.framework.P in p_set else 0
                    marker_style = "o"
                    lbl_fw = "STAR-Map" if not legend_added["fw"] else None
                    ax.scatter([x_fw + X_OFF_FW], [fw_edp],
                               marker=marker_style, s=30, color=COLOR_FW, zorder=7,
                               edgecolors="navy", linewidths=0.6, label=lbl_fw)
                    legend_added["fw"] = True

        ax.set_yscale("log")
        ax.set_xticks(range(len(p_set)))
        ax.set_xticklabels(p_set, fontsize=5, rotation=45)
        # Extend xlim to accommodate horizontal jitter offsets (±0.28)
        ax.set_xlim(-0.4, len(p_set) - 1 + 0.4)
        ax.set_title(title, fontsize=7, pad=3)

        if col == 0:
            ax.set_ylabel("EDP", fontsize=7)
        if row == nrows - 1:
            ax.set_xlabel("P (PEs)", fontsize=7)

    # Hide unused cells
    for idx in range(n_wl, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Legend at top-right
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper right",
                   ncol=len(handles), fontsize=8, framealpha=0.9,
                   edgecolor="gray", bbox_to_anchor=(0.98, 0.97))

    # Suptitle omitted — figure caption will describe the panel contents.
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    out = output_dir / "F6_edp_landscape.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# F7: T/E/EDP Reduction vs competitive baselines (2-panel)
# ═════════════════════════════════════════════════════════════════════════════
def figure_f7(
    baselines: Dict[WorkloadKey, WorkloadBaselines],
    cfg: Config,
    output_dir: Path,
) -> Path:
    """F7: Quantitative T/E/EDP decomposition vs competitive baselines.

    Two-panel layout:
      (a) STAR-Map vs CHARM-CDSE (Category A representative, SP_k=1)
      (b) STAR-Map vs Timeloop   (Category B representative, EDP objective)

    Metric — Reduction (%) := (1 - SM / baseline) × 100.
    Positive (↑) means STAR-Map improves over the baseline on that axis.
    Since EDP = T × E, we have
        (1 - R_EDP/100) = (1 - R_T/100) × (1 - R_E/100),
    so the T and E bars together decompose each EDP reduction into its
    time and energy contributions.

    Max-P GT is intentionally excluded from this figure — it is reserved
    for F6 as a contextual reference bound of the positional U-curve.
    Using Max-P GT as a Δ denominator here would risk a strawman framing.
    """
    # ── Panel-agnostic style constants ──
    COL_T_BAR   = "#E76F51"   # coral (T reduction bar)
    COL_E_BAR   = "#264653"   # dark slate (E reduction bar)
    COL_EDP_MK  = "black"     # EDP reduction diamond

    panels_spec = [
        ("(a) STAR-Map vs CHARM-CDSE",
         lambda bl: bl.charm_spk1, "CH", COLOR_CHARM),
        ("(b) STAR-Map vs Timeloop",
         lambda bl: bl.timeloop,  "TL", COLOR_TIMELOOP),
    ]

    def _meas_or_pred(res, field_meas, field_pred):
        v = getattr(res, field_meas)
        if res.in_measurements and not np.isnan(v):
            return v
        return getattr(res, field_pred)

    # ── Collect per-panel rows ──
    panel_data = []
    for title, baseline_fn, p_abbr, p_color in panels_spec:
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
                "wl":       wl,
                "label":    f"{wl[0]}×{wl[1]}×{wl[2]}",
                "p_sm":     fw.P,
                "p_base":   base.P,
                "t_red":    (1.0 - fw_t   / b_t)   * 100.0,
                "e_red":    (1.0 - fw_e   / b_e)   * 100.0,
                "edp_red":  (1.0 - fw_edp / b_edp) * 100.0,
            })
        panel_data.append((title, p_abbr, p_color, rows))

    if not any(rows for _, _, _, rows in panel_data):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No valid data for T/E/EDP Reduction decomposition",
                ha="center", va="center", transform=ax.transAxes)
        out = output_dir / "F7_t_vs_e_decomposition.pdf"
        fig.savefig(out)
        plt.close(fig)
        return out

    # Vertical stack: (a) on top, (b) on bottom, shared x-axis
    fig, axes = plt.subplots(2, 1, figsize=(11, 9.5), sharex=True)
    bar_w = 0.38

    for panel_idx, (title, p_abbr, p_color, rows) in enumerate(panel_data):
        ax = axes[panel_idx]

        if not rows:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title(title, fontsize=10, pad=6)
            ax.set_xticks([])
            continue

        x = np.arange(len(rows))
        t_vals   = [r["t_red"]   for r in rows]
        e_vals   = [r["e_red"]   for r in rows]
        edp_vals = [r["edp_red"] for r in rows]
        labels   = [r["label"]   for r in rows]

        # ── Bars: T and E reductions ──
        ax.bar(x - bar_w / 2, t_vals, width=bar_w,
               color=COL_T_BAR, alpha=0.9,
               label="T Reduction (%)", zorder=3)
        ax.bar(x + bar_w / 2, e_vals, width=bar_w,
               color=COL_E_BAR, alpha=0.9,
               label="E Reduction (%)", zorder=3)

        # ── EDP diamonds (white edge for contrast against dark bars) ──
        ax.scatter(x, edp_vals, marker="D", s=60,
                   color=COL_EDP_MK, edgecolors="white", linewidths=1.2,
                   zorder=6, label="EDP Reduction (%)")

        # EDP numeric labels — the primary metric deserves explicit values.
        # Positioned just above each ◆ with a small white halo for contrast
        # against any overlapping bar/grid element.
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

        # ── Top P-annotation strip (outside plot, just above y=100) ──
        # Row headers live in the LEFT MARGIN of the axes (outside the
        # plot area, to the left of the y-axis spine); the plot area
        # itself carries only the numeric values per workload.
        #
        # Layout:
        #   Header X (axes-fraction):  -0.01  (ha="right")
        #   Numbers X (data):           i       (ha="center")
        #   y=1.02  baseline row:  "P_CH ="     32   32   32   ...
        #   y=1.08  proposed row:  "P_SM ="      4    4    4   ...
        Y_BASE, Y_SM = 1.02, 1.08

        # Row headers in the axes' left margin (outside plot area)
        ax.text(-0.01, Y_BASE,
                rf"$P_{{\mathrm{{{p_abbr}}}}}=$",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7, color=p_color)
        ax.text(-0.01, Y_SM,
                r"$P_{\mathrm{SM}}=$",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7, color=COLOR_FW)

        # Per-workload numeric values (inside plot area, aligned with bars)
        for i, r in enumerate(rows):
            ax.text(i, Y_BASE, f"{r['p_base']}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=7, color=p_color)
            ax.text(i, Y_SM, f"{r['p_sm']}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=7, color=COLOR_FW)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        # Default xlim (no left-margin carve-out needed — headers sit outside)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        # Title placed above the P-header strip (strip ends at y≈1.10)
        ax.set_title(title, fontsize=10, y=1.14)
        ax.grid(axis="y", alpha=0.25, zorder=1)
        # Top headroom accommodates EDP value labels above ◆ markers
        ax.set_ylim(-10, 105)
        ax.set_ylabel("Reduction (%)", fontsize=9)

    # ── Figure-level legend at the very top (shared across both panels) ──
    # Explicit order: EDP (primary metric) → T → E
    handles, leg_labels = axes[0].get_legend_handles_labels()
    preferred_order = ["EDP Reduction (%)", "T Reduction (%)", "E Reduction (%)"]
    ordered = [(h, l) for name in preferred_order
               for h, l in zip(handles, leg_labels) if l == name]
    if ordered:
        oh, ol = zip(*ordered)
        fig.legend(oh, ol,
                   loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=len(ol), fontsize=9,
                   framealpha=0.9, edgecolor="gray")

    # Suptitle omitted — figure caption will describe the panel contents.
    # Top-of-figure stacking (from top to bottom):
    #   [figure legend at y≈0.955]  ← fig.legend bbox_to_anchor
    #   [gap]                        ← controlled by rect top
    #   [panel (a) title + P-strip]
    #   [panel (a) plot area]
    #   [gap]                        ← controlled by hspace
    #   [panel (b) title + P-strip]
    #   [panel (b) plot area + x-labels]
    # Tuning knobs:
    #   rect top  (smaller = more room between legend and panel (a))
    #   hspace   (smaller = tighter gap between (a) and (b))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.915))
    fig.subplots_adjust(hspace=0.35)
    out = output_dir / "F7_t_vs_e_decomposition.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# F8: NPU vs CPU vs GPU 3-way comparison (T / E / EDP Reduction, %)
# ═════════════════════════════════════════════════════════════════════════════
def figure_f8(
    output_dir: Path,
    cfg: Optional[Config] = None,
    groups: Optional[Dict[WorkloadKey, pd.DataFrame]] = None,
    baselines: Optional[Dict] = None,
) -> Path:
    """F8: NPU-optimal versus CPU (1T/24T) and iGPU on T / E / EDP.

    Input — wide-form `cpu_npu_gpu_combined_v14.csv` (one row per workload,
    NPU/CPU-1T/CPU-24T/GPU columns). 768³ rows with `npu_t_meas_us == -1`
    are skipped automatically.

    Metric — Reduction (%) := (1 − NPU / baseline) × 100.
      * Same convention as F7 (figures.py:427): positive (↑) means NPU
        improves over the baseline on that axis. T, E, and EDP Reductions
        are commensurate, so the three panels can be read together.

    Three baselines are plotted as separate lines on each panel:
      CPU-1T  — sequential CPU reference
      CPU-24T — parallel CPU reference
      GPU     — same-SoC iGPU reference (Radeon 890M, fp32)

    Three panels (consistent with the prior F8 layout):
      (a) Time Reduction (%)
      (b) Energy Reduction (%)
      (c) EDP Reduction (%)

    A 0% reference line marks parity with each baseline. Values that
    fall below the y-axis lower bound (small workloads where NPU is
    far behind CPU) are clipped and annotated with a downward arrow
    + numeric value.
    """
    from config import load_config
    if cfg is None:
        cfg = load_config()

    # ── Load wide-form combined CSV ──
    df = pd.read_csv(cfg.cpu_energy_csv, comment="#")

    # ── Build per-workload rows in cfg.WORKLOADS order ──
    rows = []
    for wl in cfg.WORKLOADS:
        m, k, n = wl
        sub = df[(df["M"] == m) & (df["K"] == k) & (df["N"] == n)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        # Skip workloads with missing NPU measurement (sentinel = -1)
        if r["npu_t_meas_us"] <= 0 or r["npu_e_meas_uj"] <= 0:
            continue
        rows.append({
            "label":   f"{wl[0]}×{wl[1]}×{wl[2]}",
            "npu_t":   float(r["npu_t_meas_us"]),
            "npu_e":   float(r["npu_e_meas_uj"]),
            "npu_edp": float(r["npu_edp_meas"]),
            "c1t_t":   float(r["cpu_1t_t_us"]),
            "c1t_e":   float(r["cpu_1t_e_uj"]),
            "c1t_edp": float(r["cpu_1t_edp"]),
            "c24_t":   float(r["cpu_24t_t_us"]),
            "c24_e":   float(r["cpu_24t_e_uj"]),
            "c24_edp": float(r["cpu_24t_edp"]),
            "gpu_t":   float(r["gpu_t_us"]),
            "gpu_e":   float(r["gpu_e_uj"]),
            "gpu_edp": float(r["gpu_edp"]),
        })

    labels = [r["label"] for r in rows]
    n_wl = len(rows)
    x = np.arange(n_wl)

    def _red(npu_vals, base_vals):
        """Reduction (%) = (1 - NPU / baseline) × 100. Positive ⇒ NPU better."""
        npu = np.asarray(npu_vals, dtype=float)
        base = np.asarray(base_vals, dtype=float)
        return (1.0 - npu / base) * 100.0

    npu_t   = np.array([r["npu_t"]   for r in rows])
    npu_e   = np.array([r["npu_e"]   for r in rows])
    npu_edp = np.array([r["npu_edp"] for r in rows])

    # Per-baseline reductions on each metric
    red_T = {
        "CPU-1T":  _red(npu_t, [r["c1t_t"] for r in rows]),
        "CPU-24T": _red(npu_t, [r["c24_t"] for r in rows]),
        "GPU":     _red(npu_t, [r["gpu_t"] for r in rows]),
    }
    red_E = {
        "CPU-1T":  _red(npu_e, [r["c1t_e"] for r in rows]),
        "CPU-24T": _red(npu_e, [r["c24_e"] for r in rows]),
        "GPU":     _red(npu_e, [r["gpu_e"] for r in rows]),
    }
    red_EDP = {
        "CPU-1T":  _red(npu_edp, [r["c1t_edp"] for r in rows]),
        "CPU-24T": _red(npu_edp, [r["c24_edp"] for r in rows]),
        "GPU":     _red(npu_edp, [r["gpu_edp"] for r in rows]),
    }

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    panels = [
        ("(a) Time Reduction (%)",   red_T),
        ("(b) Energy Reduction (%)", red_E),
        ("(c) EDP Reduction (%)",    red_EDP),
    ]

    # Color/marker — F7-aligned palette where possible:
    #   CPU-1T  : neutral gray (sequential reference)
    #   CPU-24T : F7 T-bar coral (parallel CPU competitor)
    #   GPU     : amber (new same-SoC iGPU competitor)
    style = {
        "CPU-1T":  {"color": "#666666", "marker": "s", "ls": "-",  "label": "vs CPU-1T"},
        "CPU-24T": {"color": "#E76F51", "marker": "^", "ls": "-",  "label": "vs CPU-24T"},
        "GPU":     {"color": "#F4A261", "marker": "o", "ls": "-",  "label": "vs GPU (iGPU)"},
    }
    series_order = ["CPU-1T", "CPU-24T", "GPU"]

    # Y-axis: symlog with a practical clip at -10⁴% to avoid wasting vertical
    # space on the single EDP outlier at 32×32×32 (~-2.5×10⁵% vs CPU-1T).
    #   * |y| ≤ linthresh is rendered linearly (readable near the 0% line)
    #   * |y| > linthresh is rendered on log scale in both directions
    # Points falling below Y_LO are plotted *at* Y_LO and annotated with
    # their true value (compact "k%" form).
    Y_LINTHRESH = 100.0
    Y_LO, Y_HI = -1e4, 1e2

    for i, (title, red_dict) in enumerate(panels):
        ax = axes[i]

        # Per-panel counter so multiple clipped annotations can be staggered
        # vertically (prevents horizontal overlap at tight x-spacing).
        clipped_idx = 0

        for name in series_order:
            vals = red_dict[name]
            sty = style[name]

            # Clip values below Y_LO; they are re-inserted via annotation below.
            plot_vals = np.where(vals < Y_LO, Y_LO, vals)
            ax.plot(x, plot_vals,
                    marker=sty["marker"], ms=4.5, lw=1.2,
                    color=sty["color"], ls=sty["ls"],
                    label=sty["label"] if i == 0 else None,
                    zorder=4)

            # Annotate clipped points with their true reduction in "k%" form.
            # Stagger y-offset (4pt / 12pt) to avoid overlap when adjacent x
            # positions are both clipped.
            for xi, vi in zip(x, vals):
                if vi < Y_LO:
                    y_off = 4 + (clipped_idx % 2) * 8
                    ax.annotate(
                        f"↓{vi/1000:.0f}k%",
                        xy=(xi, Y_LO), xytext=(0, y_off),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=6, color=sty["color"], alpha=0.9,
                        zorder=6,
                    )
                    clipped_idx += 1

        # 0% reference line (parity with baseline)
        ax.axhline(0.0, color="black", ls="--", lw=0.8, alpha=0.6, zorder=2)

        ax.set_yscale("symlog", linthresh=Y_LINTHRESH, linscale=0.6)
        ax.set_ylim(Y_LO, Y_HI)
        # Explicit yticks — skip ±10 to reduce clutter; keep 100 at top boundary.
        ax.set_yticks([100, 0, -100, -1000, -10000])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_title(title, fontsize=9)
        if i == 0:
            ax.set_ylabel("Reduction (%)", fontsize=8)
        ax.grid(axis="y", which="both", alpha=0.3, zorder=1)

    # ── Shared legend at the top of the figure, applies to all panels ──
    # Reserve top headroom via tight_layout(rect=...) so the legend sits
    # above the panel titles without overlap.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    handles, labels_lg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_lg,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(handles),
        fontsize=8,
        frameon=True,
    )
    out = output_dir / "F8_npu_vs_cpu.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out