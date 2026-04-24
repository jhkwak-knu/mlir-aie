"""
tables.py — Generate paper tables as LaTeX and CSV.

T5: Workload list
T6: Performance model calibration coefficients + accuracy
T7: Energy model calibration coefficients + accuracy
T8: EDP reduction quantification per workload
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import Config, DEFAULT_SEARCH_BENCH_CSV
from baselines import WorkloadBaselines, enumerate_configs, enumerate_configs_pruned
from cost_model import tporder_name, edp
from data_loader import WorkloadKey


def _wl_label(wl: WorkloadKey) -> str:
    m, k, n = wl
    return f"${m}\\times{k}\\times{n}$"


def _macs_str(wl: WorkloadKey) -> str:
    macs = wl[0] * wl[1] * wl[2]
    if macs >= 1e9:
        return f"{macs/1e9:.1f}G"
    elif macs >= 1e6:
        return f"{macs/1e6:.0f}M"
    elif macs >= 1e3:
        return f"{macs/1e3:.0f}K"
    return str(macs)


# ═════════════════════════════════════════════════════════════════════════════
# T8: Workload list  (Paper Table 8)
# ═════════════════════════════════════════════════════════════════════════════
def table_t8(
    groups: Dict[WorkloadKey, pd.DataFrame],
    cfg: Config,
    output_dir: Path,
) -> Path:
    """T8: Workload list with MACs and description."""
    rows = []
    for wl in cfg.WORKLOADS:
        rows.append({
            "Workload": _wl_label(wl),
            "MACs": _macs_str(wl),
            "Description": cfg.WORKLOAD_DESC.get(wl, ""),
        })

    df = pd.DataFrame(rows)
    out_csv = output_dir / "T8_workloads.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # LaTeX (tabularx for full-width)
    out_tex = output_dir / "T8_workloads.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabularx}{\\textwidth}{lrX}\n\\toprule\n")
        f.write("Workload & MACs & Description \\\\\n\\midrule\n")
        for _, row in df.iterrows():
            f.write(f"{row['Workload']} & {row['MACs']} & {row['Description']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabularx}\n")

    return out_tex


# ═════════════════════════════════════════════════════════════════════════════
# T9: Performance model accuracy  (Paper Table 9)
# ═════════════════════════════════════════════════════════════════════════════
def table_t9(
    df: pd.DataFrame,
    cfg: Config,
    output_dir: Path,
) -> Path:
    """T9: Performance model accuracy metrics.

    Uses CSV pre-computed t_total_pred (cycles), converted to µs via clock_mhz.
    """
    valid = df[(df["t_total_pred"] > 0) & (df["time_us"] > 0)].copy()

    if valid.empty:
        out = output_dir / "T9_perf_accuracy.tex"
        with open(out, "w", encoding="utf-8") as f:
            f.write("% No valid data for performance model accuracy\n")
        return out

    # Convert predicted cycles to µs
    pred = (valid["t_total_pred"] / cfg.hw.clock_mhz).values
    meas = valid["time_us"].values
    ape = np.abs(pred - meas) / meas
    mape = np.mean(ape) * 100
    p90 = np.percentile(ape, 90) * 100
    rho, _ = sp_stats.spearmanr(meas, pred)

    # Within-workload rho (how well does the model rank configs within each workload?)
    rhos = []
    for wl in cfg.WORKLOADS:
        wl_valid = valid[(valid["M"] == wl[0]) & (valid["K"] == wl[1]) & (valid["N"] == wl[2])]
        if len(wl_valid) >= 3:
            r, _ = sp_stats.spearmanr(wl_valid["time_us"], wl_valid["t_total_pred"] / cfg.hw.clock_mhz)
            if not np.isnan(r):
                rhos.append(r)
    within_rho = np.mean(rhos) if rhos else np.nan

    metrics = {
        "MAPE (\\%)": f"{mape:.1f}",
        "MAPE P90 (\\%)": f"{p90:.1f}",
        "Spearman $\\rho$": f"{rho:.3f}",
        "Within-workload $\\rho$": f"{within_rho:.3f}" if not np.isnan(within_rho) else "N/A",
        "\\# Samples": str(len(valid)),
    }

    # Coefficients (with units)
    p = cfg.perf
    coeffs = {
        "$\\eta_{\\mathrm{mac}}$ (MACs/cy)": f"{p.eff_macs:.2f}",
        "$\\beta_{\\mathrm{dma}}$ (B/cy)": f"{p.bw_eff_bpc:.1f}",
        "$L_{\\mathrm{SYNC}}$ (cy)": f"{int(p.L_SYNC)}",
        "$L_{\\mathrm{PE}}$ (cy)": f"{int(p.L_CORE)}",
        "$L_{\\mathrm{DMA}}$ (cy)": f"{int(p.L_DMA)}",
        "$L_{\\mathrm{STARTUP}}$ (cy)": f"{int(p.L_STARTUP)}",
    }

    out = output_dir / "T9_perf_accuracy.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\\begin{minipage}[t]{0.48\\textwidth}\n\\centering\n")
        f.write("\\begin{tabularx}{\\linewidth}{Xr}\n\\toprule\n")
        f.write("Coefficient & Value \\\\\n\\midrule\n")
        for k, v in coeffs.items():
            f.write(f"{k} & {v} \\\\\n")
        f.write("\\bottomrule\n\\end{tabularx}\n")
        f.write("\\end{minipage}\\hfill\n")

        f.write("\\begin{minipage}[t]{0.48\\textwidth}\n\\centering\n")
        f.write("\\begin{tabularx}{\\linewidth}{Xr}\n\\toprule\n")
        f.write("Metric & Value \\\\\n\\midrule\n")
        for k, v in metrics.items():
            f.write(f"{k} & {v} \\\\\n")
        f.write("\\bottomrule\n\\end{tabularx}\n")
        f.write("\\end{minipage}\n")

    return out


# ═════════════════════════════════════════════════════════════════════════════
# T10: Energy model accuracy  (Paper Table 10)
# ═════════════════════════════════════════════════════════════════════════════
def table_t10(
    df: pd.DataFrame,
    cfg: Config,
    output_dir: Path,
) -> Path:
    """T10: Energy model accuracy metrics.

    Uses JSON pre-computed e_total_pred (µJ).
    """
    valid = df[df["energy_valid"] & (df["e_total_pred"] > 0)].copy()

    if valid.empty:
        out = output_dir / "T10_energy_accuracy.tex"
        with open(out, "w", encoding="utf-8") as f:
            f.write("% No valid data for energy model accuracy\n")
        return out

    pred = valid["e_total_pred"].values
    meas = valid["energy_uj"].values
    ape = np.abs(pred - meas) / meas
    mape = np.mean(ape) * 100
    p90 = np.percentile(ape, 90) * 100
    rho, _ = sp_stats.spearmanr(meas, pred)

    # Within-workload rho
    rhos = []
    for wl in cfg.WORKLOADS:
        wl_valid = valid[(valid["M"] == wl[0]) & (valid["K"] == wl[1]) & (valid["N"] == wl[2])]
        if len(wl_valid) >= 3:
            r, _ = sp_stats.spearmanr(wl_valid["energy_uj"], wl_valid["e_total_pred"])
            if not np.isnan(r):
                rhos.append(r)
    within_rho = np.mean(rhos) if rhos else np.nan

    # Coefficients (with units) — v16 4-term decomposition
    e = cfg.energy
    coeffs = {
        "$P_{\\mathrm{PE}}$ (mW)": f"{e.P_CORE:.1f}",
        "$E_{\\mathrm{BYTE}}$ (nJ/B)": f"{e.E_BYTE * 1000:.2f}",
        "$E_{\\mathrm{STARTUP}}$ ($\\mu$J)": f"{e.E_STARTUP:.1f}",
        "$P_{\\mathrm{BASE}}$ (W)": f"{e.P_BASE:.2f}",
    }

    metrics = {
        "MAPE (\\%)": f"{mape:.1f}",
        "MAPE P90 (\\%)": f"{p90:.1f}",
        "Spearman $\\rho$": f"{rho:.3f}",
        "Within-workload $\\rho$": f"{within_rho:.3f}" if not np.isnan(within_rho) else "N/A",
        "\\# Samples": str(len(valid)),
    }

    out = output_dir / "T10_energy_accuracy.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\\begin{minipage}[t]{0.48\\textwidth}\n\\centering\n")
        f.write("\\begin{tabularx}{\\linewidth}{Xr}\n\\toprule\n")
        f.write("Coefficient & Value \\\\\n\\midrule\n")
        for k, v in coeffs.items():
            f.write(f"{k} & {v} \\\\\n")
        f.write("\\bottomrule\n\\end{tabularx}\n")
        f.write("\\end{minipage}\\hfill\n")

        f.write("\\begin{minipage}[t]{0.48\\textwidth}\n\\centering\n")
        f.write("\\begin{tabularx}{\\linewidth}{Xr}\n\\toprule\n")
        f.write("Metric & Value \\\\\n\\midrule\n")
        for k, v in metrics.items():
            f.write(f"{k} & {v} \\\\\n")
        f.write("\\bottomrule\n\\end{tabularx}\n")
        f.write("\\end{minipage}\n")

    return out


# ═════════════════════════════════════════════════════════════════════════════
# T11: EDP reduction quantification  (Paper Table 11)
# ═════════════════════════════════════════════════════════════════════════════
def table_t11(
    baselines: Dict[WorkloadKey, WorkloadBaselines],
    cfg: Config,
    output_dir: Path,
) -> Path:
    """T11 (Option 1): Ratio-only compact design, 8 columns.

    Each method cell is a single value:
        - GT column:      absolute EDP (serves as ratio denominator)
        - Other columns:  ratio EDP_method / EDP_GT, formatted "X.XX×"

    Columns: Workload | MACs | GT | Max-P GT | CHARM | Timeloop | SM-exh | STAR-Map

    Rationale for dropping (P, SP_m, SP_n, d_inner) per cell:
      - Per-method mapping choices are visualized in Figure F6 (landscape).
      - This keeps T11 focused on the *quantitative* EDP message.
      - Table height and width both shrink to roughly half of Option 2.
      - Reproducibility details (per-workload configs) are relegated to
        supplementary material (Option 2 full table).
    """
    # Method catalog: (column_label, accessor, cell_mode)
    #   cell_mode ∈ {"absolute", "ratio"}
    METHODS = [
        ("GT",        lambda bl: bl.gt,                   "absolute"),
        ("Max-P GT",  lambda bl: bl.naive_max,            "ratio"),
        ("CHARM",     lambda bl: bl.charm_spk1,           "ratio"),
        ("Timeloop",  lambda bl: bl.timeloop,             "ratio"),
        ("SM-exh",    lambda bl: bl.framework_exhaustive, "ratio"),
        ("STAR-Map",  lambda bl: bl.framework,            "ratio"),
    ]

    def _edp_of(res) -> float:
        """Return measured EDP if in_measurements, else predicted EDP."""
        if res is None:
            return float("nan")
        if res.in_measurements and not np.isnan(res.measured_edp):
            return float(res.measured_edp)
        return float(res.pred_edp)

    def _is_fallback_marked(res) -> bool:
        """Whether the reported EDP came from fallback substitution."""
        if res is None:
            return False
        return (not res.in_measurements) or getattr(res, "is_fallback", False)

    def _sci_tex(val: float, digits: int = 2) -> str:
        """LaTeX scientific notation: $m.mm \\times 10^{e}$.

        Unit: [uJ.us] (= time_us * energy_uj from the measurement pipeline).
        """
        if np.isnan(val) or val <= 0:
            return "--"
        exp = int(np.floor(np.log10(val)))
        mantissa = val / (10 ** exp)
        return f"${mantissa:.{digits}f}\\!\\times\\!10^{{{exp}}}$"

    def _sci_plain(val: float, digits: int = 2) -> str:
        """Plain-text scientific notation for CSV output: m.mme+e."""
        if np.isnan(val) or val <= 0:
            return "--"
        exp = int(np.floor(np.log10(val)))
        mantissa = val / (10 ** exp)
        return f"{mantissa:.{digits}f}e{exp}"

    rows = []
    for wl in cfg.WORKLOADS:
        bl = baselines.get(wl)
        if not bl:
            continue

        gt_res = bl.gt
        gt_edp = _edp_of(gt_res)
        has_gt = (gt_res is not None) and (not np.isnan(gt_edp)) and (gt_edp > 0)

        row = {"wl": _wl_label(wl), "macs": _macs_str(wl), "cells_tex": [], "cells_csv": []}

        for col_label, getter, mode in METHODS:
            res = getter(bl)
            edp_val = _edp_of(res)
            fb = _is_fallback_marked(res)
            # Fallback marker: removed from LaTeX body for F6 consistency
            # (methodology section documents fallback policy; supplementary
            # table retains per-cell fallback flags). CSV keeps the marker
            # for downstream reproducibility/ablation analysis.
            dagger_plain = "†" if fb else ""

            if mode == "absolute":
                if not np.isnan(edp_val) and edp_val > 0:
                    cell_tex = _sci_tex(edp_val)
                    cell_plain = f"{_sci_plain(edp_val)}{dagger_plain}"
                else:
                    cell_tex = cell_plain = "--"
            else:  # ratio
                if has_gt and not np.isnan(edp_val) and edp_val > 0:
                    ratio = edp_val / gt_edp
                    cell_tex = f"{ratio:.2f}$\\times$"
                    cell_plain = f"{ratio:.2f}x{dagger_plain}"
                else:
                    cell_tex = cell_plain = "--"

            row["cells_tex"].append(cell_tex)
            row["cells_csv"].append((col_label, cell_plain))

        rows.append(row)

    # ── CSV output (single value per method column) ──
    csv_rows = []
    for row in rows:
        rec = {"Workload": row["wl"], "MACs": row["macs"]}
        for (label, value_plain) in row["cells_csv"]:
            rec[label] = value_plain
        csv_rows.append(rec)
    df_csv = pd.DataFrame(csv_rows)
    out_csv = output_dir / "T11_edp_reduction.csv"
    df_csv.to_csv(out_csv, index=False, encoding="utf-8")

    # ── LaTeX output ──
    # 8 columns: 2 ID + 6 methods. All method columns centered.
    out_tex = output_dir / "T11_edp_reduction.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% Requires: \\usepackage{booktabs}\n")
        f.write("\\begin{tabular}{ll|crrrrr}\n")
        f.write("\\toprule\n")
        # Header row 1: method names
        f.write("Workload & MACs")
        for col_label, _, _ in METHODS:
            f.write(f" & {col_label}")
        f.write(" \\\\\n")
        # Header row 2: quantity + unit (two empty leading cells for Workload+MACs).
        #   GT column annotates both the quantity (EDP) and the unit
        #   ([uJ.us] = time_us * energy_uj, equivalently pJ.s).
        #   Other columns are dimensionless ratios.
        # Sub-header (quantity/unit annotations) is rendered one step smaller
        # than the main header row to establish a visual hierarchy: the main
        # row names the column, the sub row annotates dimension/unit.
        # This also mirrors typical typographic convention for scientific
        # tables where unit labels are smaller than variable names.
        sub_cells = []
        for _, _, mode in METHODS:
            if mode == "absolute":
                sub_cells.append(
                    "\\footnotesize(EDP\\,[$\\mu$J$\\cdot\\mu$s])"
                )
            else:
                sub_cells.append("\\footnotesize(ratio)")
        f.write(" &  & " + " & ".join(sub_cells))
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        for row in rows:
            f.write(f"{row['wl']} & {row['macs']}")
            for cell in row["cells_tex"]:
                f.write(f" & {cell}")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\n% Legend:\n")
        f.write("%   GT column shows absolute EDP in [uJ.us] (= time_us * energy_uj),\n")
        f.write("%   formatted in scientific notation. Other columns show dimensionless\n")
        f.write("%   ratio EDP_method/EDP_GT; 1.00x means a match with GT.\n")
        f.write("%   Fallback markers are intentionally omitted from this body table for\n")
        f.write("%   consistency with Figure F6; the F3+D2 nearest-neighbor fallback policy\n")
        f.write("%   is documented in the Methodology section, and per-cell fallback flags\n")
        f.write("%   are retained in the CSV sidecar and in the supplementary full table.\n")
        f.write("%   Per-cell (P, SP_m, SP_n, d_inner) choices are visualized in F6.\n")

    return out_tex


# ═════════════════════════════════════════════════════════════════════════════
# T12: Pruning effectiveness summary  (Paper Table 12)
# ═════════════════════════════════════════════════════════════════════════════
def table_t12(
    cfg: Config,
    output_dir: Path,
    groups: Optional[Dict[WorkloadKey, pd.DataFrame]] = None,
    search_bench_csv: Optional[Path] = None,
) -> Path:
    """T12: Pruning effectiveness — one row per cumulative rule application
    (Rule 1, Rule 1+2, Rule 1+2+3) summarizing three aspects of STAR-Map
    pruning:

    * **Reduction (%)** — per-workload mean of
      :math:`1 - |\\Omega_{\\text{rule}}| / |\\Omega|`,
      the fraction of the exhaustive configuration space eliminated by the
      rule. Averaged across workloads because pruning operates per workload;
      aggregating ratios over the pooled total would follow
      :math:`1 - \\sum|\\Omega_{\\text{rule}}| / \\sum|\\Omega|` which is
      weighted by large workloads and conflates the measure (Jensen's
      inequality).
    * **Meas.\\ max regret (%)** — worst-case EDP regret of the pruned
      argmin versus the exhaustive argmin, both evaluated on actual XDNA2
      measurements (F3+D2 nearest-neighbor fallback by predicted EDP when a
      config is not directly measured). This is the reviewer-facing
      "worst case for STAR-Map" number.
    * **Speedup (× geomean)** — geometric mean of
      :math:`t_{\\text{exh}} / t_{\\text{rule}}` across workloads, using
      wall-clock search times measured by ``search_bench.py``. Geomean is
      preferred over arithmetic mean for ratio quantities and mirrors the
      convention used in T13.

    Parameters
    ----------
    search_bench_csv :
        Optional path to the search-time benchmark CSV produced by
        ``search_bench.py`` (columns ``rule1_speedup``, ``rule12_speedup``,
        ``rule123_speedup``, one row per workload). If ``None`` (default),
        the function auto-detects ``<output_dir>/T13_search_time.csv``.
        When no CSV is available, the Speedup column is emitted as ``NA``
        and a warning is printed — this allows ``main.py`` to still
        regenerate T12 without the measurement environment.
    """
    from baselines import find_framework_optimal

    rule_configs = [
        ("Rule 1",     "rule1",   dict(rule1=True, rule2=False, rule3=False)),
        ("Rule 1+2",   "rule12",  dict(rule1=True, rule2=True,  rule3=False)),
        ("Rule 1+2+3", "rule123", dict(rule1=True, rule2=True,  rule3=True)),
    ]

    # ── Collect per-workload reductions and measured regrets ──
    stats = {
        label: {"reductions": [], "regrets_meas": []}
        for label, _, _ in rule_configs
    }

    for wl in cfg.WORKLOADS:
        M, K, N = wl
        wl_df = groups.get(wl) if groups is not None else None

        # Exhaustive reference (SM-exh): canonical argmin with F3+D2 fallback
        exh_cfgs = enumerate_configs(M, K, N, cfg)
        n_exh = len(exh_cfgs)
        exh_res = find_framework_optimal(M, K, N, cfg, wl_df, pruned=False)
        if exh_res is None:
            continue
        edp_exh_meas = float(exh_res.measured_edp) \
            if not np.isnan(exh_res.measured_edp) else np.nan

        for label, _key, kw in rule_configs:
            pru = enumerate_configs_pruned(M, K, N, cfg, **kw)
            pru_res = find_framework_optimal(
                M, K, N, cfg, wl_df, pruned=True, rule_kwargs=kw
            )
            if pru_res is None:
                continue

            red = (1.0 - len(pru) / n_exh) * 100.0
            if not np.isnan(edp_exh_meas) and edp_exh_meas > 0 \
                    and not np.isnan(pru_res.measured_edp):
                reg_meas = (pru_res.measured_edp / edp_exh_meas - 1.0) * 100.0
            else:
                reg_meas = np.nan

            stats[label]["reductions"].append(red)
            stats[label]["regrets_meas"].append(reg_meas)

    # ── Load search-time benchmark (T13) for Speedup column ──
    def _gmean_pos(xs):
        xs = np.asarray(xs, dtype=float)
        xs = xs[np.isfinite(xs) & (xs > 0)]
        return float(np.exp(np.log(xs).mean())) if xs.size else float("nan")

    # Search order for the search-time benchmark CSV:
    #   1. explicit argument (if given)
    #   2. fig/data/T13_search_time.csv  (DEFAULT_SEARCH_BENCH_CSV, measurement
    #      artifact alongside result_v14.csv / calibration.json)
    #   3. <output_dir>/T13_search_time.csv  (legacy location, kept for
    #      back-compat with users who place it beside other generated outputs)
    if search_bench_csv is not None:
        sb_candidates = [Path(search_bench_csv)]
    else:
        sb_candidates = [
            DEFAULT_SEARCH_BENCH_CSV,
            output_dir / "T13_search_time.csv",
        ]
    sb_csv = next((p for p in sb_candidates if p.exists()), None)

    speedup_gmean = {label: float("nan") for label, _, _ in rule_configs}
    if sb_csv is not None:
        sb_df = pd.read_csv(sb_csv)
        for label, key, _ in rule_configs:
            col = f"{key}_speedup"
            if col in sb_df.columns and len(sb_df) > 0:
                speedup_gmean[label] = _gmean_pos(sb_df[col].to_numpy())
        print(f"  [table_t12] using search-time CSV: {sb_csv}")
    else:
        print(
            f"  [table_t12] search-time CSV not found "
            f"(looked in {', '.join(str(p) for p in sb_candidates)}); "
            f"Speedup column will be NA. Run search_bench.py first."
        )

    # ── Row assembly ──
    def _nanmax(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return np.max(xs) if xs else np.nan

    def _fmt(v, unit=""):
        return f"{v:.1f}{unit}" if np.isfinite(v) else "NA"

    rows = []
    for label, _, _ in rule_configs:
        rows.append({
            "Rule":             label,
            "Reduction_pct":    float(np.mean(stats[label]["reductions"])),
            "MeasMaxRegret_pct": float(_nanmax(stats[label]["regrets_meas"])),
            "Speedup_gmean":    speedup_gmean[label],
        })

    # ── CSV sidecar ──
    csv_rows = []
    for r in rows:
        csv_rows.append({
            "Rule":                    r["Rule"],
            "Reduction (%)":           f"{r['Reduction_pct']:.1f}",
            "Meas. max regret (%)":    (
                f"{r['MeasMaxRegret_pct']:.1f}"
                if np.isfinite(r["MeasMaxRegret_pct"]) else "NA"
            ),
            "Speedup":                 (
                f"{r['Speedup_gmean']:.1f}x"
                if np.isfinite(r["Speedup_gmean"]) else "NA"
            ),
        })
    df_csv = pd.DataFrame(csv_rows)
    out_csv = output_dir / "T12_pruning_effectiveness.csv"
    df_csv.to_csv(out_csv, index=False, encoding="utf-8")

    # ── Read top_k_sp (Rule 3 parameter) for caption annotation ──
    from baselines import enumerate_configs_pruned as _ecp
    import inspect
    top_k_sp = inspect.signature(_ecp).parameters["top_k_sp"].default

    # ── LaTeX emitter (3 rows × 4 cols) ──
    out_tex = output_dir / "T12_pruning_effectiveness.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabularx}{\\textwidth}{Xrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Rule & Reduction (\\%) & Meas.\\ max regret (\\%) "
            "& Speedup \\\\\n"
        )
        f.write("\\midrule\n")
        for r in rows:
            red  = _fmt(r["Reduction_pct"])
            reg  = _fmt(r["MeasMaxRegret_pct"])
            sp   = (
                f"{r['Speedup_gmean']:.1f}$\\times$"
                if np.isfinite(r["Speedup_gmean"]) else "NA"
            )
            # Bold the final row (Rule 1+2+3)
            if r["Rule"] == "Rule 1+2+3":
                f.write(
                    f"\\textbf{{{r['Rule']}}} & \\textbf{{{red}}} & "
                    f"\\textbf{{{reg}}} & \\textbf{{{sp}}} \\\\\n"
                )
            else:
                f.write(f"{r['Rule']} & {red} & {reg} & {sp} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")

        # Caption-level notes (the paper caption lifts or paraphrases these).
        f.write("\n% Notes on T12:\n")
        f.write(
            "%   Reduction = per-workload mean of (1 - |Omega_rule| / |Omega|),\n"
            "%   where |Omega| is the exhaustive feasible configuration count.\n"
            "%   Taking the per-workload mean (not the pooled ratio) is the\n"
            "%   logically correct aggregation because pruning operates per\n"
            "%   workload; pooled-ratio aggregation is biased toward large\n"
            "%   workloads (Jensen's inequality).\n"
            "%\n"
            "%   Meas. max regret = worst-case 100*(EDP_rule/EDP_exh - 1) over\n"
            "%   workloads, both evaluated on XDNA2 measurements (F3+D2\n"
            "%   nearest-neighbor fallback by predicted EDP when a config is\n"
            "%   not directly measured). Max is taken in the \"worst for\n"
            "%   STAR-Map\" direction (largest positive).\n"
            "%\n"
            "%   Speedup = geomean across workloads of t_exh / t_rule from\n"
            "%   search_bench.py (warmup, min-of-N wall time with gc.disable).\n"
            "%\n"
            f"%   Rule 3 uses top-K SP filtering with K={top_k_sp}.\n"
        )

    return out_tex