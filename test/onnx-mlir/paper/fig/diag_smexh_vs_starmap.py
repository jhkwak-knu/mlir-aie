#!/usr/bin/env python3
"""
diag_smexh_vs_starmap.py — Diagnose SM-exh > STAR-Map anomalies in T11.

For each workload, prints:
  - SM-exh's predicted argmin (full space) : config, predicted EDP
  - STAR-Map's predicted argmin (pruned)   : config, predicted EDP
  - Is SM-exh argmin in pruned space?
  - Reported measured EDP for each (including fallback info)

Expected invariant (predicted EDP): SM-exh <= STAR-Map (always).
Observed anomaly (measured EDP):    SM-exh > STAR-Map in some rows.

Focus workloads (from T11):
  512x64x512, 32x768x768, 128x3072x768   (SM-exh > STAR-Map strictly)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import load_config, DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON
from data_loader import load_all
from cost_model import edp, MappingConfig, TPORDER_M_INNER, TPORDER_N_INNER, TPORDER_K_INNER
from baselines import enumerate_configs, enumerate_configs_pruned, find_framework_optimal


def _dinner_name(v):
    return {TPORDER_M_INNER: "M", TPORDER_N_INNER: "N", TPORDER_K_INNER: "K"}.get(v, f"?{v}")


def _config_key(mc):
    """Hashable key identifying a config."""
    return (mc.P, mc.SPm, mc.SPn, mc.TPm, mc.TPk, mc.TPn, mc.tpOrder_inner)


def _lookup_measured(wl_df, mc):
    """Return (in_meas, measured_edp) for the exact config, or (False, nan)."""
    if wl_df is None or wl_df.empty:
        return False, np.nan
    m = wl_df[
        (wl_df["SPm"] == mc.SPm) &
        (wl_df["SPn"] == mc.SPn) &
        (wl_df["TPm"] == mc.TPm) &
        (wl_df["TPk"] == mc.TPk) &
        (wl_df["TPn"] == mc.TPn) &
        (wl_df["tpOrder_inner"] == mc.tpOrder_inner)
    ]
    if m.empty:
        return False, np.nan
    row = m.iloc[0]
    if not bool(row.get("energy_valid", True)):
        return True, np.nan
    return True, float(row["edp_measured"])


def diag_one(wl, groups, cfg, focus=False):
    M, K, N = wl
    wl_df = groups.get(wl)

    # True argmins (predicted) — without fallback
    full = enumerate_configs(M, K, N, cfg)
    pruned = enumerate_configs_pruned(M, K, N, cfg)

    if not full or not pruned:
        return

    c_full = min(full, key=lambda mc: edp(mc, cfg))
    c_prun = min(pruned, key=lambda mc: edp(mc, cfg))

    pred_edp_full = edp(c_full, cfg)
    pred_edp_prun = edp(c_prun, cfg)

    in_full_meas, meas_edp_full = _lookup_measured(wl_df, c_full)
    in_prun_meas, meas_edp_prun = _lookup_measured(wl_df, c_prun)

    # Is SM-exh's argmin in pruned space?
    pruned_keys = {_config_key(mc) for mc in pruned}
    full_in_pruned = _config_key(c_full) in pruned_keys

    # What baselines.py ACTUALLY reports (after fallback)
    rep_full = find_framework_optimal(M, K, N, cfg, wl_df, pruned=False)
    rep_prun = find_framework_optimal(M, K, N, cfg, wl_df, pruned=True)

    def _fmt_cfg(mc):
        return f"{mc.P}/{mc.SPm}x{mc.SPn} TP=({mc.TPm},{mc.TPk},{mc.TPn}) d={_dinner_name(mc.tpOrder_inner)}"

    def _fmt_rep(r):
        if r is None:
            return "None"
        cfg_s = f"{r.P}/{r.SPm}x{r.SPn} TP=({r.TPm},{r.TPk},{r.TPn}) d={_dinner_name(r.tpOrder_inner)}"
        meas_s = f"meas={r.measured_edp:.0f}" if not np.isnan(r.measured_edp) else "meas=NaN"
        pred_s = f"pred={r.pred_edp:.0f}"
        fb = " [FB]" if r.is_fallback else ""
        return f"{cfg_s} | {pred_s} {meas_s}{fb}"

    print(f"\n═══ {M}x{K}x{N} ═══")
    print(f"  TRUE argmins (no fallback):")
    print(f"    SM-exh argmin : {_fmt_cfg(c_full)}  pred={pred_edp_full:.0f}  in_meas={in_full_meas}  meas_edp={meas_edp_full if not np.isnan(meas_edp_full) else 'NaN'}")
    print(f"    STAR-Map argmin: {_fmt_cfg(c_prun)}  pred={pred_edp_prun:.0f}  in_meas={in_prun_meas}  meas_edp={meas_edp_prun if not np.isnan(meas_edp_prun) else 'NaN'}")
    print(f"    Predicted invariant: SM-exh ({pred_edp_full:.0f}) <= STAR-Map ({pred_edp_prun:.0f}) ? "
          f"{'OK' if pred_edp_full <= pred_edp_prun + 1e-6 else 'VIOLATED'}")
    print(f"    SM-exh argmin in pruned space? {full_in_pruned}")
    print(f"  REPORTED (after fallback, what T11 actually shows):")
    print(f"    SM-exh  : {_fmt_rep(rep_full)}")
    print(f"    STAR-Map: {_fmt_rep(rep_prun)}")

    # Compute the ratios shown in T11 (vs GT)
    # GT: best measured EDP
    if wl_df is not None and not wl_df.empty:
        valid = wl_df[wl_df["energy_valid"]]
        if not valid.empty:
            gt_edp = float(valid["edp_measured"].min())
            r_full = rep_full.measured_edp / gt_edp if rep_full and not np.isnan(rep_full.measured_edp) else float("nan")
            r_prun = rep_prun.measured_edp / gt_edp if rep_prun and not np.isnan(rep_prun.measured_edp) else float("nan")
            anomaly = (r_full > r_prun + 1e-6) if (not np.isnan(r_full) and not np.isnan(r_prun)) else False
            print(f"  T11 ratios vs GT ({gt_edp:.0f}): SM-exh={r_full:.3f}x  STAR-Map={r_prun:.3f}x  {'[ANOMALY: SM-exh>STAR-Map]' if anomaly else ''}")


def main():
    cfg = load_config(DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON)
    _, groups = load_all(cfg)

    # Focus on the three anomaly workloads from T11
    focus = [
        (512, 64, 512),
        (32, 768, 768),
        (128, 3072, 768),
    ]

    print("=" * 78)
    print("FOCUS: T11 anomaly workloads (SM-exh ratio > STAR-Map ratio)")
    print("=" * 78)
    for wl in focus:
        diag_one(wl, groups, cfg, focus=True)

    print("\n" + "=" * 78)
    print("SANITY: all workloads — report predicted-invariant check + T11 ratios")
    print("=" * 78)
    for wl in cfg.WORKLOADS:
        diag_one(wl, groups, cfg)


if __name__ == "__main__":
    main()
