#!/usr/bin/env python3
"""
diag_charm_original.py — Expose CHARM-CDSE's ORIGINAL cost-model pick
(before measurement-fallback substitution) vs the fallback config reported
in BaselineResult. Reveals whether the reported SP shape (e.g., 32x1 for
1024^3) is CHARM's genuine choice or a nearest-EDP substitute.
"""

from pathlib import Path

from config import load_config, DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON
from data_loader import load_all
from baselines import enumerate_configs, compute_all_baselines
from baselines_charm import charm_cycle_cost
from cost_model import t_total, e_total, edp


def main():
    cfg = load_config(DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON)
    df, groups = load_all(cfg)
    baselines = compute_all_baselines(groups, cfg)

    print(f"{'Workload':20s} | "
          f"{'CHARM orig pick (P/SP/TP/tpK)':42s} | pred_cyc   | "
          f"{'Reported (post-FB)':32s} | in_meas")
    print("=" * 140)

    for wl in cfg.WORKLOADS:
        M, K, N = wl
        wl_str = f"{M}x{K}x{N}"
        bl = baselines.get(wl)
        if bl is None or bl.charm_spk1 is None:
            continue

        all_configs = enumerate_configs(M, K, N, cfg)
        # Apply same key as find_charm_cdse
        best_orig = min(
            all_configs,
            key=lambda mc: (charm_cycle_cost(mc, cfg, "sum"), mc.P, mc.TP_total)
        )
        cyc = charm_cycle_cost(best_orig, cfg, "sum")
        orig_str = (f"P={best_orig.P:2d} SP={best_orig.SPm}x{best_orig.SPn} "
                    f"TP={best_orig.TPm}x{best_orig.TPk}x{best_orig.TPn} "
                    f"tpK={best_orig.tpOrder_inner}")
        rep = bl.charm_spk1
        rep_str = (f"P={rep.P:2d} SP={rep.SPm}x{rep.SPn} "
                   f"TP={rep.TPm}x{rep.TPk}x{rep.TPn} tpK={rep.tpOrder_inner}")
        fb = "[FB]" if rep.is_fallback else "    "
        print(f"{wl_str:20s} | {orig_str:42s} | {cyc:10.1f} | {rep_str:32s} | {fb}")

    # ── Look for check: does STAR-Map's pick have lower CHARM-cycle than CHARM's pick? ──
    print("\n\nCHARM-cycle of each method's pick (lower = CHARM's objective prefers it):")
    print(f"{'Workload':20s} | {'STAR-Map':12s} | {'CHARM orig':12s} | SM/CH ratio")
    print("-" * 80)
    for wl in cfg.WORKLOADS:
        M, K, N = wl
        wl_str = f"{M}x{K}x{N}"
        bl = baselines.get(wl)
        if bl is None or bl.framework is None or bl.charm_spk1 is None:
            continue
        sm = bl.framework
        all_configs = enumerate_configs(M, K, N, cfg)
        # find SM's mc in enumerated list (nearest match)
        sm_mc = next(
            (mc for mc in all_configs
             if mc.SPm == sm.SPm and mc.SPn == sm.SPn
             and mc.TPm == sm.TPm and mc.TPk == sm.TPk and mc.TPn == sm.TPn
             and mc.tpOrder_inner == sm.tpOrder_inner),
            None,
        )
        if sm_mc is None:
            print(f"{wl_str:20s} | (SM config not in enumerate_configs)")
            continue
        sm_cyc = charm_cycle_cost(sm_mc, cfg, "sum")
        ch_orig = min(
            all_configs,
            key=lambda mc: (charm_cycle_cost(mc, cfg, "sum"), mc.P, mc.TP_total)
        )
        ch_cyc = charm_cycle_cost(ch_orig, cfg, "sum")
        print(f"{wl_str:20s} | {sm_cyc:12.1f} | {ch_cyc:12.1f} | {sm_cyc/ch_cyc:6.2f}x")


if __name__ == "__main__":
    main()
