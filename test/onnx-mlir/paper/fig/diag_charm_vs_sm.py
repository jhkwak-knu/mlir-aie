#!/usr/bin/env python3
"""
diag_charm_vs_sm.py — Per-workload diagnostic: CHARM-CDSE vs STAR-Map.

Dumps for each workload: P, SPm×SPn, tpOrder_inner, measured_time_us,
measured_energy_uj, measured_edp — for GT, STAR-Map (framework), CHARM-CDSE,
and Timeloop. Used to answer per-size-class behavior questions.
"""

from pathlib import Path

from config import load_config, DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON
from data_loader import load_all
from baselines import compute_all_baselines

TP_MAP = {0: "M", 1: "N", 2: "K"}  # best-effort; we'll display int directly too


def fmt_res(r):
    if r is None:
        return "n/a"
    P = r.P
    sp = f"{r.SPm}x{r.SPn}"
    tp = f"{r.TPm}x{r.TPk}x{r.TPn}"
    tpk = r.tpOrder_inner
    t = r.measured_time_us
    e = r.measured_energy_uj
    edp = r.measured_edp
    fb = " [FB]" if getattr(r, "is_fallback", False) else ""
    return (f"P={P:2d} SP={sp:<5s} TP={tp:<16s} tpK={tpk} "
            f"t={t:10.2f}us e={e:10.2f}uJ edp={edp:13.1f}{fb}")


def mac_count(wl):
    M, K, N = wl
    return M * K * N


def size_class(wl):
    macs = mac_count(wl)
    if macs < 10_000_000:
        return "S"
    if macs < 300_000_000:
        return "M"
    return "L"


def main():
    cfg = load_config(DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON)
    df, groups = load_all(cfg)
    baselines = compute_all_baselines(groups, cfg)

    rows = []
    for wl in cfg.WORKLOADS:
        bl = baselines.get(wl)
        if bl is None:
            continue
        rows.append((wl, bl))

    # Sort by MACs (ascending)
    rows.sort(key=lambda x: mac_count(x[0]))

    print("=" * 120)
    print(f"{'size':4s} {'Workload':20s} {'MACs':>12s}  | method    | config + measured values")
    print("=" * 120)

    for wl, bl in rows:
        cls = size_class(wl)
        M, K, N = wl
        wl_str = f"{M}x{K}x{N}"
        macs = mac_count(wl)
        header = f"{cls:4s} {wl_str:20s} {macs:12d}"
        print(f"{header}  | GT        | {fmt_res(bl.gt)}")
        print(f"{'':38s}  | STAR-Map  | {fmt_res(bl.framework)}")
        print(f"{'':38s}  | CHARM     | {fmt_res(bl.charm_spk1)}")
        print(f"{'':38s}  | Timeloop  | {fmt_res(bl.timeloop)}")
        print("-" * 120)

    # ── T vs E comparison (Reduction % of STAR-Map against CHARM) ──────────
    print("\n" + "=" * 120)
    print("CHARM vs STAR-Map (measured): delta_t%, delta_e%, delta_edp%")
    print("(positive = STAR-Map better than CHARM)")
    print("=" * 120)
    print(f"{'size':4s} {'Workload':20s} {'SM P/SP':12s} {'CH P/SP':12s} "
          f"{'t_SM':>10s} {'t_CH':>10s} {'dt%':>7s} "
          f"{'e_SM':>10s} {'e_CH':>10s} {'de%':>7s} "
          f"{'edp_SM':>12s} {'edp_CH':>12s} {'dedp%':>7s}")
    for wl, bl in rows:
        if bl.framework is None or bl.charm_spk1 is None:
            continue
        sm = bl.framework
        ch = bl.charm_spk1
        cls = size_class(wl)
        M, K, N = wl
        wl_str = f"{M}x{K}x{N}"
        sm_cfg = f"{sm.P}/{sm.SPm}x{sm.SPn}"
        ch_cfg = f"{ch.P}/{ch.SPm}x{ch.SPn}"
        dt = (1 - sm.measured_time_us / ch.measured_time_us) * 100
        de = (1 - sm.measured_energy_uj / ch.measured_energy_uj) * 100
        dedp = (1 - sm.measured_edp / ch.measured_edp) * 100
        print(f"{cls:4s} {wl_str:20s} {sm_cfg:12s} {ch_cfg:12s} "
              f"{sm.measured_time_us:10.2f} {ch.measured_time_us:10.2f} {dt:7.1f} "
              f"{sm.measured_energy_uj:10.2f} {ch.measured_energy_uj:10.2f} {de:7.1f} "
              f"{sm.measured_edp:12.1f} {ch.measured_edp:12.1f} {dedp:7.1f}")


if __name__ == "__main__":
    main()
