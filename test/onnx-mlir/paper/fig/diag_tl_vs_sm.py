#!/usr/bin/env python3
"""
diag_tl_vs_sm.py — Per-workload measured comparison: STAR-Map vs Timeloop.
"""

from config import load_config, DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON
from data_loader import load_all
from baselines import compute_all_baselines


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
    rows.sort(key=lambda x: mac_count(x[0]))

    print("=" * 140)
    print("Timeloop vs STAR-Map (measured): delta_t%, delta_e%, delta_edp%")
    print("(positive = STAR-Map better than Timeloop)")
    print("=" * 140)
    print(f"{'size':4s} {'Workload':20s} {'SM P/SP':12s} {'TL P/SP':12s} "
          f"{'t_SM':>10s} {'t_TL':>10s} {'dt%':>7s} "
          f"{'e_SM':>10s} {'e_TL':>10s} {'de%':>7s} "
          f"{'edp_SM':>13s} {'edp_TL':>13s} {'dedp%':>7s}")
    for wl, bl in rows:
        if bl.framework is None or bl.timeloop is None:
            continue
        sm = bl.framework
        tl = bl.timeloop
        cls = size_class(wl)
        M, K, N = wl
        wl_str = f"{M}x{K}x{N}"
        sm_cfg = f"{sm.P}/{sm.SPm}x{sm.SPn}"
        tl_cfg = f"{tl.P}/{tl.SPm}x{tl.SPn}"
        dt = (1 - sm.measured_time_us / tl.measured_time_us) * 100
        de = (1 - sm.measured_energy_uj / tl.measured_energy_uj) * 100
        dedp = (1 - sm.measured_edp / tl.measured_edp) * 100
        print(f"{cls:4s} {wl_str:20s} {sm_cfg:12s} {tl_cfg:12s} "
              f"{sm.measured_time_us:10.2f} {tl.measured_time_us:10.2f} {dt:7.1f} "
              f"{sm.measured_energy_uj:10.2f} {tl.measured_energy_uj:10.2f} {de:7.1f} "
              f"{sm.measured_edp:13.1f} {tl.measured_edp:13.1f} {dedp:7.1f}")


if __name__ == "__main__":
    main()
