"""Dump F7's per-workload reduction values to CSV for verification."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    load_config,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_RESULT_CSV,
    DEFAULT_TC_LIST_JSON,
)
from data_loader import load_all
from baselines import compute_all_baselines

cfg = load_config(
    DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON
)
df, groups = load_all(cfg)
baselines = compute_all_baselines(groups, cfg)


def _meas_or_pred(res, fm, fp):
    if res is None:
        return float("nan")
    v = getattr(res, fm)
    if getattr(res, "in_measurements", False) and not np.isnan(v):
        return v
    return getattr(res, fp)


def panel_rows(get_baseline):
    out = []
    for wl in cfg.WORKLOADS:
        bl = baselines.get(wl)
        if not bl or not bl.framework:
            continue
        base = get_baseline(bl)
        if base is None:
            continue
        fw = bl.framework
        ft = _meas_or_pred(fw, "measured_time_us", "pred_time")
        fe = _meas_or_pred(fw, "measured_energy_uj", "pred_energy")
        fd = _meas_or_pred(fw, "measured_edp", "pred_edp")
        bt = _meas_or_pred(base, "measured_time_us", "pred_time")
        be = _meas_or_pred(base, "measured_energy_uj", "pred_energy")
        bd = _meas_or_pred(base, "measured_edp", "pred_edp")
        if any(np.isnan(v) or v <= 0 for v in (ft, fe, fd, bt, be, bd)):
            continue
        out.append({
            "wl": f"{wl[0]}x{wl[1]}x{wl[2]}",
            "p_sm": fw.P,
            "p_base": base.P,
            "t_red_pct": (1 - ft / bt) * 100,
            "e_red_pct": (1 - fe / be) * 100,
            "edp_red_pct": (1 - fd / bd) * 100,
        })
    return out


for tag, fn in [("CHARM", lambda b: b.charm_spk1),
                ("Timeloop", lambda b: b.timeloop)]:
    rows = panel_rows(fn)
    fout = Path(__file__).parent / f"output/_F7_{tag}_dump.csv"
    with open(fout, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"=== F7 vs {tag} ===")
    for r in rows:
        print(f"  {r['wl']:>20s}  P_b={r['p_base']:>2d}  P_sm={r['p_sm']:>2d}  "
              f"T={r['t_red_pct']:6.1f}%  E={r['e_red_pct']:6.1f}%  "
              f"EDP={r['edp_red_pct']:6.1f}%")
