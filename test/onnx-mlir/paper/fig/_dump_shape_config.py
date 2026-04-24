"""Dump per-workload GT/STAR-Map configuration (P, SPm, SPn, TP, d_inner, EDP)
for verification of §4.4.3 shape-dependency claims."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    load_config, DEFAULT_CALIBRATION_PATH,
    DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON,
)
from data_loader import load_all
from baselines import compute_all_baselines

cfg = load_config(DEFAULT_CALIBRATION_PATH, DEFAULT_RESULT_CSV, DEFAULT_TC_LIST_JSON)
df, groups = load_all(cfg)
baselines = compute_all_baselines(groups, cfg)


def fmt(r):
    if r is None:
        return "              n/a"
    return (f"P={r.P:>2d} SP={r.SPm}×{r.SPn} "
            f"TP=({r.TPm},{r.TPk},{r.TPn}) d={r.tpOrder_inner}")


print(f"{'Workload':<22s} {'MACs':>7s}  | "
      f"{'GT':<38s}  {'STAR-Map':<38s}  match?")
print("-" * 120)

for wl in cfg.WORKLOADS:
    bl = baselines.get(wl)
    if bl is None:
        continue
    macs = wl[0] * wl[1] * wl[2]
    macs_str = f"{macs/1e6:.1f}M" if macs < 1e9 else f"{macs/1e9:.2f}G"
    label = f"{wl[0]}x{wl[1]}x{wl[2]}"
    gt = bl.gt
    sm = bl.framework

    match_P  = "✓" if (gt and sm and gt.P == sm.P) else "✗"
    match_SP = "✓" if (gt and sm and
                      gt.SPm == sm.SPm and gt.SPn == sm.SPn) else "✗"
    match_all = "✓" if (gt and sm and
                       gt.P == sm.P and gt.SPm == sm.SPm and
                       gt.SPn == sm.SPn and gt.TPm == sm.TPm and
                       gt.TPk == sm.TPk and gt.TPn == sm.TPn and
                       gt.tpOrder_inner == sm.tpOrder_inner) else "✗"

    print(f"{label:<22s} {macs_str:>7s}  | {fmt(gt):<38s}  {fmt(sm):<38s}  "
          f"P={match_P} SP={match_SP} all={match_all}")
