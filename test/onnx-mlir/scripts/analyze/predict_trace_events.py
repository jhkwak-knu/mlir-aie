#!/usr/bin/env python3
"""Predict expected trace event counts from tc.json and compare with actual trace data.

Helps diagnose trace buffer overflow, dropped events, or PRES miscount issues
by comparing theoretical event counts against what parse_trace.py recorded.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow importing sibling modules when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_trace import (
    parse_trace_json,
    load_tc_json,
    detect_all_transfers,
    build_dispatch_template,
)


# ---------------------------------------------------------------------------
# Per-dispatch prediction
# ---------------------------------------------------------------------------

def predict_events_per_dispatch(tc: dict) -> dict:
    """Predict per-tile event counts for one dispatch from tc.json.

    Returns raw S2MM channel event counts (before PRES split) so we can
    compare directly against trace hardware counters.
    """
    lvl = tc["levels"][0]
    inner = lvl["tpOrder"][0]
    tp_map = {0: lvl["TPm"], 1: lvl["TPn"], 2: lvl["TPk"]}
    ipd = tp_map[inner]
    needs_pres = (lvl["TPk"] > 1) and (inner != 2)

    # Raw S2MM ch0/ch1 counts include PRES when sharing a channel
    if inner == 0:  # M-inner
        s2mm0_done = ipd + (ipd if needs_pres else 0)  # LHS + PRES
        s2mm1_done = 1   # RHS (reuse)
    elif inner == 1:  # N-inner
        s2mm0_done = 1   # LHS (reuse)
        s2mm1_done = ipd + (ipd if needs_pres else 0)  # RHS + PRES
    else:  # K-inner
        s2mm0_done = ipd  # LHS
        s2mm1_done = ipd  # RHS

    mm2s0_done = ipd if inner != 2 else 1

    return {
        "kernel": ipd,
        "s2mm0_done": s2mm0_done,
        "s2mm1_done": s2mm1_done,
        "mm2s0_done": mm2s0_done,
        "needs_pres": needs_pres,
        "inner_axis": inner,
        "ipd": ipd,
    }


def predict_total_events(tc: dict, n_measured: int = 10) -> dict:
    """Predict total trace buffer events across all measured iterations.

    host_steps dispatches per matmul, n_measured matmuls.
    """
    per_disp = predict_events_per_dispatch(tc)
    lvl = tc["levels"][0]
    tp_map = {0: lvl["TPm"], 1: lvl["TPn"], 2: lvl["TPk"]}
    host_steps = tp_map[lvl["tpOrder"][1]] * tp_map[lvl["tpOrder"][2]]
    total_dispatches = n_measured * host_steps

    return {
        "kernel": per_disp["kernel"] * total_dispatches,
        "s2mm0_done": per_disp["s2mm0_done"] * total_dispatches,
        "s2mm1_done": per_disp["s2mm1_done"] * total_dispatches,
        "mm2s0_done": per_disp["mm2s0_done"] * total_dispatches,
        "total_dispatches": total_dispatches,
        "host_steps": host_steps,
        "needs_pres": per_disp["needs_pres"],
        "inner_axis": per_disp["inner_axis"],
        "ipd": per_disp["ipd"],
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_predicted_vs_actual(tc: dict, tiles: list,
                                n_measured: int = 10) -> list[dict]:
    """Compare predicted vs actual raw event counts per tile.

    Only compares tiles that have at least one event (active tiles).
    The Pass only enables trace on column 0 tiles (compTilesPerCol),
    so multi-column configurations have many tiles with zero events.
    Including these empty tiles would falsely depress data_quality.

    Returns a list of dicts with tile name, actual/expected counts, and gaps.
    """
    pred = predict_total_events(tc, n_measured)

    report = []
    for td in tiles:
        # Skip tiles with no trace data (not instrumented)
        has_any = (len(td.kernels) > 0 or len(td.s2mm0_done) > 0
                   or len(td.s2mm1_done) > 0 or len(td.mm2s0_done) > 0)
        if not has_any:
            continue

        actual = {
            "kernel": len(td.kernels),
            "s2mm0_done": len(td.s2mm0_done),
            "s2mm1_done": len(td.s2mm1_done),
            "mm2s0_done": len(td.mm2s0_done),
        }
        expected = {
            "kernel": pred["kernel"],
            "s2mm0_done": pred["s2mm0_done"],
            "s2mm1_done": pred["s2mm1_done"],
            "mm2s0_done": pred["mm2s0_done"],
        }
        gaps = {k: expected[k] - actual[k] for k in actual}
        total_expected = sum(expected.values())
        total_actual = sum(actual.values())
        match_ratio = total_actual / total_expected if total_expected > 0 else 0.0

        report.append({
            "tile": td.tile_name,
            "actual": actual,
            "expected": expected,
            "gaps": gaps,
            "match_ratio": round(match_ratio, 4),
        })

    return report


def compute_data_quality(report: list[dict]) -> float:
    """Aggregate data quality score (0.0-1.0) across all tiles."""
    if not report:
        return 0.0
    return sum(r["match_ratio"] for r in report) / len(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict and compare trace event counts vs tc.json")
    parser.add_argument("--trace", required=True,
                        help="Path to trace.json")
    parser.add_argument("--tc", required=True,
                        help="Path to tc.json")
    parser.add_argument("--n-measured", type=int, default=10,
                        help="Number of measured iterations (default: 10)")
    parser.add_argument("--json", default=None,
                        help="Write comparison report as JSON")
    args = parser.parse_args()

    tc = load_tc_json(args.tc)
    tiles = parse_trace_json(args.trace)

    if not tiles:
        print("ERROR: No tiles found in trace.json", file=sys.stderr)
        sys.exit(1)

    # Run detection to populate kernels (needed for kernel count)
    detect_all_transfers(tiles, tc)

    # Predict and compare
    per_disp = predict_events_per_dispatch(tc)
    pred_total = predict_total_events(tc, args.n_measured)
    report = compare_predicted_vs_actual(tc, tiles, args.n_measured)
    quality = compute_data_quality(report)

    # Print summary
    axis_name = "MNK"[per_disp["inner_axis"]]
    print(f"\nEvent prediction ({axis_name}-inner, ipd={per_disp['ipd']}, "
          f"pres={'yes' if per_disp['needs_pres'] else 'no'}):")
    print(f"  Per dispatch: kern={per_disp['kernel']} "
          f"s2mm0={per_disp['s2mm0_done']} "
          f"s2mm1={per_disp['s2mm1_done']} "
          f"mm2s0={per_disp['mm2s0_done']}")
    print(f"  Total expected ({pred_total['total_dispatches']} dispatches): "
          f"kern={pred_total['kernel']} "
          f"s2mm0={pred_total['s2mm0_done']} "
          f"s2mm1={pred_total['s2mm1_done']} "
          f"mm2s0={pred_total['mm2s0_done']}")

    print(f"\nPer-tile comparison:")
    all_ok = True
    for r in report:
        gaps = r["gaps"]
        has_gap = any(v != 0 for v in gaps.values())
        status = "MISMATCH" if has_gap else "OK"
        if has_gap:
            all_ok = False
        gap_str = " ".join(f"{k}={v:+d}" for k, v in gaps.items() if v != 0)
        print(f"  {r['tile']}: {status} (match={r['match_ratio']:.1%})"
              + (f"  gaps: {gap_str}" if has_gap else ""))

    print(f"\nData quality: {quality:.1%}")
    missing = sum(max(0, g) for r in report for g in r["gaps"].values())
    print(f"Total missing events: {missing}")

    if args.json:
        out = {
            "per_dispatch": per_disp,
            "total_expected": pred_total,
            "tiles": report,
            "data_quality": quality,
            "missing_events": missing,
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nJSON report written to {args.json}", file=sys.stderr)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
