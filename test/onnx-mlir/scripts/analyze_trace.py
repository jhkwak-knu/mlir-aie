#!/usr/bin/env python3
"""Analyze NPU trace data to extract per-iteration timing breakdown.

Reads parse_trace.py output (trace.json) and tiling config (tc.json) to produce:
  1. Per-iteration breakdown: kernel, pre-stall, post-stall, overhead
  2. Dispatch-averaged 1-dispatch profile
  3. Cross-tile parallelism analysis
  4. Data transfer serial/parallel inference
"""

import argparse
import json
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XDNA2_DEFAULT_CLOCK_MHZ = 1300

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StallInfo:
    """One individual lock stall event."""
    begin: int
    end: int
    duration: int
    label: str = ""       # e.g. "LHS(A) acq", "RES(C) rel"

@dataclass
class IterInfo:
    """Timing breakdown for one kernel iteration on one tile."""
    evt0_ts: int          # kernel start timestamp (cycles)
    evt1_ts: int          # kernel end timestamp (cycles)
    t_kernel: int = 0     # EVT0 → EVT1
    t_pre_stall: int = 0  # lock stall before kernel (sum)
    t_post_stall: int = 0 # lock stall after kernel (sum)
    t_gap_pre: int = 0    # total gap before kernel (prev_evt1 → evt0)
    t_gap_post: int = 0   # total gap after kernel (evt1 → next_evt0)
    pre_stalls: list = field(default_factory=list)   # [StallInfo, ...] before kernel
    post_stalls: list = field(default_factory=list)  # [StallInfo, ...] after kernel

@dataclass
class TileTrace:
    """All parsed events for one tile."""
    pid: int
    tile_name: str
    evt0_ts: list = field(default_factory=list)  # kernel start timestamps
    evt1_ts: list = field(default_factory=list)  # kernel end timestamps
    stall_pairs: list = field(default_factory=list)  # [(begin, end), ...]
    iterations: list = field(default_factory=list)  # [IterInfo, ...]

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_trace_json(path: str) -> dict[int, TileTrace]:
    """Parse trace.json into per-tile event lists."""
    with open(path) as f:
        events = json.load(f)

    tiles: dict[int, TileTrace] = {}

    # First pass: collect metadata (tile names)
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            pid = ev["pid"]
            tile_name = ev["args"].get("name", f"tile_{pid}")
            tiles[pid] = TileTrace(pid=pid, tile_name=tile_name)

    # Second pass: collect events
    # Track pending stall begins per (pid)
    stall_begin: dict[int, int] = {}

    for ev in events:
        if ev.get("ph") == "M":
            continue
        pid = ev.get("pid")
        if pid is None or pid not in tiles:
            continue
        tile = tiles[pid]
        name = ev.get("name", "")
        ph = ev.get("ph", "")
        ts = ev.get("ts", 0)

        if name == "INSTR_EVENT_0" and ph == "B":
            tile.evt0_ts.append(ts)
        elif name == "INSTR_EVENT_1" and ph == "B":
            tile.evt1_ts.append(ts)
        elif name == "LOCK_STALL":
            if ph == "B":
                stall_begin[pid] = ts
            elif ph == "E" and pid in stall_begin:
                tile.stall_pairs.append((stall_begin[pid], ts))
                del stall_begin[pid]

    return tiles


def load_tc_json(path: str) -> dict:
    """Load tiling configuration."""
    with open(path) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def collect_stalls_in_range(stall_pairs: list[tuple[int,int]],
                            lo: int, hi: int) -> list[StallInfo]:
    """Collect individual lock stall events within [lo, hi)."""
    result = []
    for b, e in stall_pairs:
        cb = max(b, lo)
        ce = min(e, hi)
        if cb < ce:
            result.append(StallInfo(begin=cb, end=ce, duration=ce - cb))
    return result


def sum_stalls_in_range(stall_pairs: list[tuple[int,int]], lo: int, hi: int) -> int:
    """Sum lock stall cycles within [lo, hi)."""
    return sum(s.duration for s in collect_stalls_in_range(stall_pairs, lo, hi))


def get_stall_labels(inner_axis: int) -> dict:
    """Return expected stall label sequences based on tpOrder[0].

    CoreOp lock order (from emitCoreOps / selectAxisBuffers):
      outer:  acquire reuse → [inner loop] → release reuse
      inner:  acquire inner1 → acquire inner2 → KERNEL
              → release inner2 → release inner1

    Mapping per axis:
      M-inner: reuse=RHS(B), inner1=LHS(A), inner2=RES(C)
      N-inner: reuse=LHS(A), inner1=RHS(B), inner2=RES(C)
      K-inner: reuse=RES(C), inner1=LHS(A), inner2=RHS(B)
    """
    axis_map = {
        0: ('RHS(B)', 'LHS(A)', 'RES(C)'),  # M-inner
        1: ('LHS(A)', 'RHS(B)', 'RES(C)'),  # N-inner
        2: ('RES(C)', 'LHS(A)', 'RHS(B)'),  # K-inner
    }
    reuse, inner1, inner2 = axis_map[inner_axis]
    return {
        # First iteration of dispatch: previous dispatch's releases + new acquires.
        # (The outer CoreOp loop runs continuously across dispatches, so iter 0
        #  pre-kernel always includes the prior dispatch's post-releases.)
        'first': [
            f'{inner2} rel',
            f'{inner1} rel',
            f'{reuse} rel',
            f'{reuse} acq',
            f'{inner1} acq',
            f'{inner2} acq',
        ],
        # Mid iteration transition: releases from prev + acquires for next
        'mid': [
            f'{inner2} rel',
            f'{inner1} rel',
            f'{inner1} acq',
            f'{inner2} acq',
        ],
        # Last iter → next dispatch boundary: + outer loop release/acquire
        'last': [
            f'{inner2} rel',
            f'{inner1} rel',
            f'{reuse} rel',
            f'{reuse} acq',
            f'{inner1} acq',
            f'{inner2} acq',
        ],
    }


def label_stalls(stalls: list[StallInfo], labels: list[str]) -> list[StallInfo]:
    """Assign labels to stall list based on expected sequence."""
    for i, s in enumerate(stalls):
        if i < len(labels):
            s.label = labels[i]
        else:
            s.label = f'unknown[{i}]'
    return stalls


def build_iterations(tile: TileTrace, iters_per_dispatch: int,
                     stall_labels: dict) -> list[IterInfo]:
    """Build per-iteration IterInfo list with individual labeled stalls."""
    n = min(len(tile.evt0_ts), len(tile.evt1_ts))
    iters = []
    for i in range(n):
        it = IterInfo(
            evt0_ts=tile.evt0_ts[i],
            evt1_ts=tile.evt1_ts[i],
            t_kernel=tile.evt1_ts[i] - tile.evt0_ts[i],
        )

        # Pre-kernel gap and stalls
        prev_end = tile.evt1_ts[i - 1] if i > 0 else 0
        it.t_gap_pre = it.evt0_ts - prev_end
        it.pre_stalls = collect_stalls_in_range(
            tile.stall_pairs, prev_end, it.evt0_ts
        )
        it.t_pre_stall = sum(s.duration for s in it.pre_stalls)

        # Determine stall label pattern based on position in dispatch
        pos_in_dispatch = i % iters_per_dispatch
        if pos_in_dispatch == 0:
            label_stalls(it.pre_stalls, stall_labels['first'])
        else:
            label_stalls(it.pre_stalls, stall_labels['mid'])

        # Post-kernel gap and stalls
        if i + 1 < n:
            next_start = tile.evt0_ts[i + 1]
            it.t_gap_post = next_start - it.evt1_ts
            it.post_stalls = collect_stalls_in_range(
                tile.stall_pairs, it.evt1_ts, next_start
            )
            it.t_post_stall = sum(s.duration for s in it.post_stalls)

            # Last iter of dispatch: post-stalls are part of dispatch boundary
            next_pos = (i + 1) % iters_per_dispatch
            if next_pos == 0:
                label_stalls(it.post_stalls, stall_labels['last'])
            else:
                label_stalls(it.post_stalls, stall_labels['mid'])

        iters.append(it)
    return iters


def detect_dispatch_boundaries(iters: list[IterInfo], iters_per_dispatch: int,
                                n_dispatches: int) -> list[int]:
    """Return iteration indices where each dispatch starts.

    Primary method: use known iters_per_dispatch to split evenly.
    Validation: check that boundary gaps are larger than within-dispatch gaps.
    """
    total = len(iters)
    expected = iters_per_dispatch * n_dispatches
    if total != expected:
        print(f"  WARNING: expected {expected} iterations, got {total}",
              file=sys.stderr)

    boundaries = list(range(0, total, iters_per_dispatch))

    # Validate: boundary gaps should be larger than within-dispatch gaps
    within_gaps = []
    boundary_gaps = []
    for i in range(1, len(iters)):
        gap = iters[i].t_gap_pre
        if i % iters_per_dispatch == 0:
            boundary_gaps.append(gap)
        else:
            within_gaps.append(gap)

    if within_gaps and boundary_gaps:
        avg_within = sum(within_gaps) / len(within_gaps)
        avg_boundary = sum(boundary_gaps) / len(boundary_gaps)
        if avg_boundary <= avg_within:
            print(f"  WARNING: dispatch boundary gaps ({avg_boundary:.0f}) "
                  f"<= within-dispatch gaps ({avg_within:.0f})", file=sys.stderr)

    return boundaries

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_cycles(c: float) -> str:
    """Format cycle count with comma separator."""
    return f"{c:,.0f}"


def fmt_us(cycles: float, clock_mhz: float) -> str:
    """Convert cycles to microseconds string."""
    return f"{cycles / clock_mhz:,.1f}"


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def report_config(tc: dict, n_dispatches: int, iters_per_dispatch: int,
                  clock_mhz: float):
    """Print configuration summary."""
    print_header("Configuration")
    lvl = tc["levels"][0]
    print(f"  Matrix:    {tc['M']}x{tc['K']}x{tc['N']} {tc['elemType']}")
    print(f"  Spatial:   SPm={lvl['SPm']} SPn={lvl['SPn']} "
          f"({tc['numCores']} tiles)")
    print(f"  Temporal:  TPm={lvl['TPm']} TPk={lvl['TPk']} TPn={lvl['TPn']}")
    print(f"  Tile size: TM={lvl['TM']} TK={lvl['TK']} TN={lvl['TN']}")
    print(f"  tpOrder:   {lvl['tpOrder']} "
          f"(inner={'MNK'[lvl['tpOrder'][0]]})")
    print(f"  Dispatches: {n_dispatches}  "
          f"Iters/dispatch: {iters_per_dispatch}")
    print(f"  Clock:     {clock_mhz} MHz")


def report_per_iteration(tiles: dict[int, TileTrace], iters_per_dispatch: int,
                          n_dispatches: int, clock_mhz: float):
    """Print averaged per-iteration breakdown."""
    print_header("Average 1-Dispatch Profile (per tile)")

    for pid in sorted(tiles.keys()):
        tile = tiles[pid]
        iters = tile.iterations
        n_total = iters_per_dispatch * n_dispatches

        if len(iters) < n_total:
            print(f"\n  [{tile.tile_name}] insufficient iterations "
                  f"({len(iters)}/{n_total})")
            continue

        print(f"\n  [{tile.tile_name}]")
        print(f"  {'Iter':>4}  {'Pre-Stall':>12}  {'Kernel':>12}  "
              f"{'Post-Stall':>12}  {'Overhead':>10}  {'Total':>12}  "
              f"{'Total(us)':>10}")
        print(f"  {'----':>4}  {'----------':>12}  {'--------':>12}  "
              f"{'----------':>12}  {'--------':>10}  {'-------':>12}  "
              f"{'--------':>10}")

        for pos in range(iters_per_dispatch):
            # Collect this position across all dispatches
            pre_stalls, kernels, post_stalls, overheads, totals = (
                [], [], [], [], []
            )
            for d in range(n_dispatches):
                idx = d * iters_per_dispatch + pos
                it = iters[idx]
                pre = it.t_pre_stall
                ker = it.t_kernel
                # Post-stall: use stall between this EVT1 and next EVT0
                # For the last iter in a dispatch, use the stored value
                post = it.t_post_stall
                gap_total = it.t_gap_pre + it.t_kernel
                overhead = it.t_gap_pre - it.t_pre_stall
                total = it.t_gap_pre + it.t_kernel

                pre_stalls.append(pre)
                kernels.append(ker)
                post_stalls.append(post)
                overheads.append(overhead)
                totals.append(total)

            avg_pre = sum(pre_stalls) / len(pre_stalls)
            avg_ker = sum(kernels) / len(kernels)
            avg_post = sum(post_stalls) / len(post_stalls)
            avg_oh = sum(overheads) / len(overheads)
            avg_tot = sum(totals) / len(totals)

            print(f"  {pos:>4}  {fmt_cycles(avg_pre):>12}  "
                  f"{fmt_cycles(avg_ker):>12}  {fmt_cycles(avg_post):>12}  "
                  f"{fmt_cycles(avg_oh):>10}  {fmt_cycles(avg_tot):>12}  "
                  f"{fmt_us(avg_tot, clock_mhz):>10}")

        # Summary for iter 1-7 (steady state, excluding first)
        ss_pre, ss_ker, ss_oh, ss_tot = [], [], [], []
        for pos in range(1, iters_per_dispatch):
            for d in range(n_dispatches):
                idx = d * iters_per_dispatch + pos
                it = iters[idx]
                ss_pre.append(it.t_pre_stall)
                ss_ker.append(it.t_kernel)
                ss_oh.append(it.t_gap_pre - it.t_pre_stall)
                ss_tot.append(it.t_gap_pre + it.t_kernel)

        avg_pre = sum(ss_pre) / len(ss_pre)
        avg_ker = sum(ss_ker) / len(ss_ker)
        avg_oh = sum(ss_oh) / len(ss_oh)
        avg_tot = sum(ss_tot) / len(ss_tot)

        print(f"  {'':>4}  {'----------':>12}  {'--------':>12}  "
              f"{'':>12}  {'--------':>10}  {'-------':>12}  {'--------':>10}")
        print(f"  {'1-7':>4}  {fmt_cycles(avg_pre):>12}  "
              f"{fmt_cycles(avg_ker):>12}  {'':>12}  "
              f"{fmt_cycles(avg_oh):>10}  {fmt_cycles(avg_tot):>12}  "
              f"{fmt_us(avg_tot, clock_mhz):>10}")
        pct_ker = avg_ker / avg_tot * 100 if avg_tot else 0
        pct_pre = avg_pre / avg_tot * 100 if avg_tot else 0
        print(f"  {'':>4}  Kernel: {pct_ker:.1f}%  Stall: {pct_pre:.1f}%")


def report_stall_detail(tiles: dict[int, TileTrace], iters_per_dispatch: int,
                         n_dispatches: int, clock_mhz: float):
    """Print individual stall breakdown with buffer labels."""
    print_header("Stall Detail (per-tile, dispatch-averaged)")

    # Use first tile for detailed output; all tiles have same lock sequence
    ref_pid = sorted(tiles.keys())[0]
    ref_tile = tiles[ref_pid]
    ref_iters = ref_tile.iterations
    n_total = iters_per_dispatch * n_dispatches

    if len(ref_iters) < n_total:
        print("  Insufficient iterations.")
        return

    print(f"\n  Tile: {ref_tile.tile_name}")

    # Show 3 transition types: first iter pre, mid transition, dispatch boundary
    sections = [
        ("Iter 0 pre-kernel (dispatch setup + first data load)",
         'pre', 0),
        ("Iter 0->1 transition (release prev + acquire next)",
         'pre', 1),
        ("Iter 6->7 transition (same as mid, last inner iter)",
         'pre', iters_per_dispatch - 1),
        ("Iter 7 post-kernel (dispatch boundary: inner + outer loop)",
         'post', iters_per_dispatch - 1),
    ]

    for title, kind, pos in sections:
        print(f"\n  [{title}]")

        # Collect stalls for this position across all dispatches
        all_stalls: dict[int, list[int]] = {}  # slot_idx → [durations]
        all_labels: dict[int, str] = {}
        for d in range(n_dispatches):
            idx = d * iters_per_dispatch + pos
            if idx >= len(ref_iters):
                continue
            it = ref_iters[idx]
            stalls = it.pre_stalls if kind == 'pre' else it.post_stalls
            for si, s in enumerate(stalls):
                if si not in all_stalls:
                    all_stalls[si] = []
                    all_labels[si] = s.label
                all_stalls[si].append(s.duration)

        if not all_stalls:
            print("    (no stalls)")
            continue

        total_avg = sum(
            sum(durs) / len(durs) for durs in all_stalls.values()
        )

        for si in sorted(all_stalls.keys()):
            durs = all_stalls[si]
            avg = sum(durs) / len(durs)
            label = all_labels.get(si, '?')
            pct = avg / total_avg * 100 if total_avg > 0 else 0
            # Determine if this is a DMA wait or lock HW access
            note = "DMA wait" if avg > 100 else "lock HW"
            print(f"    #{si+1}  {label:<14} {fmt_cycles(avg):>10} cy "
                  f"({fmt_us(avg, clock_mhz):>6} us) "
                  f"{pct:5.1f}%  {note}")

        print(f"    {'':14}  {'----------':>10}")
        print(f"    {'Total':<14} {fmt_cycles(total_avg):>10} cy "
              f"({fmt_us(total_avg, clock_mhz):>6} us)")

    # Cross-tile comparison: show all tiles for mid-transition stalls
    pids = sorted(tiles.keys())
    if len(pids) > 1:
        print(f"\n  [Cross-tile: Iter 1 pre-kernel stalls (avg over dispatches)]")
        # Header with stall labels from ref tile
        ref_it = ref_iters[1]
        labels = [s.label for s in ref_it.pre_stalls]

        header = f"    {'Tile':<18}"
        for lbl in labels:
            header += f"  {lbl:>14}"
        header += f"  {'Total':>10}"
        print(header)
        print(f"    {'-'*18}" + f"  {'-'*14}" * len(labels) + f"  {'-'*10}")

        for pid in pids:
            tile = tiles[pid]
            name = tile.tile_name.replace("core_trace for ", "")
            row = f"    {name:<18}"
            slot_avgs = []

            for si in range(len(labels)):
                durs = []
                for d in range(n_dispatches):
                    idx = d * iters_per_dispatch + 1
                    if idx < len(tile.iterations):
                        stalls = tile.iterations[idx].pre_stalls
                        if si < len(stalls):
                            durs.append(stalls[si].duration)
                if durs:
                    avg = sum(durs) / len(durs)
                    slot_avgs.append(avg)
                    row += f"  {fmt_cycles(avg):>14}"
                else:
                    row += f"  {'N/A':>14}"

            total = sum(slot_avgs)
            row += f"  {fmt_cycles(total):>10}"
            print(row)


def report_dispatch_detail(tiles: dict[int, TileTrace], tc: dict,
                            iters_per_dispatch: int, n_dispatches: int,
                            clock_mhz: float):
    """Print detailed dispatch-level (kernel.run) analysis."""
    print_header("Dispatch Detail (kernel.run level)")

    pids = sorted(tiles.keys())
    num_tiles = len(pids)
    lvl = tc["levels"][0]

    # --- Per-dispatch breakdown for reference tile ---
    ref_pid = pids[0]
    ref_tile = tiles[ref_pid]
    ref_iters = ref_tile.iterations

    # Collect per-dispatch metrics
    @dataclass
    class DispatchMetrics:
        total_time: int = 0       # first pre-gap start → last EVT1
        total_kernel: int = 0     # sum of all kernel times
        total_stall: int = 0      # sum of all stall times
        first_iter_extra: int = 0 # iter 0 pre-stall - avg(iter 1-7 pre-stall)
        rhs_acq_stall: int = 0    # RHS(B) acquire stall (from iter 0)
        lhs_acq_stall: int = 0    # sum of all LHS(A) acquire stalls
        res_acq_stall: int = 0    # sum of all RES(C) acquire stalls
        lock_hw_stall: int = 0    # sum of all release/instant stalls
        overhead: int = 0         # non-stall, non-kernel gap

    all_metrics: list = []

    for d in range(n_dispatches):
        base = d * iters_per_dispatch
        last_idx = base + iters_per_dispatch - 1
        if last_idx >= len(ref_iters):
            break

        m = DispatchMetrics()
        dispatch_start = ref_iters[base].evt0_ts - ref_iters[base].t_gap_pre
        m.total_time = ref_iters[last_idx].evt1_ts - dispatch_start

        for pos in range(iters_per_dispatch):
            idx = base + pos
            it = ref_iters[idx]
            m.total_kernel += it.t_kernel
            m.total_stall += it.t_pre_stall
            m.overhead += it.t_gap_pre - it.t_pre_stall

            # Categorize individual stalls by label
            for s in it.pre_stalls:
                if 'acq' in s.label:
                    if 'RHS' in s.label:
                        m.rhs_acq_stall += s.duration
                    elif 'LHS' in s.label:
                        m.lhs_acq_stall += s.duration
                    elif 'RES' in s.label:
                        m.res_acq_stall += s.duration
                else:
                    m.lock_hw_stall += s.duration

        # First-iter extra: dispatch setup overhead beyond steady-state
        ss_pre_avg = sum(
            ref_iters[base + p].t_pre_stall
            for p in range(1, iters_per_dispatch)
        ) / (iters_per_dispatch - 1)
        m.first_iter_extra = ref_iters[base].t_pre_stall - ss_pre_avg

        all_metrics.append(m)

    if not all_metrics:
        print("  No dispatch data.")
        return

    # --- Average dispatch breakdown ---
    def avg_field(field_name):
        vals = [getattr(m, field_name) for m in all_metrics]
        return sum(vals) / len(vals)

    avg_total = avg_field('total_time')
    avg_kernel = avg_field('total_kernel')
    avg_stall = avg_field('total_stall')
    avg_overhead = avg_field('overhead')
    avg_first_extra = avg_field('first_iter_extra')
    avg_rhs = avg_field('rhs_acq_stall')
    avg_lhs = avg_field('lhs_acq_stall')
    avg_res = avg_field('res_acq_stall')
    avg_lock_hw = avg_field('lock_hw_stall')

    print(f"\n  Reference tile: {ref_tile.tile_name}")
    print(f"  Iters/dispatch: {iters_per_dispatch}")
    print(f"")

    # Time breakdown
    print(f"  [Average Dispatch Time Breakdown]")
    print(f"    Total dispatch:    {fmt_cycles(avg_total):>12} cy  "
          f"({fmt_us(avg_total, clock_mhz):>8} us)  100.0%")
    print(f"    Kernel (x{iters_per_dispatch}):      "
          f"{fmt_cycles(avg_kernel):>12} cy  "
          f"({fmt_us(avg_kernel, clock_mhz):>8} us)  "
          f"{avg_kernel/avg_total*100:5.1f}%")
    print(f"    Stall total:       {fmt_cycles(avg_stall):>12} cy  "
          f"({fmt_us(avg_stall, clock_mhz):>8} us)  "
          f"{avg_stall/avg_total*100:5.1f}%")
    print(f"    Overhead:          {fmt_cycles(avg_overhead):>12} cy  "
          f"({fmt_us(avg_overhead, clock_mhz):>8} us)  "
          f"{avg_overhead/avg_total*100:5.1f}%")

    # Stall categorization
    print(f"")
    print(f"  [Stall Breakdown by Buffer Type]")
    print(f"    LHS(A) acq:   {fmt_cycles(avg_lhs):>12} cy  "
          f"({fmt_us(avg_lhs, clock_mhz):>8} us)  "
          f"{avg_lhs/avg_stall*100:5.1f}% of stall  "
          f"— {iters_per_dispatch}x A tile DMA wait")
    print(f"    RHS(B) acq:   {fmt_cycles(avg_rhs):>12} cy  "
          f"({fmt_us(avg_rhs, clock_mhz):>8} us)  "
          f"{avg_rhs/avg_stall*100:5.1f}% of stall  "
          f"— 1x B tile DMA wait (dispatch setup)")
    print(f"    RES(C) acq:   {fmt_cycles(avg_res):>12} cy  "
          f"({fmt_us(avg_res, clock_mhz):>8} us)  "
          f"{avg_res/avg_stall*100:5.1f}% of stall  "
          f"— {iters_per_dispatch}x C buffer ready")
    print(f"    Lock HW:      {fmt_cycles(avg_lock_hw):>12} cy  "
          f"({fmt_us(avg_lock_hw, clock_mhz):>8} us)  "
          f"{avg_lock_hw/avg_stall*100:5.1f}% of stall  "
          f"— release overhead")

    # Dispatch setup cost
    print(f"")
    print(f"  [Dispatch Setup Overhead (iter 0 extra vs steady-state)]")
    print(f"    First-iter extra stall: {fmt_cycles(avg_first_extra):>10} cy "
          f"({fmt_us(avg_first_extra, clock_mhz):>6} us)")
    print(f"    As % of dispatch:       "
          f"{avg_first_extra/avg_total*100:5.1f}%")

    # Effective compute metrics
    M, K, N = tc['M'], tc['K'], tc['N']
    flops_per_dispatch = 2 * M * K * N  # multiply-accumulate = 2 ops
    elem_bytes = 2 if tc['elemType'] == 'bf16' else 4
    # Data moved per dispatch: A + B + C (read A, read B, write C)
    bytes_per_dispatch = elem_bytes * (M * K + K * N + M * N)

    print(f"")
    print(f"  [Effective Compute Metrics (per dispatch, {num_tiles} tiles)]")
    dispatch_us = avg_total / clock_mhz
    dispatch_s = dispatch_us / 1e6
    gflops = flops_per_dispatch / dispatch_s / 1e9 if dispatch_s > 0 else 0
    gbps = bytes_per_dispatch / dispatch_s / 1e9 if dispatch_s > 0 else 0
    # Peak: each tile does r*s*t*2 = 4*8*4*2 = 256 ops/cycle for bf16 mmul
    peak_ops_per_cycle = 4 * 8 * 4 * 2  # bf16 mmul 4x8x4
    peak_gflops = num_tiles * peak_ops_per_cycle * clock_mhz * 1e6 / 1e9

    print(f"    FLOPs:          {flops_per_dispatch:>14,} (2*M*K*N)")
    print(f"    Data moved:     {bytes_per_dispatch:>14,} bytes "
          f"(A+B+C, {elem_bytes}B/elem)")
    print(f"    Dispatch time:  {dispatch_us:>14.1f} us")
    print(f"    Throughput:     {gflops:>14.1f} GFLOPS "
          f"(peak: {peak_gflops:.0f} GFLOPS, "
          f"{gflops/peak_gflops*100:.1f}%)" if peak_gflops > 0 else "")
    print(f"    Bandwidth:      {gbps:>14.2f} GB/s")

    # Kernel-only metrics (excluding stall/overhead)
    kernel_us = avg_kernel / clock_mhz
    kernel_s = kernel_us / 1e6
    kernel_gflops = flops_per_dispatch / kernel_s / 1e9 if kernel_s > 0 else 0
    print(f"    Kernel-only:    {kernel_gflops:>14.1f} GFLOPS "
          f"({gflops/kernel_gflops*100:.1f}% of kernel peak)"
          if kernel_gflops > 0 else "")

    # --- Per-dispatch table ---
    print(f"")
    print(f"  [Per-Dispatch Table]")
    print(f"  {'Disp':>4}  {'Total':>12}  {'Kernel':>12}  "
          f"{'Stall':>12}  {'Overhead':>10}  "
          f"{'Total(us)':>10}  {'Kernel%':>8}")
    print(f"  {'----':>4}  {'-'*12:>12}  {'-'*12:>12}  "
          f"{'-'*12:>12}  {'-'*10:>10}  "
          f"{'-'*10:>10}  {'-'*8:>8}")

    for d, m in enumerate(all_metrics):
        pct = m.total_kernel / m.total_time * 100 if m.total_time else 0
        print(f"  {d:>4}  {fmt_cycles(m.total_time):>12}  "
              f"{fmt_cycles(m.total_kernel):>12}  "
              f"{fmt_cycles(m.total_stall):>12}  "
              f"{fmt_cycles(m.overhead):>10}  "
              f"{fmt_us(m.total_time, clock_mhz):>10}  "
              f"{pct:>7.1f}%")

    # Std dev
    import math
    times = [m.total_time for m in all_metrics]
    avg_t = sum(times) / len(times)
    std_t = math.sqrt(sum((t - avg_t)**2 for t in times) / len(times))
    print(f"  {'':>4}  Avg: {fmt_cycles(avg_t):>10}  "
          f"Stddev: {fmt_cycles(std_t):>10}  "
          f"CV: {std_t/avg_t*100:.2f}%")

    # --- Dispatch boundary detail ---
    print(f"")
    print(f"  [Dispatch Boundary Overhead]")

    boundary_ohs = []
    for d in range(1, n_dispatches):
        base = d * iters_per_dispatch
        if base >= len(ref_iters):
            break
        boundary_gap = ref_iters[base].t_gap_pre
        ss_gap_avg = sum(
            ref_iters[base + p].t_gap_pre
            for p in range(1, iters_per_dispatch)
        ) / (iters_per_dispatch - 1)
        boundary_ohs.append(boundary_gap - ss_gap_avg)

    if boundary_ohs:
        avg_bo = sum(boundary_ohs) / len(boundary_ohs)
        print(f"    Avg boundary overhead: {fmt_cycles(avg_bo):>10} cy "
              f"({fmt_us(avg_bo, clock_mhz):>6} us)")
        print(f"    Components:")
        # Boundary = iter 0 pre-stall - steady-state pre-stall
        # = RHS rel + RHS acq + (LHS acq diff) - steady-state
        print(f"      RHS(B) acq (dispatch setup):  "
              f"{fmt_cycles(avg_rhs):>10} cy ({fmt_us(avg_rhs, clock_mhz):>6} us)")
        lhs_ss = avg_lhs / iters_per_dispatch
        lhs_extra = avg_field('first_iter_extra') - avg_rhs
        print(f"      LHS(A) acq extra (first iter): "
              f"{fmt_cycles(max(0, lhs_extra)):>9} cy "
              f"({fmt_us(max(0, lhs_extra), clock_mhz):>6} us)")
        print(f"      Lock releases (RES+LHS+RHS):  "
              f"{fmt_cycles(avg_lock_hw / iters_per_dispatch * 3):>10} cy")

    # --- Dispatch-level timeline ---
    print(f"")
    TIMELINE_WIDTH = 80
    label_width = 6
    avg_m = all_metrics[0]  # Use dispatch 0 as representative
    bar_total = avg_m.total_time
    if bar_total > 0:
        print(f"  [Dispatch Composition (dispatch 0)]")

        # Build detailed bar: B-load | [A-load kernel A-load kernel ...]
        # Simplified: show iter-by-iter
        row = []
        base = 0
        for pos in range(iters_per_dispatch):
            it = ref_iters[pos]
            stall_frac = it.t_pre_stall / bar_total * TIMELINE_WIDTH
            kernel_frac = it.t_kernel / bar_total * TIMELINE_WIDTH
            row.extend(['.'] * max(1, round(stall_frac)))
            row.extend(['#'] * max(1, round(kernel_frac)))

        # Trim or pad to width
        row = row[:TIMELINE_WIDTH]
        while len(row) < TIMELINE_WIDTH:
            row.append(' ')

        # Add iteration markers
        marker_row = [' '] * TIMELINE_WIDTH
        cumulative = 0
        for pos in range(iters_per_dispatch):
            it = ref_iters[pos]
            iter_frac = (it.t_gap_pre + it.t_kernel) / bar_total
            col = min(TIMELINE_WIDTH - 1, int(cumulative * TIMELINE_WIDTH))
            if col < TIMELINE_WIDTH:
                marker_row[col] = str(pos)
            cumulative += iter_frac

        print(f"    {'':>{label_width}} |{''.join(row)}|")
        print(f"    {'iter':>{label_width}} |{''.join(marker_row)}|")
        print(f"    {'':>{label_width}}  . = stall  # = kernel")
        print(f"    {'':>{label_width}}  "
              f"B-load({fmt_us(avg_rhs, clock_mhz)}us) + "
              f"{iters_per_dispatch}x [A-wait({fmt_us(lhs_ss, clock_mhz)}us) + "
              f"kernel({fmt_us(avg_kernel/iters_per_dispatch, clock_mhz)}us)]")


def report_cross_tile(tiles: dict[int, TileTrace], iters_per_dispatch: int,
                       n_dispatches: int, clock_mhz: float):
    """Print cross-tile parallelism analysis."""
    print_header("Cross-Tile Parallelism")

    pids = sorted(tiles.keys())
    if len(pids) < 2:
        print("  Only 1 tile — skipping cross-tile analysis.")
        return

    # Per-dispatch: tile start skew (first EVT0) and end skew (last EVT1)
    print(f"\n  Tile start skew (first EVT0 per dispatch, relative to earliest):")
    print(f"  {'Disp':>4}", end="")
    for pid in pids:
        print(f"  {tiles[pid].tile_name:>18}", end="")
    print(f"  {'Spread':>10}")

    start_skews = []
    end_skews = []
    for d in range(n_dispatches):
        starts = {}
        ends = {}
        for pid in pids:
            it = tiles[pid].iterations
            base = d * iters_per_dispatch
            last = base + iters_per_dispatch - 1
            if last < len(it):
                starts[pid] = it[base].evt0_ts
                ends[pid] = it[last].evt1_ts

        if len(starts) == len(pids):
            min_start = min(starts.values())
            max_start = max(starts.values())
            spread = max_start - min_start
            start_skews.append(spread)

            min_end = min(ends.values())
            max_end = max(ends.values())
            end_skews.append(max_end - min_end)

            if d < 3 or d == n_dispatches - 1:  # Show first 3 + last
                print(f"  {d:>4}", end="")
                for pid in pids:
                    print(f"  {starts[pid] - min_start:>18,}", end="")
                print(f"  {spread:>10,}")

    if start_skews:
        avg_start = sum(start_skews) / len(start_skews)
        avg_end = sum(end_skews) / len(end_skews)
        print(f"\n  Avg start skew: {fmt_cycles(avg_start)} cycles "
              f"({fmt_us(avg_start, clock_mhz)} us)")
        print(f"  Avg end skew:   {fmt_cycles(avg_end)} cycles "
              f"({fmt_us(avg_end, clock_mhz)} us)")

    # Parallel execution overlap
    print(f"\n  Parallel execution analysis (dispatch average):")
    overlap_ratios = []
    for d in range(n_dispatches):
        # Collect all (evt0, evt1) intervals across tiles
        intervals = []
        for pid in pids:
            base = d * iters_per_dispatch
            for pos in range(iters_per_dispatch):
                idx = base + pos
                if idx < len(tiles[pid].iterations):
                    it = tiles[pid].iterations[idx]
                    intervals.append((it.evt0_ts, it.evt1_ts))

        if not intervals:
            continue

        # Compute union of kernel intervals vs sum of individual
        intervals.sort()
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        union_time = sum(e - s for s, e in merged)
        sum_time = sum(e - s for s, e in intervals)
        if union_time > 0:
            overlap_ratios.append(sum_time / union_time)

    if overlap_ratios:
        avg_overlap = sum(overlap_ratios) / len(overlap_ratios)
        print(f"  Avg parallelism factor: {avg_overlap:.2f}x "
              f"(ideal={len(pids)}x for {len(pids)} tiles)")
        print(f"  Parallel efficiency: "
              f"{avg_overlap / len(pids) * 100:.1f}%")


def report_data_transfer(tiles: dict[int, TileTrace], tc: dict,
                          iters_per_dispatch: int, n_dispatches: int,
                          clock_mhz: float):
    """Infer data transfer patterns from stall analysis."""
    print_header("Data Transfer Analysis")

    lvl = tc["levels"][0]
    inner_axis = lvl["tpOrder"][0]
    axis_name = "MNK"[inner_axis]

    # DMA schedule description
    if inner_axis == 0:  # M-inner
        fixed_mat, per_step_mat = "B (RHS)", "A (LHS)"
        fixed_desc = (f"B[{lvl['TN']}x{lvl['TK']}] × {lvl['SPn']} tiles "
                      f"(different data → serial)")
        step_desc = (f"A[{lvl['TM']}x{lvl['TK']}] × {lvl['SPm']} unique "
                     f"(same for all N-tiles → multicast possible)")
    elif inner_axis == 1:  # N-inner
        fixed_mat, per_step_mat = "A (LHS)", "B (RHS)"
        fixed_desc = (f"A[{lvl['TM']}x{lvl['TK']}] × {lvl['SPm']} tiles "
                      f"(different data → serial)")
        step_desc = (f"B[{lvl['TN']}x{lvl['TK']}] × {lvl['SPn']} unique "
                     f"(same for all M-tiles → multicast possible)")
    else:  # K-inner
        fixed_mat, per_step_mat = "C (RES)", "A (LHS) + B (RHS)"
        fixed_desc = f"C[{lvl['TM']}x{lvl['TN']}] collected once"
        step_desc = f"A + B sent per K-step"

    print(f"\n  Inner axis: {axis_name} (tpOrder[0]={inner_axis})")
    print(f"  Fixed (1x):    {fixed_mat} — {fixed_desc}")
    print(f"  Per-step ({iters_per_dispatch}x): {per_step_mat} — {step_desc}")
    print(f"  Result:        C (RES) collected per step")

    # Iter 0 vs iter 1+ stall difference → fixed matrix transfer overhead
    pids = sorted(tiles.keys())
    print(f"\n  Fixed-matrix transfer overhead (iter0 - avg(iter1-7) pre-stall):")

    for pid in pids:
        tile = tiles[pid]
        iter0_stalls = []
        iterN_stalls = []
        for d in range(n_dispatches):
            base = d * iters_per_dispatch
            if base + iters_per_dispatch <= len(tile.iterations):
                iter0_stalls.append(tile.iterations[base].t_pre_stall)
                for p in range(1, iters_per_dispatch):
                    iterN_stalls.append(tile.iterations[base + p].t_pre_stall)

        if iter0_stalls and iterN_stalls:
            avg_i0 = sum(iter0_stalls) / len(iter0_stalls)
            avg_iN = sum(iterN_stalls) / len(iterN_stalls)
            diff = avg_i0 - avg_iN
            print(f"    {tile.tile_name}: {fmt_cycles(diff)} cycles "
                  f"({fmt_us(diff, clock_mhz)} us)")

    # Cross-tile stagger analysis
    print(f"\n  Cross-tile stagger (relative to earliest tile per iteration):")
    print(f"  {'Iter':>4}", end="")
    for pid in pids:
        name = tiles[pid].tile_name.replace("core_trace for ", "")
        print(f"  {name:>10}", end="")
    print(f"  {'Spread':>10}  {'Note':>20}")

    # Average stagger for iter 0 and iter 1
    for pos in [0, 1]:
        staggers = {pid: [] for pid in pids}
        for d in range(n_dispatches):
            ts_map = {}
            for pid in pids:
                idx = d * iters_per_dispatch + pos
                if idx < len(tiles[pid].iterations):
                    ts_map[pid] = tiles[pid].iterations[idx].evt0_ts
            if len(ts_map) == len(pids):
                min_ts = min(ts_map.values())
                for pid in pids:
                    staggers[pid].append(ts_map[pid] - min_ts)

        if all(staggers[pid] for pid in pids):
            avgs = {pid: sum(staggers[pid]) / len(staggers[pid])
                    for pid in pids}
            min_avg = min(avgs.values())
            spread = max(avgs.values()) - min_avg
            note = ("← includes fixed-mat load" if pos == 0
                    else "← per-step only")
            print(f"  {pos:>4}", end="")
            for pid in pids:
                print(f"  {avgs[pid] - min_avg:>10,.0f}", end="")
            print(f"  {spread:>10,.0f}  {note:>20}")

    # Serial transfer estimate
    if len(pids) >= 2:
        # Compute average stagger between consecutive tiles at iter 0
        iter0_starts = {pid: [] for pid in pids}
        for d in range(n_dispatches):
            for pid in pids:
                idx = d * iters_per_dispatch
                if idx < len(tiles[pid].iterations):
                    iter0_starts[pid].append(tiles[pid].iterations[idx].evt0_ts)

        avg_starts = {}
        for pid in pids:
            if iter0_starts[pid]:
                avg_starts[pid] = (sum(iter0_starts[pid])
                                    / len(iter0_starts[pid]))

        if len(avg_starts) == len(pids):
            sorted_tiles = sorted(avg_starts.items(), key=lambda x: x[1])
            print(f"\n  Tile start order (avg iter 0):")
            for i, (pid, ts) in enumerate(sorted_tiles):
                name = tiles[pid].tile_name.replace("core_trace for ", "")
                delta = ts - sorted_tiles[0][1]
                gap = (ts - sorted_tiles[i-1][1]) if i > 0 else 0
                print(f"    {i+1}. {name}: +{delta:,.0f} cycles"
                      f" (gap from prev: {gap:,.0f})")


def report_timeline(tiles: dict[int, TileTrace], iters_per_dispatch: int,
                     n_dispatches: int, clock_mhz: float):
    """Print ASCII timeline visualization of tile activity."""
    pids = sorted(tiles.keys())
    if not pids:
        return

    # Use first measurement dispatch (dispatch 0) for visualization
    DISPATCH_IDX = 0

    # ---- Full dispatch timeline ----
    print_header("Timeline: Full Dispatch (dispatch 0)")

    # Gather time range across all tiles for this dispatch
    all_evt0, all_evt1 = [], []
    for pid in pids:
        tile = tiles[pid]
        base = DISPATCH_IDX * iters_per_dispatch
        for pos in range(iters_per_dispatch):
            idx = base + pos
            if idx < len(tile.iterations):
                it = tile.iterations[idx]
                all_evt0.append(it.evt0_ts)
                all_evt1.append(it.evt1_ts)

    if not all_evt0:
        print("  No data for dispatch 0.")
        return

    # Time window: from earliest pre-gap start to latest EVT1
    t_min = min(all_evt0)
    for pid in pids:
        base = DISPATCH_IDX * iters_per_dispatch
        if base < len(tiles[pid].iterations):
            it0 = tiles[pid].iterations[base]
            t_min = min(t_min, it0.evt0_ts - it0.t_gap_pre)
    t_max = max(all_evt1)

    TIMELINE_WIDTH = 100
    t_range = t_max - t_min
    if t_range == 0:
        return
    scale = t_range / TIMELINE_WIDTH  # cycles per character

    def time_to_col(t: int) -> int:
        return min(TIMELINE_WIDTH - 1, max(0, int((t - t_min) / scale)))

    # Legend
    print(f"\n  Legend: # = kernel  . = lock stall  _ = idle/overhead")
    print(f"  Scale: 1 char = {fmt_cycles(scale)} cycles ({fmt_us(scale, clock_mhz)} us)")
    print(f"  Time range: {fmt_cycles(t_range)} cycles ({fmt_us(t_range, clock_mhz)} us)")
    print()

    # Render each tile as a row
    label_width = 10
    for pid in pids:
        tile = tiles[pid]
        name = tile.tile_name.replace("core_trace for ", "")
        row = ['_'] * TIMELINE_WIDTH

        base = DISPATCH_IDX * iters_per_dispatch

        # Paint lock stalls
        for b, e in tile.stall_pairs:
            if b >= t_max or e <= t_min:
                continue
            c0 = time_to_col(b)
            c1 = time_to_col(e)
            for c in range(c0, c1 + 1):
                if 0 <= c < TIMELINE_WIDTH:
                    row[c] = '.'

        # Paint kernel execution (overwrites stall if overlapping)
        for pos in range(iters_per_dispatch):
            idx = base + pos
            if idx < len(tile.iterations):
                it = tile.iterations[idx]
                c0 = time_to_col(it.evt0_ts)
                c1 = time_to_col(it.evt1_ts)
                for c in range(c0, c1 + 1):
                    if 0 <= c < TIMELINE_WIDTH:
                        row[c] = '#'

        print(f"  {name:>{label_width}} |{''.join(row)}|")

    # Time axis
    n_ticks = 5
    tick_positions = [int(TIMELINE_WIDTH * i / n_ticks) for i in range(n_ticks + 1)]
    axis_line = [' '] * TIMELINE_WIDTH
    for tp in tick_positions:
        if tp < TIMELINE_WIDTH:
            axis_line[tp] = '|'
    print(f"  {'':>{label_width}} +{''.join(axis_line)}+")

    # Tick labels
    label_line = [' '] * (TIMELINE_WIDTH + 10)
    for tp in tick_positions:
        t_val = t_min + tp * scale
        label = f"{t_val/clock_mhz:.0f}"
        start = max(0, tp - len(label) // 2)
        for ci, ch in enumerate(label):
            if start + ci < len(label_line):
                label_line[start + ci] = ch
    print(f"  {'(us)':>{label_width}}  {''.join(label_line)}")

    # ---- Zoomed: first 2 iterations ----
    print_header("Timeline: Zoomed (first 2 iterations, dispatch 0)")

    # Time window: from dispatch start to end of iteration 1
    z_min = t_min
    z_max_candidates = []
    for pid in pids:
        base = DISPATCH_IDX * iters_per_dispatch
        idx = base + 1  # end of iteration 1
        if idx < len(tiles[pid].iterations):
            it = tiles[pid].iterations[idx]
            z_max_candidates.append(it.evt1_ts + int(it.t_gap_post * 0.1))
    z_max = max(z_max_candidates) if z_max_candidates else t_max

    z_range = z_max - z_min
    z_scale = z_range / TIMELINE_WIDTH
    if z_scale == 0:
        return

    def z_time_to_col(t: int) -> int:
        return min(TIMELINE_WIDTH - 1, max(0, int((t - z_min) / z_scale)))

    print(f"\n  Legend: # = kernel  . = lock stall  _ = idle/overhead")
    print(f"  Scale: 1 char = {fmt_cycles(z_scale)} cycles ({fmt_us(z_scale, clock_mhz)} us)")
    print()

    for pid in pids:
        tile = tiles[pid]
        name = tile.tile_name.replace("core_trace for ", "")
        row = ['_'] * TIMELINE_WIDTH

        # Paint lock stalls in zoom range
        for b, e in tile.stall_pairs:
            if b >= z_max or e <= z_min:
                continue
            c0 = z_time_to_col(b)
            c1 = z_time_to_col(e)
            for c in range(c0, c1 + 1):
                if 0 <= c < TIMELINE_WIDTH:
                    row[c] = '.'

        # Paint kernels
        base = DISPATCH_IDX * iters_per_dispatch
        for pos in range(min(2, iters_per_dispatch)):
            idx = base + pos
            if idx < len(tile.iterations):
                it = tile.iterations[idx]
                c0 = z_time_to_col(it.evt0_ts)
                c1 = z_time_to_col(it.evt1_ts)
                for c in range(c0, c1 + 1):
                    if 0 <= c < TIMELINE_WIDTH:
                        row[c] = '#'

        print(f"  {name:>{label_width}} |{''.join(row)}|")

    # Time axis
    axis_line = [' '] * TIMELINE_WIDTH
    for tp in tick_positions:
        if tp < TIMELINE_WIDTH:
            axis_line[tp] = '|'
    print(f"  {'':>{label_width}} +{''.join(axis_line)}+")

    label_line = [' '] * (TIMELINE_WIDTH + 10)
    for tp in tick_positions:
        t_val = z_min + tp * z_scale
        label = f"{t_val/clock_mhz:.0f}"
        start = max(0, tp - len(label) // 2)
        for ci, ch in enumerate(label):
            if start + ci < len(label_line):
                label_line[start + ci] = ch
    print(f"  {'(us)':>{label_width}}  {''.join(label_line)}")

    # ---- Zoomed: single steady-state iteration (iter 1) ----
    print_header("Timeline: Single Iteration (iter 1, all tiles aligned)")

    # Show iter 1 for each tile, aligned to each tile's own prev_evt1
    # This shows the stall→kernel pattern at high resolution
    ref_pid = pids[0]
    base = DISPATCH_IDX * iters_per_dispatch + 1
    if base >= len(tiles[ref_pid].iterations):
        return

    # Use iter 1 of first tile to define the time window
    ref_it = tiles[ref_pid].iterations[base]
    ref_prev_end = tiles[ref_pid].iterations[base - 1].evt1_ts
    s_min = ref_prev_end
    s_max = ref_it.evt1_ts + 1000  # small padding

    s_range = s_max - s_min
    s_scale = s_range / TIMELINE_WIDTH
    if s_scale == 0:
        return

    def s_time_to_col(t: int) -> int:
        return min(TIMELINE_WIDTH - 1, max(0, int((t - s_min) / s_scale)))

    print(f"\n  Legend: # = kernel  . = lock stall  _ = idle/overhead")
    print(f"  Scale: 1 char = {fmt_cycles(s_scale)} cycles ({fmt_us(s_scale, clock_mhz)} us)")
    print()

    for pid in pids:
        tile = tiles[pid]
        name = tile.tile_name.replace("core_trace for ", "")
        row = ['_'] * TIMELINE_WIDTH

        # Paint stalls in this range
        for b, e in tile.stall_pairs:
            if b >= s_max or e <= s_min:
                continue
            c0 = s_time_to_col(b)
            c1 = s_time_to_col(e)
            for c in range(c0, c1 + 1):
                if 0 <= c < TIMELINE_WIDTH:
                    row[c] = '.'

        # Paint kernel
        idx = DISPATCH_IDX * iters_per_dispatch + 1
        if idx < len(tile.iterations):
            it = tile.iterations[idx]
            c0 = s_time_to_col(it.evt0_ts)
            c1 = s_time_to_col(it.evt1_ts)
            for c in range(c0, c1 + 1):
                if 0 <= c < TIMELINE_WIDTH:
                    row[c] = '#'

        print(f"  {name:>{label_width}} |{''.join(row)}|")

    # Time axis with labels
    axis_line = [' '] * TIMELINE_WIDTH
    for tp in tick_positions:
        if tp < TIMELINE_WIDTH:
            axis_line[tp] = '|'
    print(f"  {'':>{label_width}} +{''.join(axis_line)}+")

    label_line = [' '] * (TIMELINE_WIDTH + 10)
    for tp in tick_positions:
        t_val = s_min + tp * s_scale
        label = f"{t_val/clock_mhz:.0f}"
        start = max(0, tp - len(label) // 2)
        for ci, ch in enumerate(label):
            if start + ci < len(label_line):
                label_line[start + ci] = ch
    print(f"  {'(us)':>{label_width}}  {''.join(label_line)}")

    # ---- Proportional bar chart: 1 iteration breakdown ----
    print_header("Proportional Breakdown: Average Steady-State Iteration")

    BAR_WIDTH = 80
    for pid in pids:
        tile = tiles[pid]
        name = tile.tile_name.replace("core_trace for ", "")

        ss_pre, ss_ker, ss_oh = [], [], []
        for d in range(n_dispatches):
            for pos in range(1, iters_per_dispatch):
                idx = d * iters_per_dispatch + pos
                if idx < len(tile.iterations):
                    it = tile.iterations[idx]
                    ss_pre.append(it.t_pre_stall)
                    ss_ker.append(it.t_kernel)
                    ss_oh.append(it.t_gap_pre - it.t_pre_stall)

        if not ss_pre:
            continue

        avg_pre = sum(ss_pre) / len(ss_pre)
        avg_ker = sum(ss_ker) / len(ss_ker)
        avg_oh = sum(ss_oh) / len(ss_oh)
        avg_tot = avg_pre + avg_ker + avg_oh

        n_pre = max(1, round(avg_pre / avg_tot * BAR_WIDTH))
        n_ker = max(1, round(avg_ker / avg_tot * BAR_WIDTH))
        n_oh = BAR_WIDTH - n_pre - n_ker

        bar = '.' * n_pre + '#' * n_ker + '_' * max(0, n_oh)
        print(f"  {name:>{label_width}} |{bar}|")

    print(f"\n  {'':>{label_width}}  "
          f". = stall ({fmt_cycles(avg_pre)} cy, {avg_pre/avg_tot*100:.1f}%)  "
          f"# = kernel ({fmt_cycles(avg_ker)} cy, {avg_ker/avg_tot*100:.1f}%)  "
          f"_ = overhead ({fmt_cycles(avg_oh)} cy, {avg_oh/avg_tot*100:.1f}%)")


def report_summary(tiles: dict[int, TileTrace], iters_per_dispatch: int,
                    n_dispatches: int, clock_mhz: float):
    """Print final 1-iteration average summary."""
    print_header("Average Single Iteration Summary (steady-state, iter 1+)")

    pids = sorted(tiles.keys())
    all_pre, all_ker, all_oh, all_tot = [], [], [], []

    for pid in pids:
        tile = tiles[pid]
        for d in range(n_dispatches):
            for pos in range(1, iters_per_dispatch):
                idx = d * iters_per_dispatch + pos
                if idx < len(tile.iterations):
                    it = tile.iterations[idx]
                    pre = it.t_pre_stall
                    ker = it.t_kernel
                    oh = it.t_gap_pre - it.t_pre_stall
                    tot = it.t_gap_pre + it.t_kernel
                    all_pre.append(pre)
                    all_ker.append(ker)
                    all_oh.append(oh)
                    all_tot.append(tot)

    if not all_tot:
        print("  No steady-state iterations found.")
        return

    avg_pre = sum(all_pre) / len(all_pre)
    avg_ker = sum(all_ker) / len(all_ker)
    avg_oh = sum(all_oh) / len(all_oh)
    avg_tot = sum(all_tot) / len(all_tot)

    print(f"  Kernel:     {fmt_cycles(avg_ker):>12} cycles "
          f"({avg_ker/avg_tot*100:5.1f}%)  {fmt_us(avg_ker, clock_mhz):>8} us")
    print(f"  Pre-stall:  {fmt_cycles(avg_pre):>12} cycles "
          f"({avg_pre/avg_tot*100:5.1f}%)  {fmt_us(avg_pre, clock_mhz):>8} us")
    print(f"  Overhead:   {fmt_cycles(avg_oh):>12} cycles "
          f"({avg_oh/avg_tot*100:5.1f}%)  {fmt_us(avg_oh, clock_mhz):>8} us")
    print(f"  Total:      {fmt_cycles(avg_tot):>12} cycles "
          f"(100.0%)  {fmt_us(avg_tot, clock_mhz):>8} us")
    print(f"\n  NPU Utilization: {avg_ker/avg_tot*100:.1f}% "
          f"(kernel / total per iteration)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze NPU trace data for per-iteration timing breakdown"
    )
    parser.add_argument("--trace", required=True,
                        help="Path to trace.json (parse_trace.py output)")
    parser.add_argument("--tc", required=True,
                        help="Path to tc.json (tiling configuration)")
    parser.add_argument("--n-dispatches", type=int, default=10,
                        help="Number of measurement dispatches (default: 10)")
    parser.add_argument("--clock-mhz", type=float,
                        default=XDNA2_DEFAULT_CLOCK_MHZ,
                        help=f"NPU clock frequency in MHz "
                             f"(default: {XDNA2_DEFAULT_CLOCK_MHZ})")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: stdout)")
    args = parser.parse_args()

    # Redirect stdout if output specified
    if args.output:
        sys.stdout = open(args.output, "w")

    # Load inputs
    tc = load_tc_json(args.tc)
    tiles = parse_trace_json(args.trace)

    # Determine iterations per dispatch from tc.json
    lvl = tc["levels"][0]
    inner_axis = lvl["tpOrder"][0]
    tp_map = {0: lvl["TPm"], 1: lvl["TPn"], 2: lvl["TPk"]}
    iters_per_dispatch = tp_map[inner_axis]

    # Build iteration structures with labeled stalls
    stall_labels = get_stall_labels(inner_axis)
    for tile in tiles.values():
        tile.iterations = build_iterations(tile, iters_per_dispatch, stall_labels)

    # Validate
    for pid in sorted(tiles.keys()):
        tile = tiles[pid]
        expected = iters_per_dispatch * args.n_dispatches
        actual = len(tile.iterations)
        if actual != expected:
            print(f"WARNING: {tile.tile_name} has {actual} iterations, "
                  f"expected {expected}", file=sys.stderr)

    # Detect dispatch boundaries (for validation)
    for tile in tiles.values():
        detect_dispatch_boundaries(
            tile.iterations, iters_per_dispatch, args.n_dispatches
        )

    # Generate reports
    report_config(tc, args.n_dispatches, iters_per_dispatch, args.clock_mhz)
    report_per_iteration(tiles, iters_per_dispatch, args.n_dispatches,
                         args.clock_mhz)
    report_stall_detail(tiles, iters_per_dispatch, args.n_dispatches,
                        args.clock_mhz)
    report_dispatch_detail(tiles, tc, iters_per_dispatch, args.n_dispatches,
                           args.clock_mhz)
    report_cross_tile(tiles, iters_per_dispatch, args.n_dispatches,
                      args.clock_mhz)
    report_data_transfer(tiles, tc, iters_per_dispatch, args.n_dispatches,
                         args.clock_mhz)
    report_timeline(tiles, iters_per_dispatch, args.n_dispatches,
                    args.clock_mhz)
    report_summary(tiles, iters_per_dispatch, args.n_dispatches,
                   args.clock_mhz)

    if args.output:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
