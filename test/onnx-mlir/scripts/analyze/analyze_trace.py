#!/usr/bin/env python3
"""Analyze NPU trace data: kernel timing + DMA transfer analysis.

Reads parse_trace.py output (trace.json) and tiling config (tc.json).
Uses memory module trace events (STREAM_STARVATION, STALLED_LOCK, FINISHED_BD)
to measure actual DMA transfer windows per buffer type (LHS/RHS/RES).

Trace event layout (per compute tile):
  Core trace (pid 0..N-1):
    - INSTR_EVENT_0: kernel start
    - INSTR_EVENT_1: kernel end
  Memory module trace (pid N..2N-1):
    - DMA_S2MM_0_STREAM_STARVATION + FINISHED_BD: LHS input (+ PRES when M-inner)
    - DMA_S2MM_1_STREAM_STARVATION + FINISHED_BD: RHS input (+ PRES when N-inner)
    - DMA_MM2S_0_STALLED_LOCK + FINISHED_BD: RES output
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XDNA2_DEFAULT_CLOCK_MHZ = 1500
# Starvation/stall gaps longer than this are inter-iteration idle periods,
# not fill-latency bursts within a single transfer.
LONG_GAP_THRESHOLD = 1000  # cycles


# ---------------------------------------------------------------------------
# Raw trace timestamp calibration
# ---------------------------------------------------------------------------
# parse_trace.py (upstream) reconstructs timestamps by accumulating cycle
# deltas from timer=0 per tile.  The hardware resets all trace timers
# simultaneously via BROADCAST_15 at each dispatch start and records the
# absolute timer value in a Start command.  parse_trace.py ignores Start,
# so tiles with different DMA wait times accumulate different delta totals
# per dispatch, causing cross-tile span to grow linearly.
#
# Fix: extract Start timer_values from the raw hex trace and re-anchor
# each tile's timestamps at every Start boundary.
# ---------------------------------------------------------------------------

def _check_pkt_parity(word: int) -> bool:
    val = 0
    for i in range(32):
        val ^= ((word >> i) & 1)
    return val == 1


def _parse_raw_trace_streams(raw_path: str) -> dict:
    """Parse raw trace hex dump into per-tile byte streams.

    Returns {(pkt_type, row, col): list[int]}  (byte values).
    """
    with open(raw_path) as f:
        words = [w.strip() for w in f if w.strip()]

    # Trim at fefefefe sentinel
    for i, w in enumerate(words):
        if w.lower() == "fefefefe" and i + 2 < len(words):
            if words[i + 1] == "00000000" and words[i + 2] == "00000000":
                words = words[: i + 1]
                break

    # De-interleave by packet header (every 8th word)
    streams: dict[tuple, list[str]] = {}
    curr_key = None
    for i, w in enumerate(words):
        if i % 8 == 0:
            val = int(w, 16)
            if _check_pkt_parity(val):
                row = (val >> 16) & 0x1F
                col = (val >> 21) & 0x7F
                ptype = (val >> 12) & 0x3
                unused_ok = (
                    ((val >> 5) & 0x7F) == 0
                    and ((val >> 19) & 0x1) == 0
                    and ((val >> 28) & 0x7) == 0
                )
                if unused_ok:
                    curr_key = (ptype, row, col)
                    if curr_key not in streams:
                        streams[curr_key] = []
        else:
            if curr_key is not None:
                streams[curr_key].append(w)

    # Convert hex words to byte arrays
    byte_streams: dict[tuple, list[int]] = {}
    for key, hex_words in streams.items():
        bs: list[int] = []
        for w in hex_words:
            if w in ("", "a5a5a5a5", "fefefefe"):
                continue
            val = int(w, 16)
            for top in range(4):
                byte = 3 - top
                bs.append((val >> (byte * 8)) & 0xFF)
        byte_streams[key] = bs

    return byte_streams


def _extract_start_corrections(byte_stream: list[int]) -> list[tuple]:
    """Walk trace byte stream tracking accumulated timer vs Start hw_timer.

    Returns [(accumulated_at_start, hw_timer), ...] for each Start command.
    """
    corrections: list[tuple] = []
    timer = 0
    cursor = 0

    try:
        while cursor < len(byte_stream):
            b = byte_stream[cursor]

            if (b & 0b11111011) == 0b11110000:  # Start command (8 bytes)
                hw_timer = 0
                for i in range(7):
                    if cursor + i + 1 < len(byte_stream):
                        hw_timer += byte_stream[cursor + i + 1] * (256 ** (6 - i))
                corrections.append((timer, hw_timer))
                cursor += 8
                continue

            if (b & 0b10000000) == 0:  # Single0
                timer += 1 + (b & 0xF)
                cursor += 1; continue
            if (b & 0b11100000) == 0b10000000:  # Single1
                timer += 1 + (b & 0x3) * 256 + byte_stream[cursor + 1]
                cursor += 2; continue
            if (b & 0b11100000) == 0b10100000:  # Single2
                timer += 1 + ((b & 0x3) * 65536
                              + byte_stream[cursor + 1] * 256
                              + byte_stream[cursor + 2])
                cursor += 3; continue
            if (b & 0b11110000) == 0b11000000:  # Multiple0
                timer += 1 + (byte_stream[cursor + 1] & 0xF)
                cursor += 2; continue
            if (b & 0b11111100) == 0b11010000:  # Multiple1
                timer += 1 + ((byte_stream[cursor + 1] & 0x3) * 256
                              + byte_stream[cursor + 2])
                cursor += 3; continue
            if (b & 0b11111100) == 0b11010100:  # Multiple2
                timer += 1 + ((byte_stream[cursor + 1] & 0x3) * 65536
                              + byte_stream[cursor + 2] * 256
                              + byte_stream[cursor + 3])
                cursor += 4; continue
            if (b & 0b11110000) == 0b11100000:  # Repeat0
                timer += b & 0xF
                cursor += 1; continue
            if (b & 0b11111100) == 0b11011000:  # Repeat1
                timer += (b & 0x3) * 256 + byte_stream[cursor + 1]
                cursor += 2; continue
            if (b & 0b11111100) == 0b11011100:  # 4-byte skip
                cursor += 4; continue
            if b == 0xFE:  # filler
                cursor += 1; continue
            if b == 0xFF:  # Event_Sync
                timer += 0x3FFFF
                cursor += 1; continue
            cursor += 1  # unknown byte
    except IndexError:
        pass

    return corrections


def apply_timestamp_corrections(tiles, raw_path: str) -> list[int]:
    """Correct cross-tile timestamp drift using Start timer_values.

    parse_trace.py accumulates deltas from timer=0 per tile.  Because
    tiles have different event counts per dispatch, the accumulated timer
    drifts apart across tiles.  The hardware Start command records the
    same hw_timer across all tiles at each dispatch boundary, enabling
    drift correction.

    Strategy: for each tile pair of Start commands with matching hw_timer,
    compute the drift between the reference tile's accumulated and this
    tile's accumulated.  Interpolate the drift linearly between Start
    boundaries to smoothly correct all events.

    Returns list of hw_timer values at dispatch boundaries (one per
    dispatch).  These are absolute NPU clock values and can be used to
    compute full dispatch cycle times including host overhead:
      cycle[d] = hw_timers[d+1] - hw_timers[d]
    """
    byte_streams = _parse_raw_trace_streams(raw_path)

    # Build corrections per tile per trace type (0=core, 1=mem)
    tile_corrections: dict[str, dict[int, list[tuple]]] = {}
    for (ptype, row, col), bs in byte_streams.items():
        if ptype not in (0, 1):
            continue
        tile_key = f"tile{row},{col}"
        if tile_key not in tile_corrections:
            tile_corrections[tile_key] = {}
        tile_corrections[tile_key][ptype] = _extract_start_corrections(bs)

    # Collect hw_timer values at dispatch boundaries from core traces
    hw_timers: list[int] = []

    for trace_type in (0, 1):
        all_corrs = {}
        for tile_key, type_dict in tile_corrections.items():
            if trace_type in type_dict and type_dict[trace_type]:
                all_corrs[tile_key] = type_dict[trace_type]

        if len(all_corrs) < 2:
            continue

        # Group Start commands into dispatch boundaries.
        # Each Start records (accumulated_timer, hw_timer) where hw_timer
        # is the pre-reset cycle count (= time since previous Start).
        # Multiple Starts at the SAME dispatch boundary have nearly
        # identical accumulated values; separate dispatches have large
        # accumulated gaps (at least one kernel execution apart).
        def _group_boundaries(corrs):
            groups = []
            for acc, hw in corrs:
                if not groups or (acc - groups[-1][0]) > 1000:
                    groups.append((acc, hw))
            return groups

        grouped = {k: _group_boundaries(c) for k, c in all_corrs.items()}
        n_common = min(len(g) for g in grouped.values())
        if n_common < 2:
            continue

        # Reference tile: largest accumulated at last common boundary
        ref_key = max(grouped, key=lambda k: grouped[k][n_common - 1][0])
        ref_bounds = grouped[ref_key]

        # Extract hw_timer values from core trace boundaries
        if trace_type == 0 and not hw_timers:
            hw_timers = [ref_bounds[i][1] for i in range(n_common)]

        for tile_key, bounds in grouped.items():
            if tile_key == ref_key:
                continue

            td = next((t for t in tiles if t.tile_name == tile_key), None)
            if td is None:
                continue

            # Build drift at each boundary: drift_i = ref_acc_i - this_acc_i
            # The drift increases linearly with dispatch index.
            # For events between boundary i and i+1, interpolate linearly.
            boundary_drifts = []
            for idx in range(min(n_common, len(bounds))):
                this_acc = bounds[idx][0]
                ref_acc = ref_bounds[idx][0]
                boundary_drifts.append((this_acc, ref_acc - this_acc))

            # Apply interpolated drift to all events
            if trace_type == 0:
                _apply_interpolated_drift_core(td, boundary_drifts)
            else:
                _apply_interpolated_drift_mem(td, boundary_drifts)

    return hw_timers


def _interpolate_drift(ts: int, boundaries: list[tuple]) -> int:
    """Compute interpolated drift correction for a given timestamp.

    boundaries: [(accumulated_boundary, drift), ...] sorted by accumulated.
    Between boundaries, drift is linearly interpolated.
    Before first boundary: use first drift.
    After last boundary: use last drift.
    """
    if not boundaries:
        return 0
    if ts <= boundaries[0][0]:
        return boundaries[0][1]
    if ts >= boundaries[-1][0]:
        return boundaries[-1][1]

    # Find the bracketing boundaries
    for i in range(len(boundaries) - 1):
        a0, d0 = boundaries[i]
        a1, d1 = boundaries[i + 1]
        if a0 <= ts < a1:
            # Linear interpolation
            if a1 == a0:
                return d0
            frac = (ts - a0) / (a1 - a0)
            return round(d0 + frac * (d1 - d0))
    return boundaries[-1][1]


def _apply_interpolated_drift_core(td, boundaries: list[tuple]) -> None:
    """Apply interpolated drift correction to kernel timestamps."""
    for k in td.kernels:
        drift = _interpolate_drift(k.evt0_ts, boundaries)
        k.evt0_ts += drift
        k.evt1_ts += drift
    for j in range(len(td.evt0_ts)):
        td.evt0_ts[j] += _interpolate_drift(td.evt0_ts[j], boundaries)
    for j in range(len(td.evt1_ts)):
        td.evt1_ts[j] += _interpolate_drift(td.evt1_ts[j], boundaries)


def _apply_interpolated_drift_mem(td, boundaries: list[tuple]) -> None:
    """Apply interpolated drift correction to DMA timestamps."""
    for xfer_list in (td.lhs_xfers, td.rhs_xfers, td.res_xfers, td.pres_xfers):
        for x in xfer_list:
            drift = _interpolate_drift(x.start_ts, boundaries)
            x.start_ts += drift
            x.done_ts += drift
    for lst in (td.s2mm0_done, td.s2mm1_done, td.mm2s0_done):
        for j in range(len(lst)):
            lst[j] += _interpolate_drift(lst[j], boundaries)
    for pairs in (td.s2mm0_starv, td.s2mm1_starv, td.mm2s0_stall):
        for j in range(len(pairs)):
            b, e = pairs[j]
            drift = _interpolate_drift(b, boundaries)
            pairs[j] = (b + drift, e + drift)



# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DmaTransfer:
    """One DMA BD transfer window."""
    start_ts: int       # data starts flowing (starvation_E or stall_E)
    done_ts: int        # BD completion
    window_cy: int      # done - start (includes fill gaps)
    fill_gap_cy: int    # sum of short starvation/idle gaps within window
    xfer_cy: int        # actual data on wire = window - fill_gap
    data_bytes: int = 0
    bw: float = 0.0     # data_bytes / xfer_cy (B/cycle)


@dataclass
class KernelExec:
    """One kernel invocation."""
    evt0_ts: int
    evt1_ts: int
    duration: int


@dataclass
class TileData:
    """Raw parsed events for one compute tile (core + mem trace pair)."""
    tile_name: str       # e.g. "tile(2, 0)"
    core_pid: int
    mem_pid: int
    # Core trace
    evt0_ts: list = field(default_factory=list)   # kernel start timestamps
    evt1_ts: list = field(default_factory=list)   # kernel end timestamps
    # Mem trace: S2MM ch0 (LHS input)
    s2mm0_starv: list = field(default_factory=list)   # [(B_ts, E_ts), ...]
    s2mm0_done: list = field(default_factory=list)    # [ts, ...]
    # Mem trace: S2MM ch1 (RHS input)
    s2mm1_starv: list = field(default_factory=list)
    s2mm1_done: list = field(default_factory=list)
    # Mem trace: MM2S ch0 (RES output)
    mm2s0_stall: list = field(default_factory=list)   # [(B_ts, E_ts), ...]
    mm2s0_done: list = field(default_factory=list)
    # Detected transfers
    lhs_xfers: list = field(default_factory=list)
    rhs_xfers: list = field(default_factory=list)
    res_xfers: list = field(default_factory=list)
    pres_xfers: list = field(default_factory=list)
    kernels: list = field(default_factory=list)


@dataclass
class DispatchTemplate:
    """Expected event counts per dispatch, derived from tc.json."""
    iters_per_dispatch: int   # tpOrder[0] temporal value (= ipd)
    kern_per_dispatch: int    # ipd (kernel runs on each data arrival)
    lhs_per_dispatch: int     # ipd or 1 (1 when N-inner, reuse)
    rhs_per_dispatch: int     # ipd or 1 (1 when M-inner, reuse)
    res_per_dispatch: int     # ipd or 1 (1 when K-inner, accumulate)
    inner_axis: int           # tpOrder[0]: 0=M, 1=N, 2=K
    needs_pres: bool = False  # True when PRES active (TPk>1 and inner!=K)
    pres_per_dispatch: int = 0  # ipd when PRES, 0 otherwise


@dataclass
class TileValidation:
    """Actual vs expected event counts for one tile."""
    tile_name: str
    actual_kern: int
    actual_lhs: int
    actual_rhs: int
    actual_res: int
    actual_pres: int = 0
    valid_dispatches: int = 0  # min of (count // per_dispatch) across all events
    has_data: bool = False     # any kernels at all


def build_dispatch_template(tc: dict) -> DispatchTemplate:
    """Build expected event counts per dispatch from tiling config.

    DMA schedule per tpOrder[0] (innermost axis in NPU CoreOp SCF ForOp):
      M-inner(0): kern=ipd, lhs=ipd, rhs=1(reuse), res=ipd
      N-inner(1): kern=ipd, lhs=1(reuse), rhs=ipd, res=ipd
      K-inner(2): kern=ipd, lhs=ipd, rhs=ipd, res=1(accumulate)

    PRES (partial result accumulation) is active when TPk>1 and inner!=K.
    PRES shares the S2MM channel with LHS (M-inner) or RHS (N-inner).
    After split_pres_from_channel(), lhs/rhs counts reflect pure transfers.
    """
    lvl = tc["levels"][0]
    inner_axis = lvl["tpOrder"][0]
    tp_map = {0: lvl["TPm"], 1: lvl["TPn"], 2: lvl["TPk"]}
    ipd = tp_map[inner_axis]
    needs_pres = (lvl["TPk"] > 1) and (inner_axis != 2)

    if inner_axis == 0:    # M-inner
        t = DispatchTemplate(ipd, ipd, ipd, 1, ipd, inner_axis)
    elif inner_axis == 1:  # N-inner
        t = DispatchTemplate(ipd, ipd, 1, ipd, ipd, inner_axis)
    else:                  # K-inner
        t = DispatchTemplate(ipd, ipd, ipd, ipd, 1, inner_axis)

    t.needs_pres = needs_pres
    t.pres_per_dispatch = ipd if needs_pres else 0
    return t


def validate_tiles(tiles: list[TileData], template: DispatchTemplate,
                   n_dispatches: int) -> list[TileValidation]:
    """Compare actual event counts against expected template.

    Returns validation info for each tile. valid_dispatches is the maximum
    number of complete dispatches derivable from the available events.
    """
    validations = []
    for td in tiles:
        ak = len(td.kernels)
        al = len(td.lhs_xfers)
        ar = len(td.rhs_xfers)
        ares = len(td.res_xfers)
        apres = len(td.pres_xfers)

        # Compute valid dispatches: min of (actual // per_dispatch) for
        # each event type, capped at n_dispatches.
        counts = [ak // template.kern_per_dispatch]
        if template.lhs_per_dispatch > 0:
            counts.append(al // template.lhs_per_dispatch)
        if template.rhs_per_dispatch > 0:
            counts.append(ar // template.rhs_per_dispatch)
        if template.res_per_dispatch > 0:
            counts.append(ares // template.res_per_dispatch)
        if template.pres_per_dispatch > 0:
            counts.append(apres // template.pres_per_dispatch)
        valid_d = min(counts) if counts else 0
        valid_d = min(valid_d, n_dispatches)

        validations.append(TileValidation(
            tile_name=td.tile_name,
            actual_kern=ak,
            actual_lhs=al,
            actual_rhs=ar,
            actual_res=ares,
            actual_pres=apres,
            valid_dispatches=valid_d,
            has_data=(ak > 0),
        ))

    return validations


def compute_dispatch_dur_indexed(
    td: TileData, d: int, template: DispatchTemplate
) -> tuple[int, int, bool]:
    """Compute dispatch duration using index-based event matching.

    Returns (t_start, t_end, is_valid).

    DMA order determines which event starts the dispatch:
      M-inner: RHS(reuse,1x) first → t_start = rhs[d].start_ts
      N-inner: LHS(reuse,1x) first → t_start = lhs[d].start_ts
      K-inner: LHS first in loop   → t_start = lhs[d*ipd].start_ts

    End of dispatch:
      M/N-inner: last RES done → res[d*ipd + ipd - 1].done_ts
      K-inner:   single RES   → res[d].done_ts
    """
    ipd = template.iters_per_dispatch
    kbase = d * template.kern_per_dispatch

    # Check kernel bounds
    if kbase + template.kern_per_dispatch > len(td.kernels):
        return (0, 0, False)

    # Compute event indices for this dispatch
    lhs_base = d * template.lhs_per_dispatch
    rhs_base = d * template.rhs_per_dispatch
    res_base = d * template.res_per_dispatch

    # Bounds check for all event types — fallback to kernel timestamps
    # when DMA events are missing. Kernel bounds already validated above.
    dma_missing = (
        lhs_base + template.lhs_per_dispatch > len(td.lhs_xfers)
        or rhs_base + template.rhs_per_dispatch > len(td.rhs_xfers)
        or res_base + template.res_per_dispatch > len(td.res_xfers)
    )
    if dma_missing:
        k_first = td.kernels[kbase]
        k_last = td.kernels[kbase + template.kern_per_dispatch - 1]
        return (k_first.evt0_ts, k_last.evt1_ts, True)

    # Use kernel start → kernel end as dispatch boundaries.
    #
    # DMA indices are unreliable after stale-drop: init BD removal shifts
    # reuse-axis DMA indices by 1. Using kernel timestamps avoids this
    # entirely. The dispatch duration measures compute + DMA overlap
    # within the kernel window, which is the tile-local execution time.
    # Host overhead (inter-dispatch latency) is captured by host_steps.
    k_first = td.kernels[kbase]
    k_last = td.kernels[kbase + template.kern_per_dispatch - 1]
    t_start = k_first.evt0_ts
    t_end = k_last.evt1_ts

    return (t_start, t_end, True)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_trace_json(path: str) -> list[TileData]:
    """Parse trace.json into per-tile event structures.

    Returns list of TileData sorted by tile row (ascending = closest to shim first).
    Pairs core_trace and mem_trace pids by matching tile coordinates.
    """
    with open(path) as f:
        raw = json.load(f)
    events = raw.get("traceEvents", raw) if isinstance(raw, dict) else raw

    # First pass: discover pid → (trace_type, tile_name) from metadata
    pid_info: dict[int, tuple[str, str]] = {}  # pid → (type, tile_coord)
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            pname = ev["args"].get("name", "")
            pid = ev["pid"]
            if "core_trace" in pname:
                coord = pname.replace("core_trace for ", "")
                pid_info[pid] = ("core", coord)
            elif "mem_trace" in pname:
                coord = pname.replace("mem_trace for ", "")
                pid_info[pid] = ("mem", coord)

    # Group by tile coordinate
    tile_map: dict[str, dict] = {}  # coord → {"core_pid", "mem_pid"}
    for pid, (ttype, coord) in pid_info.items():
        if coord not in tile_map:
            tile_map[coord] = {}
        tile_map[coord][f"{ttype}_pid"] = pid

    # Create TileData for each paired tile
    tiles: list[TileData] = []
    for coord, pids in tile_map.items():
        if "core_pid" in pids and "mem_pid" in pids:
            tiles.append(TileData(
                tile_name=coord,
                core_pid=pids["core_pid"],
                mem_pid=pids["mem_pid"],
            ))

    # Sort by tile row ascending (closest to shim first).
    # Supports both "tile(2, 0)" and "tile2,0" formats.
    import re
    def row_key(td):
        m = re.search(r"tile\(?(\d+)", td.tile_name)
        return int(m.group(1)) if m else 0
    tiles.sort(key=row_key)

    # Build lookup for fast event routing
    core_pids = {td.core_pid: td for td in tiles}
    mem_pids = {td.mem_pid: td for td in tiles}

    # Pending B timestamps for level events
    pending_starv0: dict[int, int] = {}  # mem_pid → B_ts
    pending_starv1: dict[int, int] = {}
    pending_stall0: dict[int, int] = {}

    # Second pass: collect events
    for ev in events:
        if ev.get("ph") == "M":
            continue
        pid = ev.get("pid")
        name = ev.get("name", "")
        ph = ev.get("ph", "")
        ts = ev.get("ts", 0)

        # Core trace events
        if pid in core_pids:
            td = core_pids[pid]
            if name == "INSTR_EVENT_0" and ph == "B":
                td.evt0_ts.append(ts)
            elif name == "INSTR_EVENT_1" and ph == "B":
                td.evt1_ts.append(ts)

        # Memory module trace events
        elif pid in mem_pids:
            td = mem_pids[pid]

            if "S2MM_0_STREAM_STARVATION" in name:
                if ph == "B":
                    pending_starv0[pid] = ts
                elif ph == "E" and pid in pending_starv0:
                    td.s2mm0_starv.append((pending_starv0[pid], ts))
                    del pending_starv0[pid]
            elif "S2MM_0_FINISHED_BD" in name and ph == "B":
                td.s2mm0_done.append(ts)

            elif "S2MM_1_STREAM_STARVATION" in name:
                if ph == "B":
                    pending_starv1[pid] = ts
                elif ph == "E" and pid in pending_starv1:
                    td.s2mm1_starv.append((pending_starv1[pid], ts))
                    del pending_starv1[pid]
            elif "S2MM_1_FINISHED_BD" in name and ph == "B":
                td.s2mm1_done.append(ts)

            elif "MM2S_0_STALLED_LOCK" in name:
                if ph == "B":
                    pending_stall0[pid] = ts
                elif ph == "E" and pid in pending_stall0:
                    td.mm2s0_stall.append((pending_stall0[pid], ts))
                    del pending_stall0[pid]
            elif "MM2S_0_FINISHED_BD" in name and ph == "B":
                td.mm2s0_done.append(ts)

    # Sort all event lists by timestamp
    for td in tiles:
        td.evt0_ts.sort()
        td.evt1_ts.sort()
        td.s2mm0_starv.sort()
        td.s2mm0_done.sort()
        td.s2mm1_starv.sort()
        td.s2mm1_done.sort()
        td.mm2s0_stall.sort()
        td.mm2s0_done.sort()

    return tiles


def load_tc_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Transfer detection
# ---------------------------------------------------------------------------

def detect_s2mm_transfers(starv_pairs: list[tuple[int, int]],
                          done_ts_list: list[int],
                          data_bytes: int) -> list[DmaTransfer]:
    """Detect S2MM (input) transfer windows from starvation pairs and DONE events.

    Transfer model:
      - A long starvation gap (>LONG_GAP_THRESHOLD) = idle between iterations
      - When the long starvation ends (E event) = first data byte arrives
      - Short B-E bursts after that = fill-latency gaps (pipeline warming)
      - FINISHED_BD = all data received
      - actual_xfer = (DONE - first_data) - sum(short_starvation_gaps)
    """
    transfers = []
    for done_ts in done_ts_list:
        # Find the last long starvation E before this DONE → first_data
        first_data = None
        for sb, se in starv_pairs:
            dur = se - sb
            if se < done_ts and dur > LONG_GAP_THRESHOLD:
                if first_data is None or se > first_data:
                    first_data = se

        if first_data is None:
            continue

        window = done_ts - first_data

        # Sum short starvation gaps fully within [first_data, done_ts]
        fill_gap = 0
        for sb, se in starv_pairs:
            if sb > first_data and se <= done_ts and (se - sb) < LONG_GAP_THRESHOLD:
                fill_gap += se - sb

        xfer = window - fill_gap
        bw = data_bytes / xfer if xfer > 0 else 0.0
        transfers.append(DmaTransfer(first_data, done_ts, window, fill_gap, xfer,
                                     data_bytes, bw))
    return transfers


def detect_mm2s_transfers(stall_pairs: list[tuple[int, int]],
                          done_ts_list: list[int],
                          data_bytes: int) -> list[DmaTransfer]:
    """Detect MM2S (output) transfer windows from stall pairs and DONE events.

    Transfer model:
      - STALL_B..STALL_E = waiting for lock (kernel producing data)
      - STALL_E = lock acquired → DMA starts transferring
      - DONE = BD complete → transfer finished
      - total_time = DONE - STALL_E (includes network contention for shared port)
    """
    transfers = []
    for done_ts in done_ts_list:
        # Find the STALL_E closest to (but before) this DONE
        stall_e = None
        for sb, se in stall_pairs:
            if se < done_ts:
                if stall_e is None or se > stall_e:
                    stall_e = se
        if stall_e is None:
            continue

        window = done_ts - stall_e
        bw = data_bytes / window if window > 0 else 0.0
        transfers.append(DmaTransfer(stall_e, done_ts, window, 0, window,
                                     data_bytes, bw))
    return transfers


def compute_data_sizes(tc: dict) -> dict:
    """Compute per-BD data sizes from tiling config."""
    lvl = tc["levels"][0]
    elem_bytes = 2 if tc["elemType"] == "bf16" else 4
    return {
        "lhs": lvl["TM"] * lvl["TK"] * elem_bytes,
        "rhs": lvl["TN"] * lvl["TK"] * elem_bytes,
        "res": lvl["TM"] * lvl["TN"] * elem_bytes,
        "pres": lvl["TM"] * lvl["TN"] * elem_bytes,
        "elem_bytes": elem_bytes,
    }


def _pair_kernel_events(
    evt0_ts: list[int], evt1_ts: list[int]
) -> list[KernelExec]:
    """Pair EVENT_0 (start) with the nearest following EVENT_1 (end).

    Both lists must be sorted ascending.  For each start timestamp, we
    find the first end timestamp that is >= the start using bisect.
    This handles stale events at the beginning, duplicate events
    mid-stream, and dropped events without mis-pairing.
    """
    import bisect
    kernels = []
    consumed = 0  # tracks how far into evt1_ts we've consumed
    for start in evt0_ts:
        # Find the first evt1 >= start that hasn't been consumed yet
        pos = bisect.bisect_left(evt1_ts, start, lo=consumed)
        if pos >= len(evt1_ts):
            break
        end = evt1_ts[pos]
        kernels.append(KernelExec(start, end, end - start))
        consumed = pos + 1
    return kernels


def split_pres_from_channel(
    done_ts_list: list[int],
    starv_pairs: list[tuple[int, int]],
    data_bytes: int,
    pres_bytes: int,
    first_kernel_ts: int = 0,
) -> tuple[list[DmaTransfer], list[DmaTransfer]]:
    """Split interleaved LHS+PRES (or RHS+PRES) DONE events on a shared S2MM channel.

    The Pass DMA schedule (assembleScheduleByAxis) emits BDs in strict order
    per iteration:
      M-inner: [LHS_BD, PRES_BD, RES_BD] x TPm  (all on S2MM ch0)
      N-inner: [RHS_BD, PRES_BD, RES_BD] x TPn  (all on S2MM ch1)

    So S2MM DONE events alternate: data(0), pres(1), data(2), pres(3), ...

    IMPORTANT: stale init BD events must be removed BEFORE even/odd split.
    The NPU DMA engine executes one init BD before computation starts,
    producing a single stale DONE event at index 0.  If not removed, it
    shifts all subsequent even/odd assignments.

    Stale detection: total expected events = 2 * ipd * dispatches (even count).
    If len(done_ts_list) is odd, the extra event at the front is stale.
    We also drop any DONE events before first_kernel_ts as a safety check.
    """
    # Remove stale init BD: if total DONE count is odd, drop the first event.
    # With PRES, each dispatch produces 2*ipd DONE events (data + pres),
    # so the total should always be even.  An odd count means one stale init BD.
    filtered = done_ts_list
    if len(done_ts_list) % 2 == 1:
        filtered = done_ts_list[1:]

    data_done = [filtered[i] for i in range(len(filtered)) if i % 2 == 0]
    pres_done = [filtered[i] for i in range(len(filtered)) if i % 2 == 1]

    data_xfers = detect_s2mm_transfers(starv_pairs, data_done, data_bytes)
    pres_xfers = detect_s2mm_transfers(starv_pairs, pres_done, pres_bytes)
    return data_xfers, pres_xfers


def detect_all_transfers(tiles: list[TileData], tc: dict):
    """Detect DMA transfers for all tiles and build kernel lists.

    When PRES is active (TPk>1 and inner!=K), the PRES BD shares a S2MM
    channel with LHS (M-inner) or RHS (N-inner).  We split the channel's
    DONE events by even/odd index to separate data vs PRES transfers.
    """
    sizes = compute_data_sizes(tc)
    lvl = tc["levels"][0]
    inner_axis = lvl["tpOrder"][0]
    needs_pres = (lvl["TPk"] > 1) and (inner_axis != 2)

    for td in tiles:
        # Kernels: pair each EVENT_0 with the next EVENT_1 by timestamp.
        td.kernels = _pair_kernel_events(td.evt0_ts, td.evt1_ts)

        if needs_pres and inner_axis == 0:
            # M-inner: S2MM ch0 has interleaved LHS + PRES BDs
            td.lhs_xfers, td.pres_xfers = split_pres_from_channel(
                td.s2mm0_done, td.s2mm0_starv,
                sizes["lhs"], sizes["pres"])
            td.rhs_xfers = detect_s2mm_transfers(
                td.s2mm1_starv, td.s2mm1_done, sizes["rhs"])
        elif needs_pres and inner_axis == 1:
            # N-inner: S2MM ch1 has interleaved RHS + PRES BDs
            td.lhs_xfers = detect_s2mm_transfers(
                td.s2mm0_starv, td.s2mm0_done, sizes["lhs"])
            td.rhs_xfers, td.pres_xfers = split_pres_from_channel(
                td.s2mm1_done, td.s2mm1_starv,
                sizes["rhs"], sizes["pres"])
        else:
            # No PRES or K-inner (needsPres=false when inner==K)
            td.lhs_xfers = detect_s2mm_transfers(
                td.s2mm0_starv, td.s2mm0_done, sizes["lhs"])
            td.rhs_xfers = detect_s2mm_transfers(
                td.s2mm1_starv, td.s2mm1_done, sizes["rhs"])

        # MM2S_0 → RES output
        td.res_xfers = detect_mm2s_transfers(
            td.mm2s0_stall, td.mm2s0_done, sizes["res"])

        # Drop stale DMA transfers from initialization.  NPU DMA engines
        # execute a BD during setup before real computation starts,
        # producing exactly one extra event at the front of each list.
        if td.kernels:
            first_k_ts = td.kernels[0].evt0_ts
            if td.res_xfers and td.res_xfers[0].done_ts < first_k_ts:
                td.res_xfers = td.res_xfers[1:]
            if (td.lhs_xfers and td.lhs_xfers[0].done_ts < first_k_ts
                    and len(td.lhs_xfers) > len(td.kernels)):
                td.lhs_xfers = td.lhs_xfers[1:]
            if (td.rhs_xfers and td.rhs_xfers[0].done_ts < first_k_ts
                    and len(td.rhs_xfers) > len(td.kernels)):
                td.rhs_xfers = td.rhs_xfers[1:]
            if (td.pres_xfers and td.pres_xfers[0].done_ts < first_k_ts
                    and len(td.pres_xfers) > len(td.kernels)):
                td.pres_xfers = td.pres_xfers[1:]


def compute_combined_res(tiles: list[TileData]) -> list[DmaTransfer]:
    """Compute combined RES transfers from shim perspective.

    Multiple comp tiles share one shim S2MM port for RES output.
    Combined window: earliest STALL_E → latest DONE across all tiles
    for each iteration.  Total bytes = sum of all tiles' data.
    """
    if not tiles:
        return []

    # Align by iteration index: use the minimum count across tiles
    n_iters = min(len(td.res_xfers) for td in tiles)
    combined = []
    for i in range(n_iters):
        starts = [td.res_xfers[i].start_ts for td in tiles]
        ends = [td.res_xfers[i].done_ts for td in tiles]
        g_start = min(starts)
        g_end = max(ends)
        window = g_end - g_start
        total_bytes = sum(td.res_xfers[i].data_bytes for td in tiles)
        bw = total_bytes / window if window > 0 else 0.0
        combined.append(DmaTransfer(g_start, g_end, window, 0, window,
                                    total_bytes, bw))
    return combined


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_cy(c) -> str:
    return f"{c:,.0f}"


def fmt_us(cycles, clock_mhz) -> str:
    return f"{cycles / clock_mhz:,.1f}"


def fmt_bw(bw) -> str:
    return f"{bw:.2f}"


def print_header(text: str):
    print(f"\n{'=' * 72}")
    print(f"  {text}")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_config(tc: dict, tiles: list[TileData], n_dispatches: int,
                  iters_per_dispatch: int, clock_mhz: float):
    """Print configuration summary."""
    print_header("Configuration")
    lvl = tc["levels"][0]
    inner = lvl["tpOrder"][0]
    sizes = compute_data_sizes(tc)
    print(f"  Matrix:    {tc['M']}x{tc['K']}x{tc['N']} {tc['elemType']}")
    print(f"  Spatial:   SPm={lvl['SPm']} SPn={lvl['SPn']} "
          f"({tc['numCores']} comp tiles)")
    print(f"  Temporal:  TPm={lvl['TPm']} TPk={lvl['TPk']} TPn={lvl['TPn']}")
    print(f"  Tile size: TM={lvl['TM']} TK={lvl['TK']} TN={lvl['TN']}")
    print(f"  tpOrder:   {lvl['tpOrder']} "
          f"(inner={'MNK'[inner]})")
    print(f"  Dispatches: {n_dispatches}  "
          f"Iters/dispatch: {iters_per_dispatch}")
    print(f"  Clock:     {clock_mhz} MHz")
    print(f"  Data sizes:  LHS={sizes['lhs']:,}B  "
          f"RHS={sizes['rhs']:,}B  RES={sizes['res']:,}B")
    print(f"  Tiles captured: {len(tiles)} "
          f"({', '.join(td.tile_name for td in tiles)})")


def report_dma_summary(tiles: list[TileData], combined_res: list[DmaTransfer],
                       clock_mhz: float):
    """Print DMA transfer summary: per-tile for input, combined for output."""
    print_header("DMA Transfer Summary")

    # Input transfers (LHS/RHS): identical across tiles, show one representative
    ref = tiles[0]
    print(f"\n  [Input Transfers (per tile, all tiles identical)]")
    print(f"  {'Channel':<10} {'Buffer':<6} {'Count':>6} "
          f"{'Avg Window':>12} {'Avg Fill':>10} {'Avg Xfer':>10} "
          f"{'Bytes':>8} {'Avg BW':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*6} "
          f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for label, xfers, ch_name in [
        ("LHS", ref.lhs_xfers, "S2MM_0"),
        ("RHS", ref.rhs_xfers, "S2MM_1"),
    ]:
        if not xfers:
            print(f"  {ch_name:<10} {label:<6} {'0':>6}")
            continue
        avg_w = sum(x.window_cy for x in xfers) / len(xfers)
        avg_f = sum(x.fill_gap_cy for x in xfers) / len(xfers)
        avg_x = sum(x.xfer_cy for x in xfers) / len(xfers)
        avg_bw = sum(x.bw for x in xfers) / len(xfers)
        data_b = xfers[0].data_bytes
        print(f"  {ch_name:<10} {label:<6} {len(xfers):>6} "
              f"{fmt_cy(avg_w):>12} {fmt_cy(avg_f):>10} {fmt_cy(avg_x):>10} "
              f"{data_b:>8,} {fmt_bw(avg_bw):>8} B/cy")

    # RES output: per-tile (shows contention gradient) + combined (shim view)
    print(f"\n  [Output Transfers — RES per tile (MM2S_0, shows port contention)]")
    print(f"  {'Tile':<14} {'Count':>6} {'Avg Xfer':>10} "
          f"{'Bytes':>8} {'Avg BW':>10} {'Note'}")
    print(f"  {'-'*14} {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*20}")

    for td in tiles:
        xfers = td.res_xfers
        if not xfers:
            print(f"  {td.tile_name:<14} {'0':>6}")
            continue
        avg_x = sum(x.xfer_cy for x in xfers) / len(xfers)
        avg_bw = sum(x.bw for x in xfers) / len(xfers)
        data_b = xfers[0].data_bytes
        # Determine if this tile has minimal contention
        note = "closest to shim" if td == tiles[0] else ""
        print(f"  {td.tile_name:<14} {len(xfers):>6} {fmt_cy(avg_x):>10} "
              f"{data_b:>8,} {fmt_bw(avg_bw):>8} B/cy  {note}")

    # Combined RES (shim perspective)
    if combined_res:
        avg_w = sum(x.window_cy for x in combined_res) / len(combined_res)
        avg_bw = sum(x.bw for x in combined_res) / len(combined_res)
        total_b = combined_res[0].data_bytes
        print(f"\n  [Combined RES (shim perspective: {len(tiles)} tiles → 1 port)]")
        print(f"  Count: {len(combined_res)}  "
              f"Avg window: {fmt_cy(avg_w)} cy  "
              f"Total bytes: {total_b:,}  "
              f"Avg BW: {fmt_bw(avg_bw)} B/cy")


def report_iteration_detail(tiles: list[TileData],
                            combined_res: list[DmaTransfer],
                            iters_per_dispatch: int,
                            n_dispatches: int, clock_mhz: float):
    """Print per-iteration detail (dispatch 0).

    LHS/Kernel from reference tile; RES from combined shim perspective.
    """
    print_header("Iteration Detail (dispatch 0)")

    if not tiles:
        return
    td = tiles[0]  # reference tile for LHS/kernel
    n_kern = min(len(td.kernels), iters_per_dispatch)

    if n_kern == 0:
        print("  No kernel iterations captured.")
        return

    print(f"\n  LHS/Kernel: {td.tile_name}  |  "
          f"RES: combined ({len(tiles)} tiles → shim)")
    print(f"  {'Iter':>4}  {'LHS start':>12} {'LHS end':>12} {'LHS xfer':>10} "
          f"{'Kern start':>12} {'Kern end':>12} {'Kern cy':>10} "
          f"{'RES start':>12} {'RES end':>12} {'RES cy':>10} {'RES BW':>10}")
    print(f"  {'----':>4}  {'-'*12} {'-'*12} {'-'*10} "
          f"{'-'*12} {'-'*12} {'-'*10} "
          f"{'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for i in range(n_kern):
        k = td.kernels[i]

        # Index-based matching: kernel i → LHS[i], combined_res[i]
        lhs = td.lhs_xfers[i] if i < len(td.lhs_xfers) else None
        res = combined_res[i] if combined_res and i < len(combined_res) else None

        lhs_s = fmt_cy(lhs.start_ts) if lhs else "-"
        lhs_e = fmt_cy(lhs.done_ts) if lhs else "-"
        lhs_x = fmt_cy(lhs.xfer_cy) if lhs else "-"
        res_s = fmt_cy(res.start_ts) if res else "-"
        res_e = fmt_cy(res.done_ts) if res else "-"
        res_x = fmt_cy(res.xfer_cy) if res else "-"
        res_bw = fmt_bw(res.bw) if res else "-"

        print(f"  {i:>4}  {lhs_s:>12} {lhs_e:>12} {lhs_x:>10} "
              f"{fmt_cy(k.evt0_ts):>12} {fmt_cy(k.evt1_ts):>12} {fmt_cy(k.duration):>10} "
              f"{res_s:>12} {res_e:>12} {res_x:>10} {res_bw:>8} B/cy")

    # RHS transfers (fixed per dispatch, fewer events)
    if td.rhs_xfers:
        rhs = td.rhs_xfers[0]
        print(f"\n  RHS (fixed, dispatch 0): start={fmt_cy(rhs.start_ts)} "
              f"done={fmt_cy(rhs.done_ts)} xfer={fmt_cy(rhs.xfer_cy)} cy "
              f"({fmt_bw(rhs.bw)} B/cy)")


def report_dispatch_breakdown(tiles: list[TileData], tc: dict,
                              iters_per_dispatch: int, n_dispatches: int,
                              clock_mhz: float, template: DispatchTemplate = None,
                              tile_validations: list[TileValidation] = None):
    """Print dispatch-level timing breakdown using index-based matching."""
    print_header("Dispatch Breakdown (averaged over active tiles and dispatches)")

    if not tiles:
        return

    # Build template if not provided (backward compat)
    if template is None:
        template = build_dispatch_template(tc)
    if tile_validations is None:
        tile_validations = validate_tiles(tiles, template, n_dispatches)

    # Collect per-dispatch combined time (all active tiles → global start/end)
    combined_dispatch_times = []
    all_kernel_totals = []

    # Determine valid dispatch range across all active tiles
    active_pairs = [(td, tv) for td, tv in zip(tiles, tile_validations)
                    if tv.has_data and tv.valid_dispatches > 0]
    if not active_pairs:
        print("  No active tiles with valid dispatches.")
        return

    n_valid = min(tv.valid_dispatches for _, tv in active_pairs)
    for d in range(n_valid):
        starts = []
        ends = []
        for td, _ in active_pairs:
            t_start, t_end, valid = compute_dispatch_dur_indexed(td, d, template)
            if valid:
                starts.append(t_start)
                ends.append(t_end)
        if starts and ends:
            combined_dispatch_times.append(max(ends) - min(starts))

        # Kernel totals (reference tile)
        ref_td = active_pairs[0][0]
        kbase = d * template.kern_per_dispatch
        if kbase + template.kern_per_dispatch <= len(ref_td.kernels):
            all_kernel_totals.append(sum(
                ref_td.kernels[kbase + p].duration
                for p in range(template.kern_per_dispatch)
            ))

    if not combined_dispatch_times:
        print("  Insufficient data.")
        return

    avg_dispatch = sum(combined_dispatch_times) / len(combined_dispatch_times)
    avg_kernel = sum(all_kernel_totals) / len(all_kernel_totals) if all_kernel_totals else 0
    avg_non_kernel = avg_dispatch - avg_kernel

    # Kernel time breakdown
    ref = tiles[0]
    avg_kern_single = (sum(k.duration for k in ref.kernels[:iters_per_dispatch])
                       / iters_per_dispatch) if ref.kernels else 0

    print(f"\n  [Time Breakdown (per dispatch)]")
    print(f"    Dispatch total:  {fmt_cy(avg_dispatch):>12} cy "
          f"({fmt_us(avg_dispatch, clock_mhz):>8} us)  100.0%")
    print(f"    Kernel total:    {fmt_cy(avg_kernel):>12} cy "
          f"({fmt_us(avg_kernel, clock_mhz):>8} us)  "
          f"{avg_kernel / avg_dispatch * 100:5.1f}%  "
          f"({iters_per_dispatch} x {fmt_cy(avg_kern_single)} cy)")
    print(f"    DMA + overhead:  {fmt_cy(avg_non_kernel):>12} cy "
          f"({fmt_us(avg_non_kernel, clock_mhz):>8} us)  "
          f"{avg_non_kernel / avg_dispatch * 100:5.1f}%")

    # Effective metrics
    M, K, N = tc['M'], tc['K'], tc['N']
    elem_bytes = 2 if tc['elemType'] == 'bf16' else 4
    flops = 2 * M * K * N
    data_bytes = elem_bytes * (M * K + K * N + M * N)
    dispatch_s = avg_dispatch / clock_mhz / 1e6

    if dispatch_s > 0:
        gflops = flops / dispatch_s / 1e9
        gbps = data_bytes / dispatch_s / 1e9
        # Peak: bf16 mmul 4x8x4 = 256 ops/cycle per tile
        peak_ops = 4 * 8 * 4 * 2
        n_active = len(active_pairs)
        peak_gflops = n_active * peak_ops * clock_mhz * 1e6 / 1e9

        print(f"\n  [Effective Metrics ({n_active} active tiles, "
              f"{n_valid} valid dispatches, "
              f"over dispatch wall-clock {fmt_us(avg_dispatch, clock_mhz)} us)]")
        print(f"    Throughput:  {gflops:>8.1f} GFLOPS  "
              f"(peak: {peak_gflops:.0f} GFLOPS, "
              f"{gflops / peak_gflops * 100:.1f}%)")
        print(f"    Bandwidth:   {gbps:>8.2f} GB/s  "
              f"(LHS+RHS+RES = {data_bytes:,} B)")


def report_cross_tile_res(tiles: list[TileData], clock_mhz: float):
    """Analyze RES output serialization across tiles sharing a shim port."""
    print_header("Cross-Tile RES Output Analysis (shared shim port)")

    if len(tiles) < 2:
        print("  Only 1 tile — skipping.")
        return

    # Compare first iteration's RES transfer across tiles
    print(f"\n  [RES Transfer Per Tile (first 5 iterations)]")
    print(f"  {'Tile':<14} {'Iter':>4} {'STALL_E':>12} {'DONE':>12} "
          f"{'Total cy':>10} {'BW (B/cy)':>10}")
    print(f"  {'-'*14} {'-'*4} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for td in tiles:
        for i, x in enumerate(td.res_xfers[:5]):
            print(f"  {td.tile_name:<14} {i:>4} {fmt_cy(x.start_ts):>12} "
                  f"{fmt_cy(x.done_ts):>12} {fmt_cy(x.xfer_cy):>10} "
                  f"{fmt_bw(x.bw):>10}")

    # Cross-tile RES serialization for first few iterations
    max_iters = min(5, min(len(td.res_xfers) for td in tiles) if tiles else 0)
    if max_iters == 0:
        return

    print(f"\n  [Combined RES Bandwidth (all tiles per iteration)]")
    print(f"  {'Iter':>4} {'Global Start':>14} {'Global End':>14} "
          f"{'Total cy':>10} {'Total bytes':>12} {'Combined BW':>12}")
    print(f"  {'-'*4} {'-'*14} {'-'*14} {'-'*10} {'-'*12} {'-'*12}")

    for i in range(max_iters):
        starts = [td.res_xfers[i].start_ts for td in tiles]
        ends = [td.res_xfers[i].done_ts for td in tiles]
        g_start = min(starts)
        g_end = max(ends)
        total_dur = g_end - g_start
        total_bytes = sum(td.res_xfers[i].data_bytes for td in tiles)
        combined_bw = total_bytes / total_dur if total_dur > 0 else 0

        print(f"  {i:>4} {fmt_cy(g_start):>14} {fmt_cy(g_end):>14} "
              f"{fmt_cy(total_dur):>10} {total_bytes:>12,} "
              f"{fmt_bw(combined_bw):>10} B/cy")

    # Serialization order
    if tiles[0].res_xfers:
        print(f"\n  [RES Serialization Order (iter 0, sorted by DONE)]")
        order = sorted(
            [(td.tile_name, td.res_xfers[0]) for td in tiles],
            key=lambda x: x[1].done_ts
        )
        prev_done = None
        for name, x in order:
            gap = f"  gap from prev: {fmt_cy(x.done_ts - prev_done)} cy" if prev_done else ""
            print(f"    {name}: DONE={fmt_cy(x.done_ts)} "
                  f"(xfer={fmt_cy(x.xfer_cy)} cy, {fmt_bw(x.bw)} B/cy){gap}")
            prev_done = x.done_ts


def report_timeline(tiles: list[TileData], iters_per_dispatch: int,
                    n_dispatches: int, clock_mhz: float):
    """Print ASCII timeline showing kernel + DMA activity per tile."""
    if not tiles:
        return

    # ---- Full dispatch timeline ----
    print_header("Timeline: Full Dispatch (dispatch 0)")

    # Determine time range from all events in dispatch 0
    t_min, t_max = _dispatch_time_range(tiles, iters_per_dispatch, 0)
    if t_min is None:
        print("  No data.")
        return

    _render_timeline(tiles, iters_per_dispatch, 0, t_min, t_max, clock_mhz,
                     "full")

    # ---- Zoomed: first 2 iterations ----
    print_header("Timeline: Zoomed (first 2 iterations, dispatch 0)")

    # Narrow time range to first 2 kernel iterations
    z_max = t_min
    for td in tiles:
        if len(td.kernels) >= 2:
            # Index-based: iter 1 → res_xfers[1]
            res = td.res_xfers[1] if 1 < len(td.res_xfers) else None
            end = res.done_ts if res else td.kernels[1].evt1_ts
            z_max = max(z_max, end + 500)

    _render_timeline(tiles, iters_per_dispatch, 0, t_min, z_max, clock_mhz,
                     "zoom")

    # ---- Single steady-state iteration (iter 1) ----
    print_header("Timeline: Single Iteration (iter 1)")

    # Time range: from iter 0 RES done to iter 1 RES done
    s_min = None
    s_max = None
    for td in tiles:
        if len(td.kernels) >= 2:
            # Start from iter 0 RES done (end of previous iteration)
            res0 = td.res_xfers[0] if td.res_xfers else None
            prev_end = res0.done_ts if res0 else td.kernels[0].evt1_ts
            res1 = td.res_xfers[1] if 1 < len(td.res_xfers) else None
            end = res1.done_ts if res1 else td.kernels[1].evt1_ts

            s_min = min(s_min, prev_end) if s_min is not None else prev_end
            s_max = max(s_max, end + 500) if s_max is not None else end + 500

    if s_min is not None and s_max is not None:
        _render_timeline(tiles, iters_per_dispatch, 0, s_min, s_max,
                         clock_mhz, "iter1")


def _dispatch_time_range(tiles, iters_per_dispatch, disp_idx):
    """Get time range for a dispatch across all tiles."""
    t_min = None
    t_max = None
    base = disp_idx * iters_per_dispatch

    for td in tiles:
        # Earliest event: first LHS start or first kernel start
        first_lhs = td.lhs_xfers[base] if base < len(td.lhs_xfers) else None
        if first_lhs:
            t_min = min(t_min, first_lhs.start_ts) if t_min else first_lhs.start_ts

        if base < len(td.kernels):
            kstart = td.kernels[base].evt0_ts
            t_min = min(t_min, kstart) if t_min else kstart

        # Latest event: last RES done or last kernel end
        last_idx = base + iters_per_dispatch - 1
        if last_idx < len(td.kernels):
            kend = td.kernels[last_idx].evt1_ts
            t_max = max(t_max, kend) if t_max else kend
            # Index-based: last iteration → res_xfers[last_idx]
            res = td.res_xfers[last_idx] if last_idx < len(td.res_xfers) else None
            if res:
                t_max = max(t_max, res.done_ts)

    return t_min, t_max


def _render_timeline(tiles, iters_per_dispatch, disp_idx,
                     t_min, t_max, clock_mhz, mode):
    """Render ASCII timeline for the given time range."""
    WIDTH = 100
    t_range = t_max - t_min
    if t_range <= 0:
        return
    scale = t_range / WIDTH

    def to_col(t):
        return min(WIDTH - 1, max(0, int((t - t_min) / scale)))

    print(f"\n  Legend: L=LHS  R=RHS  #=Kernel  C=RES  _=idle")
    print(f"  Scale: 1 char = {fmt_cy(scale)} cy ({fmt_us(scale, clock_mhz)} us)")
    print(f"  Time range: {fmt_cy(t_range)} cy ({fmt_us(t_range, clock_mhz)} us)")
    print()

    label_width = 14
    base = disp_idx * iters_per_dispatch

    for td in tiles:
        row = ['_'] * WIDTH

        # Determine iteration range to render
        if mode == "iter1":
            iter_range = range(base + 1, min(base + 2, len(td.kernels)))
        elif mode == "zoom":
            iter_range = range(base, min(base + 2, len(td.kernels)))
        else:
            iter_range = range(base, min(base + iters_per_dispatch, len(td.kernels)))

        # Paint LHS transfers
        for x in td.lhs_xfers:
            if x.done_ts < t_min or x.start_ts > t_max:
                continue
            c0, c1 = to_col(x.start_ts), to_col(x.done_ts)
            for c in range(c0, c1 + 1):
                row[c] = 'L'

        # Paint RHS transfers
        for x in td.rhs_xfers:
            if x.done_ts < t_min or x.start_ts > t_max:
                continue
            c0, c1 = to_col(x.start_ts), to_col(x.done_ts)
            for c in range(c0, c1 + 1):
                row[c] = 'R'

        # Paint kernel execution (overwrites DMA if overlapping)
        for i in iter_range:
            k = td.kernels[i]
            c0, c1 = to_col(k.evt0_ts), to_col(k.evt1_ts)
            for c in range(c0, c1 + 1):
                row[c] = '#'

        # Paint RES transfers
        for x in td.res_xfers:
            if x.done_ts < t_min or x.start_ts > t_max:
                continue
            c0, c1 = to_col(x.start_ts), to_col(x.done_ts)
            for c in range(c0, c1 + 1):
                row[c] = 'C'

        print(f"  {td.tile_name:>{label_width}} |{''.join(row)}|")

    # Time axis
    n_ticks = 5
    tick_pos = [int(WIDTH * i / n_ticks) for i in range(n_ticks + 1)]
    axis = [' '] * WIDTH
    for tp in tick_pos:
        if tp < WIDTH:
            axis[tp] = '|'
    print(f"  {'':>{label_width}} +{''.join(axis)}+")

    label_line = [' '] * (WIDTH + 15)
    for tp in tick_pos:
        t_val = t_min + tp * scale
        label = f"{t_val / clock_mhz:.0f}"
        start = max(0, tp - len(label) // 2)
        for ci, ch in enumerate(label):
            if start + ci < len(label_line):
                label_line[start + ci] = ch
    print(f"  {'(us)':>{label_width}}  {''.join(label_line)}")


def report_summary(tiles: list[TileData], combined_res: list[DmaTransfer],
                   tc: dict, iters_per_dispatch: int,
                   n_dispatches: int, clock_mhz: float):
    """Print single-iteration average summary (steady-state).

    Uses combined RES (shim perspective) instead of per-tile RES.
    """
    print_header("Average Steady-State Iteration (iter 1+)")

    if not tiles:
        return

    # Use reference tile for kernel/LHS (all tiles identical)
    ref = tiles[0]
    kern_durs, lhs_xfers, rhs_xfers = [], [], []
    res_combined_xfers = []
    iter_totals = []

    for d in range(n_dispatches):
        for pos in range(1, iters_per_dispatch):
            idx = d * iters_per_dispatch + pos
            if idx >= len(ref.kernels):
                continue
            k = ref.kernels[idx]
            kern_durs.append(k.duration)

            # Index-based matching: kernel idx → LHS[idx], combined_res[idx]
            lhs = ref.lhs_xfers[idx] if idx < len(ref.lhs_xfers) else None
            res = combined_res[idx] if idx < len(combined_res) else None

            if lhs:
                lhs_xfers.append(lhs.xfer_cy)
            if res:
                res_combined_xfers.append(res.xfer_cy)

            # Total iteration: LHS start → combined RES done
            t_start = lhs.start_ts if lhs else k.evt0_ts
            t_end = res.done_ts if res else k.evt1_ts
            iter_totals.append(t_end - t_start)

    # RHS
    for x in ref.rhs_xfers:
        rhs_xfers.append(x.xfer_cy)

    if not kern_durs:
        print("  No steady-state iterations found.")
        return

    avg_kern = sum(kern_durs) / len(kern_durs)
    avg_lhs = sum(lhs_xfers) / len(lhs_xfers) if lhs_xfers else 0
    avg_rhs = sum(rhs_xfers) / len(rhs_xfers) if rhs_xfers else 0
    avg_res = sum(res_combined_xfers) / len(res_combined_xfers) if res_combined_xfers else 0
    avg_total = sum(iter_totals) / len(iter_totals) if iter_totals else 0

    sizes = compute_data_sizes(tc)
    res_total_bytes = sizes["res"] * len(tiles)

    print(f"\n  {'Component':<20} {'Avg Cycles':>12} {'Avg (us)':>10} "
          f"{'% of Total':>10} {'Bytes':>10} {'BW (B/cy)':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    def _row(name, cy, data_b=0):
        pct = cy / avg_total * 100 if avg_total > 0 else 0
        bw = f"{data_b / cy:.2f}" if cy > 0 and data_b > 0 else "-"
        bytes_s = f"{data_b:>10,}" if data_b else f"{'-':>10}"
        print(f"  {name:<20} {fmt_cy(cy):>12} {fmt_us(cy, clock_mhz):>10} "
              f"{pct:>9.1f}% {bytes_s} {bw:>10}")

    _row("LHS input", avg_lhs, sizes["lhs"])
    _row("RHS input", avg_rhs, sizes["rhs"])
    _row("Kernel", avg_kern)
    _row(f"RES output (x{len(tiles)})", avg_res, res_total_bytes)

    gap = avg_total - avg_kern - avg_lhs - avg_res
    _row("Lock + overhead", max(0, gap))
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}")
    _row("Total iteration", avg_total)

    if avg_total > 0:
        print(f"\n  Kernel utilization: {avg_kern / avg_total * 100:.1f}%")
        print(f"  DMA transfer:      "
              f"{(avg_lhs + avg_res) / avg_total * 100:.1f}%")

        # Combined DMA bandwidth: total bytes / total transfer cycles
        total_dma_bytes = sizes["lhs"] + sizes["rhs"] + res_total_bytes
        total_dma_cy = avg_lhs + avg_rhs + avg_res
        if total_dma_cy > 0:
            combined_bw = total_dma_bytes / total_dma_cy
            print(f"  DMA bandwidth:     {combined_bw:.2f} B/cy "
                  f"({total_dma_bytes:,} B / {fmt_cy(total_dma_cy)} cy, "
                  f"LHS+RHS+RES)")


# ---------------------------------------------------------------------------
# Machine-readable summary
# ---------------------------------------------------------------------------

def compute_summary_dict(tiles: list[TileData], combined_res: list[DmaTransfer],
                         tc: dict, iters_per_dispatch: int,
                         n_dispatches: int, clock_mhz: float,
                         template: DispatchTemplate = None,
                         tile_validations: list[TileValidation] = None,
                         hw_timers: list[int] = None) -> dict:
    """Compute machine-readable summary metrics for JSON export.

    Uses index-based dispatch matching with combined (all-tile) dispatch
    duration to avoid cross-dispatch event mismatching.

    hw_timers: absolute NPU clock values at each dispatch boundary from
    apply_timestamp_corrections(). Used to compute full dispatch cycle
    times including host overhead (XRT command submission, interrupt, etc.).
    """
    if template is None:
        template = build_dispatch_template(tc)
    if tile_validations is None:
        tile_validations = validate_tiles(tiles, template, n_dispatches)

    # Filter to active tiles with valid dispatches
    active_pairs = [(td, tv) for td, tv in zip(tiles, tile_validations)
                    if tv.has_data and tv.valid_dispatches > 0]
    n_active = len(active_pairs)
    n_valid = min(tv.valid_dispatches for _, tv in active_pairs) if active_pairs else 0

    # Two timing metrics per dispatch:
    #   dispatch_durs[d]  = kernel-only span (max(ends) - min(starts) across tiles)
    #   dispatch_starts[d] = earliest kernel start across tiles
    #
    # dispatch_cycles[d] = dispatch_starts[d+1] - dispatch_starts[d]
    #   = full dispatch cycle including DMA + kernel + sync overhead.
    #   Requires timestamp calibration via apply_timestamp_corrections().
    ref = active_pairs[0][0] if active_pairs else tiles[0]

    dispatch_durs = []     # kernel-only span per dispatch
    dispatch_starts = []   # earliest kernel start per dispatch
    kernel_totals = []
    for d in range(n_valid):
        starts = []
        ends = []
        for td, _ in active_pairs:
            t_start, t_end, valid = compute_dispatch_dur_indexed(td, d, template)
            if valid:
                starts.append(t_start)
                ends.append(t_end)
        if starts and ends:
            dispatch_durs.append(max(ends) - min(starts))
            dispatch_starts.append(min(starts))

        kbase = d * template.kern_per_dispatch
        if kbase + template.kern_per_dispatch <= len(ref.kernels):
            kernel_totals.append(sum(
                ref.kernels[kbase + p].duration
                for p in range(template.kern_per_dispatch)
            ))

    # Full dispatch cycle time from hw_timer values in Start commands.
    #
    # BROADCAST_15 resets the trace timer at each dispatch start and
    # records the pre-reset timer value in a Start command.  So
    # hw_timers[d] = elapsed NPU cycles from dispatch d-1 start to
    # dispatch d start.  This captures the FULL NPU-side dispatch period
    # including DMA idle/wait time that is invisible to accumulated
    # event deltas.
    #
    # hw_timers[0]: pre-dispatch startup time (skip)
    # hw_timers[1]: dispatch 0 cycle time
    # hw_timers[d]: dispatch d-1 cycle time
    #
    # Fallback (no raw trace): kernel-start differences from calibrated
    # timestamps.  These only measure kernel execution windows, not
    # full dispatch periods.
    if hw_timers and len(hw_timers) >= 2:
        # hw_timers[1:] = cycle times for dispatches 0, 1, ..., N-2
        dispatch_cycles = hw_timers[1:]
    else:
        dispatch_cycles = [dispatch_starts[d + 1] - dispatch_starts[d]
                           for d in range(len(dispatch_starts) - 1)]

    avg_kernel_span = (sum(dispatch_durs) / len(dispatch_durs)
                       if dispatch_durs else 0)
    avg_kernel = sum(kernel_totals) / len(kernel_totals) if kernel_totals else 0
    avg_dispatch_cycle = (sum(dispatch_cycles) / len(dispatch_cycles)
                          if dispatch_cycles else avg_kernel_span)

    # Steady-state iteration / dispatch measurement.
    # Two modes:
    #   intra_dispatch (ipd>1): skip iter 0 within each dispatch (warmup),
    #     measure iter 1+ using kernel start-to-start gaps on reference tile.
    #   inter_dispatch (ipd=1): skip dispatch 0 (startup), measure
    #     dispatch 1+ using inter-dispatch cycles.
    ss_kern, ss_total = [], []
    if iters_per_dispatch == 1:
        ss_method = "inter_dispatch"
        # dispatch_cycles[0] = dispatch 0 full cycle (may include startup
        # overhead).  Use cycles[1+] for steady-state (dispatch 1+ cycles).
        # Filter outlier warmup cycles using median: early dispatches can
        # be 2-4x longer due to cold cache / DMA pipeline initialization.
        raw_ss = dispatch_cycles[1:] if len(dispatch_cycles) > 1 else dispatch_cycles
        if len(raw_ss) >= 3:
            sorted_ss = sorted(raw_ss)
            median_val = sorted_ss[len(sorted_ss) // 2]
            threshold = median_val * 1.5
            for c, val in enumerate(raw_ss):
                if val <= threshold:
                    ss_total.append(val)
                    if c + 1 < len(kernel_totals):
                        ss_kern.append(kernel_totals[c + 1])
        else:
            for c in range(len(raw_ss)):
                ss_total.append(raw_ss[c])
                if c + 1 < len(kernel_totals):
                    ss_kern.append(kernel_totals[c + 1])
    else:
        ss_method = "intra_dispatch"
        for d in range(n_valid):
            for pos in range(1, iters_per_dispatch):
                idx = d * iters_per_dispatch + pos
                prev_idx = idx - 1
                if idx >= len(ref.kernels):
                    continue
                k = ref.kernels[idx]
                k_prev = ref.kernels[prev_idx]
                ss_kern.append(k.duration)
                ss_total.append(k.evt0_ts - k_prev.evt0_ts)

    avg_ss_kern = sum(ss_kern) / len(ss_kern) if ss_kern else 0
    avg_ss_total = sum(ss_total) / len(ss_total) if ss_total else 0

    # Host temporal steps (dispatches per matmul)
    lvl = tc["levels"][0]
    tp_map = {0: lvl["TPm"], 1: lvl["TPn"], 2: lvl["TPk"]}
    host_steps = tp_map[lvl["tpOrder"][1]] * tp_map[lvl["tpOrder"][2]]

    # Matmul NPU time estimate using steady-state timing.
    #
    # Uses avg_ss_total (which skips warmup dispatches/iterations) to
    # avoid startup overhead bias.
    #
    # For intra_dispatch (ipd>1): ss_iter = kernel-to-kernel period
    #   within a dispatch.  matmul_npu = ss_iter * ipd * host_steps.
    #
    # For inter_dispatch (ipd=1): ss_iter = dispatch cycle (from
    #   hw_timer[2+]).  matmul_npu = ss_iter * host_steps.
    if avg_ss_total > 0:
        matmul_npu_cy_val = avg_ss_total * iters_per_dispatch * host_steps
    else:
        matmul_npu_cy_val = avg_dispatch_cycle * host_steps

    # Effective metrics
    M, K, N = tc['M'], tc['K'], tc['N']
    flops = 2 * M * K * N
    dispatch_cycle_s = avg_dispatch_cycle / clock_mhz / 1e6
    dispatch_cycle_us = avg_dispatch_cycle / clock_mhz
    matmul_npu_us_val = matmul_npu_cy_val / clock_mhz

    return {
        "dispatch_cy": round(avg_dispatch_cycle),
        "dispatch_us": round(dispatch_cycle_us, 2),
        "kernel_total_cy": round(avg_kernel),
        "kernel_pct": round(avg_kernel / avg_dispatch_cycle * 100, 1) if avg_dispatch_cycle > 0 else 0,
        "ss_iter_cy": round(avg_ss_total),
        "ss_iter_us": round(avg_ss_total / clock_mhz, 2),
        "ss_kernel_cy": round(avg_ss_kern),
        "ss_kernel_util_pct": round(avg_ss_kern / avg_ss_total * 100, 1) if avg_ss_total > 0 else 0,
        "ss_method": ss_method,
        "pres_active": template.needs_pres,
        "gflops": round(flops / dispatch_cycle_s / 1e9, 2) if dispatch_cycle_s > 0 else 0,
        "n_tiles": n_active,
        "n_valid_dispatches": n_valid,
        "host_steps": host_steps,
        "matmul_npu_us": round(matmul_npu_us_val, 2),
        "matmul_npu_cy": round(matmul_npu_cy_val),
        "kernel_span_cy": round(avg_kernel_span),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze NPU trace data: kernel timing + DMA transfer analysis"
    )
    parser.add_argument("--trace", required=True,
                        help="Path to trace.json (parse_trace.py output)")
    parser.add_argument("--tc", required=True,
                        help="Path to tc.json (tiling configuration)")
    parser.add_argument("--n-dispatches", type=int, default=0,
                        help="Number of dispatches to analyze. "
                             "0 = auto (n_measured * host_steps)")
    parser.add_argument("--n-measured", type=int, default=10,
                        help="Number of measured iterations in host (default: 10)")
    parser.add_argument("--clock-mhz", type=float,
                        default=XDNA2_DEFAULT_CLOCK_MHZ,
                        help=f"NPU clock MHz (default: {XDNA2_DEFAULT_CLOCK_MHZ})")
    parser.add_argument("--trace-raw", default=None,
                        help="Path to raw trace hex dump for cross-tile "
                             "timestamp calibration")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: stdout)")
    parser.add_argument("--json-summary", default=None,
                        help="Write machine-readable summary JSON to this path")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, "w")

    # Load inputs
    tc = load_tc_json(args.tc)
    tiles = parse_trace_json(args.trace)

    if not tiles:
        print("ERROR: No tile pairs (core+mem trace) found in trace.json",
              file=sys.stderr)
        sys.exit(1)

    # Determine iterations per dispatch
    lvl = tc["levels"][0]
    inner_axis = lvl["tpOrder"][0]
    tp_map = {0: lvl["TPm"], 1: lvl["TPn"], 2: lvl["TPk"]}
    iters_per_dispatch = tp_map[inner_axis]

    # Compute host_steps (dispatches per MatMul iteration)
    host_steps = tp_map[lvl["tpOrder"][1]] * tp_map[lvl["tpOrder"][2]]

    # Auto n_dispatches: capture all dispatches from measured iterations
    if args.n_dispatches <= 0:
        args.n_dispatches = args.n_measured * host_steps

    # Detect all DMA transfers
    detect_all_transfers(tiles, tc)

    # Calibrate cross-tile timestamps using Start timer_values from raw trace
    hw_timers = []
    if args.trace_raw:
        hw_timers = apply_timestamp_corrections(tiles, args.trace_raw)

    # Build dispatch template and validate tile event counts
    template = build_dispatch_template(tc)
    tile_validations = validate_tiles(tiles, template, args.n_dispatches)

    # Print validation summary
    active_tiles = [tv for tv in tile_validations if tv.has_data]
    n_active = len(active_tiles)
    n_valid_min = min(tv.valid_dispatches for tv in active_tiles) if active_tiles else 0
    pres_str = f" pres={template.pres_per_dispatch}" if template.needs_pres else ""
    print(f"\n  Dispatch template: inner={'MNK'[template.inner_axis]} "
          f"ipd={template.iters_per_dispatch} "
          f"(kern={template.kern_per_dispatch} "
          f"lhs={template.lhs_per_dispatch} "
          f"rhs={template.rhs_per_dispatch} "
          f"res={template.res_per_dispatch}{pres_str})", file=sys.stderr)
    print(f"  Active tiles: {n_active}/{len(tiles)}  "
          f"Valid dispatches: {n_valid_min}/{args.n_dispatches}",
          file=sys.stderr)
    for tv in tile_validations:
        expected_k = template.kern_per_dispatch * args.n_dispatches
        expected_l = template.lhs_per_dispatch * args.n_dispatches
        expected_r = template.rhs_per_dispatch * args.n_dispatches
        expected_res = template.res_per_dispatch * args.n_dispatches
        expected_pres = template.pres_per_dispatch * args.n_dispatches
        mismatch = (tv.actual_kern != expected_k or tv.actual_lhs != expected_l
                    or tv.actual_rhs != expected_r or tv.actual_res != expected_res
                    or (template.needs_pres and tv.actual_pres != expected_pres))
        if mismatch:
            pres_info = f" pres={tv.actual_pres}/{expected_pres}" if template.needs_pres else ""
            print(f"  WARNING: {tv.tile_name}: "
                  f"kern={tv.actual_kern}/{expected_k} "
                  f"lhs={tv.actual_lhs}/{expected_l} "
                  f"rhs={tv.actual_rhs}/{expected_r} "
                  f"res={tv.actual_res}/{expected_res}{pres_info} "
                  f"valid_d={tv.valid_dispatches}",
                  file=sys.stderr)

    # Filter to active tiles (have kernel data) for analysis.
    # Empty tiles (trace buffer overflow) would cause min(res_xfers)=0.
    active_tile_set = {tv.tile_name for tv in tile_validations if tv.has_data}
    active_tiles_data = [td for td in tiles if td.tile_name in active_tile_set]

    # Compute combined RES from shim perspective (active tiles only)
    combined_res = compute_combined_res(active_tiles_data)

    # Generate reports — use active tiles for analysis, all tiles for config info
    report_config(tc, tiles, args.n_dispatches, iters_per_dispatch,
                  args.clock_mhz)
    report_dma_summary(active_tiles_data, combined_res, args.clock_mhz)
    report_iteration_detail(active_tiles_data, combined_res,
                            iters_per_dispatch, args.n_dispatches,
                            args.clock_mhz)
    report_dispatch_breakdown(active_tiles_data, tc, iters_per_dispatch,
                              args.n_dispatches, args.clock_mhz,
                              template,
                              [tv for tv in tile_validations if tv.has_data])
    report_cross_tile_res(active_tiles_data, args.clock_mhz)
    report_timeline(active_tiles_data, iters_per_dispatch, args.n_dispatches,
                    args.clock_mhz)
    report_summary(active_tiles_data, combined_res, tc, iters_per_dispatch,
                   args.n_dispatches, args.clock_mhz)

    # Write machine-readable summary JSON if requested
    if args.json_summary:
        summary = compute_summary_dict(
            active_tiles_data, combined_res, tc,
            iters_per_dispatch, args.n_dispatches,
            args.clock_mhz, template,
            [tv for tv in tile_validations if tv.has_data],
            hw_timers)
        with open(args.json_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON written to {args.json_summary}", file=sys.stderr)

    if args.output:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
