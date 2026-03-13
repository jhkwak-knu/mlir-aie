#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cost_model.py — Exhaustive search + constraint filtering for spatio-temporal
parallelization on XDNA2 AIE arrays.

Stage 1: Enumerate all (numCores, SPm, SPn, TPm, TPk, TPn) candidates.
Stage 2: Apply hardware/software constraints to filter valid candidates.
Stage 3: Cost model evaluation (performance + energy).
Stage 4: EDP-based optimal selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ============================================================
# Paths
# ============================================================
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[1]           # test/onnx-mlir/
REPO_ROOT = THIS_FILE.parents[3]           # mlir-aie/
DATA_DIR  = ROOT_DIR / "data"
OUT_DIR   = ROOT_DIR / "out"

DEFAULT_OP_PATH  = DATA_DIR / "op_list.json"
DEFAULT_SYS_PATH = REPO_ROOT / "include" / "onnx" / "Target" / "XDNA2" / "xdna2_info.json"

# SW overhead subtracted from HW tile memory (stack, heap, reserved).
CTILE_RESERVED_BYTES = 4 * 1024

ELEM_SIZE_MAP = {
    "f16": 2, "bf16": 2, "f32": 4,
    "i8": 1, "i16": 2, "i32": 4,
    "ui8": 1, "ui16": 2, "ui32": 4,
}

# bf16 mmul<4,8,8> shape constraints on tile dimensions (aie2p).
# 2x2 expansion requires TM % (2*MMUL_R) == 0 and TN % (2*MMUL_T) == 0.
MMUL_R, MMUL_S, MMUL_T = 4, 8, 8

# Hardware coefficients for XDNA2 (Ryzen AI 9 HX 370, Strix Point, TSMC N4P).
# Sources: AMD XDNA2 spec (256 MACs/cycle BF16 per tile, 32 tiles, ~1.5 GHz),
#          LPDDR5X-7500 measured bandwidth ~80 GB/s (Chips and Cheese),
#          Horowitz 2014 scaled to N4P for energy estimates.
PEAK_MACS       = 256    # MACs/Cycle/Tile (BF16)
BANDWIDTH_BPC   = 4      # Bytes/Cycle (NoC stream bandwidth per channel)
ALPHA_CYCLES    = 20     # Cycles per temporal iteration (pipeline drain/fill)
E_MAC_PJ        = 0.2    # pJ per MAC (TSMC N4P estimate)
E_DRAM_PJ       = 40     # pJ per Byte DRAM access (LPDDR5X estimate)
P_STATIC_PJ     = 27     # pJ/Cycle per Tile (default mode, 0.04W @ 1.5 GHz)

# tpOrder axis indices (innermost temporal loop axis).
TP_ORDER_M, TP_ORDER_N, TP_ORDER_K = 0, 1, 2


# ============================================================
# Data structures
# ============================================================
@dataclass
class OpCase:
    """Single matrix multiplication specification."""
    M: int
    K: int
    N: int
    elem_type: str = "bf16"

    @property
    def elem_bytes(self) -> int:
        return ELEM_SIZE_MAP.get(self.elem_type.lower(), 4)


@dataclass
class SystemInfo:
    """Hardware parameters loaded from xdna2_info.json."""
    total_cores: int
    comp_tiles_per_col: int
    max_columns: int
    spm_size_bytes: int
    mem_tile_mem_bytes: int

    @property
    def ct_usable_bytes(self) -> int:
        return self.spm_size_bytes - CTILE_RESERVED_BYTES


@dataclass
class Candidate:
    """A single spatio-temporal parallelization configuration."""
    # Spatial
    num_cores: int
    num_columns: int
    SPm: int
    SPn: int
    # Temporal
    TPm: int
    TPk: int
    TPn: int
    # Tile sizes (derived)
    TM: int
    TK: int
    TN: int
    # Working set
    ws_bytes: int = 0

    @property
    def tp_total(self) -> int:
        return self.TPm * self.TPk * self.TPn


@dataclass
class CostResult:
    """Cost model evaluation result for a candidate + tpOrder pair."""
    candidate: Candidate
    tp_order: int           # 0=M, 1=N, 2=K (innermost temporal axis)
    # Performance (Cycles)
    t_comp: float
    t_comm: float
    t_overhead: float
    t_total: float
    # Energy (pJ)
    e_dynamic_comp: float
    e_dynamic_comm: float
    e_static: float
    e_total: float
    # EDP (pJ * Cycles)
    edp: float


# ============================================================
# I/O
# ============================================================
def load_op_list(path: Path) -> List[OpCase]:
    if not path.is_file():
        raise FileNotFoundError(f"op_list.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    cases = doc.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Invalid op_list.json: missing 'cases' array")
    return [
        OpCase(
            M=int(c["M"]), K=int(c["K"]), N=int(c["N"]),
            elem_type=str(c.get("elemType", "bf16")),
        )
        for c in cases
    ]


def load_system_info(path: Path) -> SystemInfo:
    if not path.is_file():
        raise FileNotFoundError(f"system info not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    sys_obj = doc["system"]
    device = sys_obj.get("device", {})
    spm_levels = sys_obj.get("spm_levels", [])
    if not spm_levels:
        raise ValueError("empty spm_levels")
    return SystemInfo(
        total_cores=int(sys_obj["total_cores"]),
        comp_tiles_per_col=int(device.get("comp_tiles_per_col", 4)),
        max_columns=int(device.get("max_columns", 8)),
        spm_size_bytes=int(spm_levels[0]["spm_size_bytes"]),
        mem_tile_mem_bytes=int(device.get("mem_tile_mem_bytes", 524288)),
    )


# ============================================================
# Stage 1: Exhaustive enumeration
# ============================================================
def _divisors(n: int) -> List[int]:
    """All positive divisors of n, sorted ascending."""
    ds = set()
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            ds.add(d)
            ds.add(n // d)
    return sorted(ds)


def _factor_pairs(n: int) -> List[Tuple[int, int]]:
    """All (a, b) pairs with a * b == n, sorted."""
    return [(d, n // d) for d in _divisors(n)]


def ws_bytes(TM: int, TK: int, TN: int, elem_bytes: int) -> int:
    """Working-set size for one compute tile: A(TM×TK) + B(TK×TN) + C(TM×TN)."""
    return elem_bytes * (TM * TK + TK * TN + TM * TN)


def enumerate_candidates(op: OpCase, sys_info: SystemInfo) -> List[Candidate]:
    """
    Enumerate ALL (num_cores, SPm, SPn, TPm, TPk, TPn) combinations.

    Loop hierarchy (resource usage as the primary independent variable):
      1. num_cores  ∈ [1, C]         — resource budget (PE count)
      2. SPm        ∈ [1, num_cores]  — spatial M-axis split
         SPn        = num_cores / SPm  (determined, not a free variable)
      3. TPm ∈ [1, M/SPm], TPk ∈ [1, K], TPn ∈ [1, N/SPn] — temporal splits
         (exact ranges depend on spatial split; M/SPm ≤ M, N/SPn ≤ N)

    Theoretical search space (upper bound):
      Total spatial iterations = ∑_{c=1}^{C} c = C(C+1)/2 = O(C²).
      Temporal ranges are bounded by M, K, N (since M/SPm ≤ M, N/SPn ≤ N).
      Upper bound: O(C² × M × K × N).

    Implementation uses divisor ranges as a lossless compression:
      - SPm × SPn via factor_pairs(num_cores)  — only integer SPn values
      - TPm/TPk/TPn via divisors(M/K/N)        — non-divisors always fail C2/C3
      Using divisors of M (not M/SPm) is safe: values that do not divide
      M/SPm are eliminated by constraint C3, so the result set is identical.

    Actual complexity: O(C · d(C) · d(M) · d(K) · d(N))
      d(n) = number of divisors of n  (average O(ln n))

    All hardware constraints are deferred to Stage 2 filters.
    """
    candidates: List[Candidate] = []
    eb = op.elem_bytes

    divs_M = _divisors(op.M)
    divs_K = _divisors(op.K)
    divs_N = _divisors(op.N)

    for num_cores in range(1, sys_info.total_cores + 1):
        for SPm, SPn in _factor_pairs(num_cores):
            for TPm in divs_M:
                for TPk in divs_K:
                    for TPn in divs_N:
                        # Tile sizes — 0 when spatial×temporal does not
                        # evenly divide the problem dimension.
                        sp_tp_m = SPm * TPm
                        sp_tp_n = SPn * TPn
                        TM = op.M // sp_tp_m if op.M % sp_tp_m == 0 else 0
                        TK = op.K // TPk
                        TN = op.N // sp_tp_n if op.N % sp_tp_n == 0 else 0
                        ws = ws_bytes(TM, TK, TN, eb) if (TM > 0 and TN > 0) else 0
                        num_cols = math.ceil(num_cores / sys_info.comp_tiles_per_col)

                        candidates.append(Candidate(
                            num_cores=num_cores,
                            num_columns=num_cols,
                            SPm=SPm, SPn=SPn,
                            TPm=TPm, TPk=TPk, TPn=TPn,
                            TM=TM, TK=TK, TN=TN,
                            ws_bytes=ws,
                        ))

    return candidates


# ============================================================
# Stage 2: Constraint filters
# ============================================================
def c1_memory(c: Candidate, ct_limit: int) -> bool:
    """C1: Working set must fit in compute tile memory."""
    return c.ws_bytes <= ct_limit


def c2_spatial_divisibility(c: Candidate, op: OpCase) -> bool:
    """C2: Spatial split must evenly divide M and N."""
    return (op.M % c.SPm == 0) and (op.N % c.SPn == 0)


def c3_temporal_divisibility(c: Candidate, op: OpCase) -> bool:
    """C3: Temporal split must evenly divide per-core block sizes."""
    M0 = op.M // c.SPm
    K0 = op.K
    N0 = op.N // c.SPn
    return (M0 % c.TPm == 0) and (K0 % c.TPk == 0) and (N0 % c.TPn == 0)


def c4_column_alignment(c: Candidate, sys_info: SystemInfo) -> bool:
    """C4: Core count must be a multiple of comp_tiles_per_col (column power gating)."""
    return c.num_cores % sys_info.comp_tiles_per_col == 0


def c5_mmul_shape(c: Candidate) -> bool:
    """C5: Tile dimensions must satisfy bf16 mmul<4,8,8> 2x2 expansion alignment."""
    return (c.TM % (2 * MMUL_R) == 0) and (c.TK % MMUL_S == 0) and (c.TN % (2 * MMUL_T) == 0)


@dataclass
class FilterResult:
    """Tracks how many candidates each constraint removes."""
    name: str
    before: int
    after: int

    @property
    def removed(self) -> int:
        return self.before - self.after


def filter_candidates(
    candidates: List[Candidate],
    op: OpCase,
    sys_info: SystemInfo,
) -> Tuple[List[Candidate], List[FilterResult]]:
    """
    Apply all constraints sequentially and track the filtering effect of each.
    """
    ct_limit = sys_info.ct_usable_bytes
    results: List[FilterResult] = []

    filters = [
        ("C1: memory",               lambda c: c1_memory(c, ct_limit)),
        ("C2: spatial divisibility",  lambda c: c2_spatial_divisibility(c, op)),
        ("C3: temporal divisibility", lambda c: c3_temporal_divisibility(c, op)),
        ("C4: column alignment",      lambda c: c4_column_alignment(c, sys_info)),
        ("C5: mmul shape",            lambda c: c5_mmul_shape(c)),
    ]

    current = candidates
    for name, fn in filters:
        before = len(current)
        current = [c for c in current if fn(c)]
        results.append(FilterResult(name=name, before=before, after=len(current)))

    return current, results


# ============================================================
# Stage 3: Cost model evaluation
# ============================================================
def total_data_bytes(op: OpCase, c: Candidate, tp_order: int) -> int:
    """
    Total data transfer volume (bytes) across all cores, with reuse applied.

    The innermost temporal axis determines which operands stay in tile memory
    (temporal reuse) and which are shared across spatial cores (spatial reuse).
    Final transfer = raw transfer / (TR × SR) per operand.
    """
    M, K, N = op.M, op.K, op.N
    eb = op.elem_bytes

    if tp_order == TP_ORDER_M:
        # M innermost: RHS reused across TPm iterations and SPm cores.
        lhs = M * K * c.TPn
        rhs = K * N
        out = 2 * M * N * c.TPk
    elif tp_order == TP_ORDER_N:
        # N innermost: LHS reused across TPn iterations and SPn cores.
        lhs = M * K
        rhs = K * N * c.TPm
        out = 2 * M * N * c.TPk
    else:
        # K innermost: OUT reused across TPk iterations (local accumulation).
        lhs = M * K * c.TPn
        rhs = K * N * c.TPm
        out = 2 * M * N

    return (lhs + rhs + out) * eb


# --- Performance functions (unit: Cycles) ---

def perf_compute(op: OpCase, c: Candidate) -> float:
    """T_comp: compute time assuming all cores run in parallel at peak."""
    return (op.M * op.N * op.K) / (c.SPm * c.SPn * PEAK_MACS)


def perf_comm(op: OpCase, c: Candidate, tp_order: int) -> float:
    """T_comm: data transfer time through shared DRAM bandwidth."""
    return total_data_bytes(op, c, tp_order) / BANDWIDTH_BPC


def perf_overhead(c: Candidate) -> float:
    """T_overhead: pipeline drain/fill cost per temporal iteration."""
    return ALPHA_CYCLES * (c.TPm * c.TPn * c.TPk)


# --- Energy functions (unit: pJ) ---

def energy_dynamic_comp(op: OpCase) -> float:
    """E_dynamic_comp: total MAC energy (constant across candidates)."""
    return op.M * op.N * op.K * E_MAC_PJ


def energy_dynamic_comm(op: OpCase, c: Candidate, tp_order: int) -> float:
    """E_dynamic_comm: DRAM access energy proportional to transfer volume."""
    return total_data_bytes(op, c, tp_order) * E_DRAM_PJ


def energy_static(c: Candidate, t_total: float) -> float:
    """E_static: leakage energy for active tiles over total execution time."""
    return (c.SPm * c.SPn) * P_STATIC_PJ * t_total


# --- Combined evaluation ---

def evaluate_candidate(op: OpCase, c: Candidate, tp_order: int) -> CostResult:
    """Evaluate a single candidate with a specific tpOrder."""
    tc = perf_compute(op, c)
    tm = perf_comm(op, c, tp_order)
    to = perf_overhead(c)
    tt = tc + tm + to

    edc = energy_dynamic_comp(op)
    edm = energy_dynamic_comm(op, c, tp_order)
    es = energy_static(c, tt)
    et = edc + edm + es

    return CostResult(
        candidate=c, tp_order=tp_order,
        t_comp=tc, t_comm=tm, t_overhead=to, t_total=tt,
        e_dynamic_comp=edc, e_dynamic_comm=edm, e_static=es, e_total=et,
        edp=tt * et,
    )


# ============================================================
# Stage 4: EDP-based optimal selection
# ============================================================
def select_optimal(valid: List[Candidate], op: OpCase) -> List[CostResult]:
    """
    For each candidate, evaluate all 3 tpOrders, keep the one with lowest EDP.
    Return the full list sorted by EDP ascending.
    """
    best_per_candidate: List[CostResult] = []
    for c in valid:
        results = [evaluate_candidate(op, c, tpo) for tpo in (0, 1, 2)]
        best = min(results, key=lambda r: r.edp)
        best_per_candidate.append(best)

    best_per_candidate.sort(key=lambda r: r.edp)
    return best_per_candidate


# ============================================================
# Reporting
# ============================================================
def print_search_summary(
    op: OpCase,
    total: int,
    valid: List[Candidate],
    filter_results: List[FilterResult],
) -> None:
    print(f"\n{'='*60}")
    print(f"Op: M={op.M}, K={op.K}, N={op.N}, type={op.elem_type}")
    print(f"{'='*60}")
    print(f"Total enumerated:  {total}")

    for fr in filter_results:
        pct = fr.removed / fr.before * 100 if fr.before > 0 else 0
        print(f"  {fr.name:<30s}  {fr.before:>6d} -> {fr.after:>6d}  "
              f"(removed {fr.removed:>5d}, {pct:5.1f}%)")

    print(f"Valid candidates:  {len(valid)}")

    if valid:
        # Summary by num_cores
        core_counts: Dict[int, int] = {}
        for c in valid:
            core_counts[c.num_cores] = core_counts.get(c.num_cores, 0) + 1
        print(f"\nCandidates by core count:")
        for nc in sorted(core_counts):
            print(f"  {nc:>3d} cores: {core_counts[nc]:>5d} candidates")

        # Show a few examples
        print(f"\nFirst 5 candidates:")
        for c in valid[:5]:
            print(f"  cores={c.num_cores} SP=({c.SPm},{c.SPn}) "
                  f"TP=({c.TPm},{c.TPk},{c.TPn}) "
                  f"Tile=({c.TM},{c.TK},{c.TN}) ws={c.ws_bytes}B")


TP_ORDER_NAMES = {0: "M", 1: "N", 2: "K"}


def print_cost_summary(op: OpCase, ranked: List[CostResult]) -> None:
    """Print Stage 3+4 results: best candidates by core count and overall."""
    if not ranked:
        print("\n  No valid candidates to evaluate.")
        return

    print(f"\n--- Stage 3+4: Cost Model & EDP Ranking ---")

    # Best per core count
    best_by_cores: Dict[int, CostResult] = {}
    for r in ranked:
        nc = r.candidate.num_cores
        if nc not in best_by_cores:
            best_by_cores[nc] = r

    print(f"\n  Best EDP per core count:")
    print(f"  {'cores':>5s}  {'SP':>7s}  {'TP':>11s}  {'tpOrd':>5s}  "
          f"{'T_total':>12s}  {'E_total':>12s}  {'EDP':>14s}")
    for nc in sorted(best_by_cores):
        r = best_by_cores[nc]
        c = r.candidate
        print(f"  {nc:>5d}  ({c.SPm:>2d},{c.SPn:>2d})  "
              f"({c.TPm:>2d},{c.TPk:>2d},{c.TPn:>2d})  "
              f"    {TP_ORDER_NAMES[r.tp_order]}  "
              f"{r.t_total:>12.1f}  {r.e_total:>12.1f}  {r.edp:>14.1f}")

    # Overall top 5
    print(f"\n  Top 5 by EDP:")
    for i, r in enumerate(ranked[:5]):
        c = r.candidate
        print(f"  #{i+1}  cores={c.num_cores} SP=({c.SPm},{c.SPn}) "
              f"TP=({c.TPm},{c.TPk},{c.TPn}) "
              f"tpOrder={TP_ORDER_NAMES[r.tp_order]}  "
              f"T={r.t_total:.1f} E={r.e_total:.1f} EDP={r.edp:.1f}")


# ============================================================
# tc_list.json output (flat schema for the build/run pipeline)
# ============================================================
def cost_result_to_tc(op: OpCase, cr: CostResult) -> Dict[str, Any]:
    """Convert a CostResult to a flat tc.json entry for the pipeline."""
    c = cr.candidate
    # tpOrder: [innermost, middle, outermost] as axis IDs (0=M, 1=N, 2=K)
    # cr.tp_order is the innermost axis; fill remaining with K>M>N default.
    default_order = [TP_ORDER_K, TP_ORDER_M, TP_ORDER_N]
    tp_order_full = [cr.tp_order] + [ax for ax in default_order if ax != cr.tp_order]
    return {
        "M": op.M, "K": op.K, "N": op.N,
        "elemType": op.elem_type,
        "numCores": c.num_cores,
        "doubleBuffer": False,
        "t_total_pred": round(cr.t_total, 2),
        "levels": [
            {
                "SPm": c.SPm, "SPn": c.SPn,
                "TPm": c.TPm, "TPk": c.TPk, "TPn": c.TPn,
                "TM": c.TM, "TK": c.TK, "TN": c.TN,
                "tpOrder": tp_order_full,
            }
        ],
    }


def write_tc_list(tc_cases: List[Dict[str, Any]], out_path: Path) -> None:
    """Write tc_list.json for the build/run pipeline."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"cases": tc_cases}, f, indent=2)


# ============================================================
# CLI
# ============================================================
def select_validation_candidates(
    ranked: List[CostResult],
) -> List[CostResult]:
    """
    Return all candidates sorted by predicted T_total (ascending).
    Used for full-spectrum validation against actual NPU measurements.
    """
    return sorted(ranked, key=lambda r: r.t_total)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exhaustive search + constraint filtering for parallelization configs")
    p.add_argument("--op", default=str(DEFAULT_OP_PATH),
                   help="Path to op_list.json")
    p.add_argument("--sys", default=str(DEFAULT_SYS_PATH),
                   help="Path to xdna2_info.json")
    p.add_argument("--out", default="",
                   help="Path to write cost results JSON (optional)")
    p.add_argument("--validate", default="",
                   help="Path to write validation tc_list.json (optional)")
    p.add_argument("--op-index", type=int, default=-1,
                   help="Run only this 0-based op index (-1 = all)")
    return p.parse_args(argv)


def process_op(op: OpCase, sys_info: SystemInfo) -> List[CostResult]:
    """Run Stage 1 through Stage 4 for a single op case."""
    all_candidates = enumerate_candidates(op, sys_info)
    valid, filter_results = filter_candidates(all_candidates, op, sys_info)
    print_search_summary(op, len(all_candidates), valid, filter_results)
    ranked = select_optimal(valid, op)
    print_cost_summary(op, ranked)
    return ranked


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    op_path = Path(args.op).resolve()
    sys_path = Path(args.sys).resolve()

    try:
        ops = load_op_list(op_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    try:
        sys_info = load_system_info(sys_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"[INFO] {len(ops)} ops from {op_path}")
    print(f"[INFO] HW: {sys_info.total_cores} cores, "
          f"{sys_info.comp_tiles_per_col} tiles/col, "
          f"{sys_info.max_columns} cols, "
          f"{sys_info.spm_size_bytes}B/tile "
          f"(usable {sys_info.ct_usable_bytes}B)")

    # Select ops to process
    if args.op_index >= 0:
        if args.op_index >= len(ops):
            print(f"[ERROR] op-index {args.op_index} out of range", file=sys.stderr)
            return 1
        targets = [(args.op_index, ops[args.op_index])]
    else:
        targets = list(enumerate(ops))

    all_ranked: Dict[int, List[CostResult]] = {}
    for idx, op in targets:
        ranked = process_op(op, sys_info)
        all_ranked[idx] = ranked

    # Write output if requested
    if args.out:
        out_path = Path(args.out).resolve()
        output = {}
        for idx, ranked in all_ranked.items():
            op = ops[idx]
            key = f"M{op.M}_K{op.K}_N{op.N}"
            output[key] = {
                "op": {"M": op.M, "K": op.K, "N": op.N, "elemType": op.elem_type},
                "num_ranked": len(ranked),
                "ranked": [
                    {
                        **asdict(r.candidate),
                        "tp_order": r.tp_order,
                        "t_comp": r.t_comp,
                        "t_comm": r.t_comm,
                        "t_overhead": r.t_overhead,
                        "t_total": r.t_total,
                        "e_dynamic_comp": r.e_dynamic_comp,
                        "e_dynamic_comm": r.e_dynamic_comm,
                        "e_static": r.e_static,
                        "e_total": r.e_total,
                        "edp": r.edp,
                    }
                    for r in ranked
                ],
            }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\n[INFO] Wrote {out_path}")

    # Write validation tc_list.json if requested
    if args.validate:
        val_path = Path(args.validate).resolve()
        tc_cases: List[Dict[str, Any]] = []
        for idx, ranked in all_ranked.items():
            op = ops[idx]
            selected = select_validation_candidates(ranked)
            for cr in selected:
                tc_cases.append(cost_result_to_tc(op, cr))
            print(f"\n[INFO] Op M{op.M}_K{op.K}_N{op.N}: "
                  f"{len(selected)} validation candidates selected")
        write_tc_list(tc_cases, val_path)
        print(f"[INFO] Wrote {val_path} ({len(tc_cases)} cases)")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
