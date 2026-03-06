#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_tc_list.py
- Read op_list.json (input specs) -> return list of op cases
- Accept a list of tc cases -> write tc_list.json (output for run_tc_all.sh)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import tempfile
import shutil
import math

# ---------- Defaults based on repo layout ----------
# scripts/  (this file)
# data/     (op_list.json lives here)
# out/      (tc_list.json is written here)
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[1]           # test/onnx-mlir/
REPO_ROOT = THIS_FILE.parents[3]           # mlir-aie/
DATA_DIR  = ROOT_DIR / "data"
OUT_DIR   = ROOT_DIR / "out"

DEFAULT_OP_PATH  = DATA_DIR / "op_list.json"
DEFAULT_TC_PATH  = OUT_DIR / "tc_list.json"
DEFAULT_SYS_PATH = REPO_ROOT / "include" / "onnx" / "Target" / "XDNA2" / "xdna2_info.json"


# ---------- I/O primitives ----------
def load_op_list(op_json_path: Path) -> List[Dict[str, Any]]:
    """
    Read data/op_list.json and return the list under "cases" (Minimal validation only).
    """
    if not op_json_path.is_file():
        raise FileNotFoundError(f"op_list.json not found: {op_json_path}")

    with op_json_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    cases = doc.get("cases", None)
    if not isinstance(cases, list):
        raise ValueError("Invalid op_list.json: missing 'cases' array")

    # (Optional) light validation of each item
    for i, c in enumerate(cases):
        if not isinstance(c, dict):
            raise ValueError(f"Invalid case at index {i}: not an object")

    return cases


def atomic_write_json(obj: Any, out_path: Path) -> None:
    """
    Write JSON atomically to out_path (prevent partial writes).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False, dir=str(out_path.parent), encoding="utf-8") as tmp:
            json.dump(obj, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            tmp_path = Path(tmp.name)
        # POSIX atomic replace
        shutil.move(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(text: str, out_path: Path) -> None:
    """
    Write TEXT atomically to out_path (prevent partial writes).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".logtmp", delete=False, dir=str(out_path.parent), encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def write_tc_list(tc_cases: List[Dict[str, Any]], out_json_path: Path) -> None:
    """
    Accept a list of testcase dicts and write to out/tc_list.json
    Structure: { "cases": [ ... ] }
    """
    payload = {"cases": tc_cases}
    atomic_write_json(payload, out_json_path)


# ---------- Tiling primitives ----------
# SW overhead (stack + heap + reserved) subtracted from HW tile memory.
# This is a software constant, not a hardware property.
CTILE_RESERVED_BYTES = 4 * 1024

TM_UNIT, TK_UNIT, TN_UNIT = 1, 1, 1     # unit sizes for tiles
ELEM_SIZE_MAP = {"f16": 2, "bf16": 2, "f32": 4, "i8": 1, "i16": 2, "i32": 4, "ui8": 1, "ui16": 2, "ui32": 4}


def load_system_info(sys_json_path: Path) -> Dict[str, Any]:
    """
    Read xdna2_info.json and return a flat dict of hardware parameters
    used by the tiling search.
    """
    if not sys_json_path.is_file():
        raise FileNotFoundError(f"system info JSON not found: {sys_json_path}")

    with sys_json_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    sys_obj = doc.get("system")
    if not isinstance(sys_obj, dict):
        raise ValueError("Invalid system info JSON: missing 'system' object")

    spm_levels = sys_obj.get("spm_levels", [])
    if not spm_levels:
        raise ValueError("Invalid system info JSON: empty 'spm_levels'")

    device = sys_obj.get("device", {})

    return {
        "total_cores":       int(sys_obj["total_cores"]),
        "comp_tiles_per_col": int(device.get("comp_tiles_per_col", 4)),
        "spm_size_bytes":    int(spm_levels[0]["spm_size_bytes"]),
        "mem_tile_mem_bytes": int(device.get("mem_tile_mem_bytes", 524288)),
    }

def _est_ws_bytes(TM: int, TK: int, TN: int, elem_bytes: int) -> int:
    """
    Estimate the working-set size, in bytes, to process one GEMM tile at a given level.
    Current placeholder: A (TM*TK) + B (TK*TN) + C (TM*TN accum)
    """
    return elem_bytes * (TM * TK + TK * TN + TM * TN)

def _divisors(n: int) -> List[int]:
    """All positive divisors of n (unordered)."""
    ds = set()
    for b in range(1, int(math.isqrt(n)) + 1):
        if n % b == 0:
            ds.add(b)
            ds.add(n // b)
    return sorted(ds)

def _factor_pairs(n: int) -> List[Tuple[int, int]]:
    """
    Enumerate all pairs (a,b) with a*b == n. (ordered)
    """
    pairs: List[Tuple[int, int]] = []
    for a in _divisors(n):
        b = n // a
        pairs.append((a, b))
    return pairs

def _triple_factorizations(n: int) -> List[Tuple[int, int, int]]:
    """
    Enumerate all triples (a,b,c) with a*b*c == n. (ordered)
    """
    triples: List[Tuple[int, int, int]] = []
    for a in _divisors(n):
        rem = n // a
        for b in _divisors(rem):
            c = rem // b
            if a * b * c == n:
                triples.append((a, b, c))
    return triples

def _estimate_cost(
    SPm: int, SPn: int,
    TPm: int, TPk: int, TPn: int,
    TM: int, TK: int, TN: int,
) -> Tuple[Dict[str, int], Dict[str, int], int, str, List[int]]:
    """
    Compute data-reuse cost for a given tiling configuration.
    Returns (total_traffic, reuse_savings, total_sum, winner_axis, tpOrder).

    total_traffic: per-tensor byte counts {"MK", "KN", "MN"}
    reuse_savings: bytes saved per axis {"M", "N", "K"}
    total_sum:     sum of all tensor traffic
    winner_axis:   axis with lowest score ("M", "N", or "K")
    tpOrder:       loop order [innermost, middle, outermost] as axis ids (0=M,1=N,2=K)
    """
    # Spatial reuse: how many times each tensor is reused across the SPm*SPn tile grid.
    # All axes see the same spatial reuse (A reused SPn times, B reused SPm times).
    spatial_reuse_rate = {
        "M": {"MK": SPn, "KN": SPm, "MN": 1},
        "N": {"MK": SPn, "KN": SPm, "MN": 1},
        "K": {"MK": SPn, "KN": SPm, "MN": 1},
    }

    # Temporal reuse: depends on which axis is innermost (reuse axis).
    # The reuse axis stays in local memory; other tensors cycle each iteration.
    temporal_reuse_rate = {
        "M": {"MK": 1,   "KN": TPm, "MN": 1},
        "N": {"MK": TPn, "KN": 1,   "MN": 1},
        "K": {"MK": 1,   "KN": 1,   "MN": TPk},
    }

    # Total element accesses per tensor across all iterations and tiles.
    # MN factor of 2: read for accumulation + write output.
    total = {
        "MK": (TM * TK) * TPm * TPn * TPk * SPm * SPn,
        "KN": (TK * TN) * TPm * TPn * TPk * SPm * SPn,
        "MN": (2 * TM * TN) * TPm * TPn * TPk * SPm * SPn,
    }
    total_sum = sum(total.values())

    # Reuse savings: redundant loads eliminated by keeping data on-chip.
    # reuse[axis] = sum over tensors of: total * (reuse_factor - 1) / reuse_factor
    reuse: Dict[str, int] = {}
    for axis in ("M", "N", "K"):
        s = 0
        for tensor in ("MK", "KN", "MN"):
            t = total[tensor]
            srr = spatial_reuse_rate[axis][tensor]
            trr = temporal_reuse_rate[axis][tensor]
            s += t * max(srr * trr - 1, 0) // max(srr * trr, 1)
        reuse[axis] = s

    # Score = total traffic minus reuse savings; lower is better.
    score: Dict[str, int] = {}
    for axis in ("M", "N", "K"):
        score[axis] = total_sum - reuse[axis]

    # Winner: axis with lowest score; ties broken by largest reuse, then K>M>N priority.
    tie_rank = {"M": 1, "N": 0, "K": 2}
    winner = min(("M", "N", "K"), key=lambda ax: (score[ax], -reuse[ax], -tie_rank[ax]))

    # tpOrder: winner first, then default K->M->N for remaining axes.
    axis_id = {"M": 0, "N": 1, "K": 2}
    default_order = [axis_id["K"], axis_id["M"], axis_id["N"]]
    win_id = axis_id[winner]
    tp_order = [win_id] + [ax for ax in default_order if ax != win_id]

    return total, reuse, score, total_sum, winner, tp_order


def _find_best_config(
    M0: int, K0: int, N0: int,
    SPm: int, SPn: int,
    elem_bytes: int,
    ct_limit: int,
    log_lines: List[str],
) -> Optional[Tuple[Any, ...]]:
    """
    Search for the best (TPm,TPk,TPn) factorization that fits in ct_limit.

    Greedy: starts from the minimum TPtotal that could fit, stops at the first
    feasible TPtotal value. Within each TPtotal, picks the factorization with
    the lowest cost score.

    Returns None if no feasible config exists, otherwise a tuple:
      (key, (TPm,TPk,TPn), tp_order, TM, TK, TN, total_sum, reuse, score)
    """
    ws_bytes = _est_ws_bytes(M0, K0, N0, elem_bytes)
    TPtotal_init = max(1, math.ceil(ws_bytes / ct_limit))
    TPtotal_max  = max(1, M0 * K0 * N0)

    tie_rank = {"M": 1, "N": 0, "K": 2}
    best = None

    for TPtotal in range(TPtotal_init, TPtotal_max + 1):
        triples = _triple_factorizations(TPtotal)
        found_this_total = False

        for (TPm, TPk, TPn) in triples:
            if (M0 % TPm) or (K0 % TPk) or (N0 % TPn):
                continue

            TM = M0 // TPm
            TK = K0 // TPk
            TN = N0 // TPn

            ws_step = _est_ws_bytes(TM, TK, TN, elem_bytes)
            if ws_step > ct_limit:
                continue

            total, reuse, score, total_sum, winner, tp_order = _estimate_cost(
                SPm, SPn, TPm, TPk, TPn, TM, TK, TN
            )

            log_lines.append(
                "  TPtotal={}: TP=(m={},k={},n={}) Tiles(TM,TK,TN)=({},{},{}) ws_step={}B "
                "total={} reuse{{M:{}, N:{}, K:{}}} score{{M:{}, N:{}, K:{}}} "
                "winner={} reuse={} score={} tpOrder={}".format(
                    TPtotal, TPm, TPk, TPn, TM, TK, TN, ws_step, total_sum,
                    reuse["M"], reuse["N"], reuse["K"],
                    score["M"], score["N"], score["K"],
                    winner, reuse[winner], score[winner], tp_order
                )
            )

            key = (score[winner], -reuse[winner], -tie_rank[winner])

            if (best is None) or (key < best[0]):
                best = (key, (TPm, TPk, TPn), tp_order, TM, TK, TN, total_sum, reuse[winner], score[winner])
                found_this_total = True

        # Stop at first feasible TPtotal (greedy: smallest temporal split)
        if found_this_total:
            break

    return best


def make_tc_cases(op_cases: List[Dict[str, Any]], sys_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Turn op cases into tc cases using hardware parameters from sys_info.
    """
    ctile_max_count = sys_info["total_cores"]
    ctile_step      = sys_info["comp_tiles_per_col"]
    ctile_mem_limit = sys_info["spm_size_bytes"] - CTILE_RESERVED_BYTES

    tc_cases: List[Dict[str, Any]] = []
    log_lines: List[str] = []

    # ---------- check the op cases ----------
    for idx, base in enumerate(op_cases, start=1):
        try:
            M = int(base["M"])
            K = int(base["K"])
            N = int(base["N"])
        except (KeyError, TypeError, ValueError):
            msg = f"[ERROR] op_case#{idx} incorrect format: {base}"
            print(msg, file=sys.stderr)
            log_lines.append(msg)
            continue

        errs = []
        if M % TM_UNIT != 0:
            errs.append(f"M={M} (mod {TM_UNIT} != 0)")
        if K % TK_UNIT != 0:
            errs.append(f"K={K} (mod {TK_UNIT} != 0)")
        if N % TN_UNIT != 0:
            errs.append(f"N={N} (mod {TN_UNIT} != 0)")

        if errs:
            msg = "[ERROR] op_case#{} unit-size divisibility not satisfied: {}".format(idx, ", ".join(errs))
            print(msg, file=sys.stderr)
            log_lines.append(msg)
            continue  # skip this op case

        elem_type = str(base.get("elemType", "f32"))
        elem_bytes = ELEM_SIZE_MAP.get(elem_type.lower(), 4)

        # ---------- outermost: double buffer toggle ----------
        # for double_buffer in (False, True):
        for double_buffer in (False,):
            ct_limit = (ctile_mem_limit // 2) if double_buffer else ctile_mem_limit

            # ---------- numLastSpm: number of compute tiles ----------
            for numLastSpm in range(ctile_step, ctile_max_count + 1, ctile_step):

                # ---------- # DRAM→L1: (SPm,SPn) ----------
                for SPm, SPn in _factor_pairs(numLastSpm):

                    # Must divide M and N to form CT-assigned block
                    if (M % SPm) or (N % SPn):
                        continue

                    # CT-assigned block before temporal splitting
                    M0 = M // SPm
                    N0 = N // SPn
                    K0 = K

                    ws_bytes = _est_ws_bytes(M0, K0, N0, elem_bytes)
                    TPtotal_init = max(1, math.ceil(ws_bytes / ct_limit))
                    TPtotal_max  = max(1, M0 * K0 * N0)

                    # Log header for this (SPm,SPn)
                    log_lines.append(
                        f"[OP CASE#{idx}] MKN=({M},{K},{N}) elemType={elem_type} numLastSpm={numLastSpm} SP=(m={SPm},n={SPn}) "
                        f"CTblock(M0,K0,N0)=({M0},{K0},{N0}) ct_limit={ct_limit}B ws_block={ws_bytes}B "
                        f"TPtotal_init={TPtotal_init} TPtotal_max={TPtotal_max}"
                    )

                    best = _find_best_config(M0, K0, N0, SPm, SPn, elem_bytes, ct_limit, log_lines)

                    if best is None:
                        log_lines.append("  -> No feasible TP for this SPm/SPn; continue")
                        continue

                    (_, (TPm, TPk, TPn), tp_order, TM, TK, TN, total, reuse, score) = best

                    log_lines.append(
                        "  [SELECT] TP=(m={},k={},n={}) Tiles(TM,TK,TN)=({},{},{}) tpOrder={} "
                        "reuse_axis={} total={} reuse={} score={}".format(
                            TPm, TPk, TPn, TM, TK, TN, tp_order,
                            ["M","N","K"][tp_order[0]],
                            total, reuse, score
                        )
                    )

                    tc_cases.append({
                        "M": M, "K": K, "N": N,
                        "elemType": elem_type,
                        "numCores": numLastSpm,
                        "doubleBuffer": double_buffer,
                        "levels": [
                            {
                                "SPm": SPm, "SPn": SPn,
                                "TPm": TPm, "TPk": TPk, "TPn": TPn,
                                "TM": TM, "TK": TK, "TN": TN,
                                "tpOrder": tp_order,
                            }
                        ],
                    })

    return tc_cases, log_lines


# ---------- CLI ----------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate out/tc_list.json from data/op_list.json")
    p.add_argument("--op",  default=str(DEFAULT_OP_PATH), help="Path to input op_list.json (default: data/op_list.json)")
    p.add_argument("--sys", default=str(DEFAULT_SYS_PATH), help="Path to xdna2_info.json hardware config (default: include/onnx/Target/XDNA2/xdna2_info.json)")
    p.add_argument("--out", default=str(DEFAULT_TC_PATH), help="Path to output tc_list.json (default: out/tc_list.json)")
    p.add_argument("--dry-run", action="store_true", help="Do not write, only print summary")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    op_path  = Path(args.op).resolve()
    sys_path = Path(args.sys).resolve()
    out_path = Path(args.out).resolve()
    log_path = out_path.with_name("tc_list.log")

    # 1) Read op_list.json -> list of op cases
    try:
        op_cases = load_op_list(op_path)
    except Exception as e:
        print(f"[ERROR] Failed to load op_list: {e}", file=sys.stderr)
        return 2

    print(f"[INFO] Loaded {len(op_cases)} op cases from: {op_path}")

    # 2) Read xdna2_info.json -> hardware parameters
    try:
        sys_info = load_system_info(sys_path)
    except Exception as e:
        print(f"[ERROR] Failed to load system info: {e}", file=sys.stderr)
        return 2

    print(f"[INFO] Loaded system info from: {sys_path}")
    print(f"[INFO]   total_cores={sys_info['total_cores']}"
          f" comp_tiles_per_col={sys_info['comp_tiles_per_col']}"
          f" spm_size={sys_info['spm_size_bytes']}B"
          f" ct_usable={sys_info['spm_size_bytes'] - CTILE_RESERVED_BYTES}B")

    # 3) Make tc cases from op cases
    tc_cases, log_lines = make_tc_cases(op_cases, sys_info)

    print(f"[INFO] Makes {len(tc_cases)} tc cases.")

    # 3) Write tc_list.json
    if args.dry_run:
        print("[DRY-RUN] Would write tc_list.json to:", out_path)
        print("[DRY-RUN] Would write tc_list.log to:", log_path)
    else:
        try:
            write_tc_list(tc_cases, out_path)
            print(f"[INFO] Wrote tc_list.json to: {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write tc_list.json: {e}", file=sys.stderr)
            return 3

        try:
            atomic_write_text("\n".join(log_lines) + "\n", log_path)
            print(f"[INFO] Wrote tc_list.log to: {log_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write tc_list.log: {e}", file=sys.stderr)
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
