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
from typing import List, Dict, Any, Tuple
import tempfile
import shutil
import math

# ---------- Defaults based on repo layout ----------
# scripts/  (this file)
# data/     (op_list.json lives here)
# out/      (tc_list.json is written here)
THIS_FILE = Path(__file__).resolve()
ROOT_DIR  = THIS_FILE.parents[1]
DATA_DIR  = ROOT_DIR / "data"
OUT_DIR   = ROOT_DIR / "out"

DEFAULT_OP_PATH  = DATA_DIR / "op_list.json"
DEFAULT_TC_PATH  = OUT_DIR / "tc_list.json"


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
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False, dir=str(out_path.parent), encoding="utf-8") as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        tmp_path = Path(tmp.name)
    # POSIX atomic replace
    shutil.move(str(tmp_path), str(out_path))


def atomic_write_text(text: str, out_path: Path) -> None:
    """
    Write TEXT atomically to out_path (prevent partial writes).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".logtmp", delete=False, dir=str(out_path.parent), encoding="utf-8") as tmp:
        tmp.write(text)
        tmp.flush()
        tmp_path = Path(tmp.name)
    shutil.move(str(tmp_path), str(out_path))


def write_tc_list(tc_cases: List[Dict[str, Any]], out_json_path: Path) -> None:
    """
    Accept a list of testcase dicts and write to out/tc_list.json
    Structure: { "cases": [ ... ] }
    """
    payload = {"cases": tc_cases}
    atomic_write_json(payload, out_json_path)


# ---------- Tiling primitives ----------
CTILE_MEM_LIMIT = 64 * 1024 - 4 * 1024  # 64KB - 1KB (stack) - 1KB (heap) - 2KB (reserved)
MEMTILE_MEM_LIMIT = 512 * 1024          # 512KB (unused)
TM_UNIT, TK_UNIT, TN_UNIT = 1, 1, 1     # unit sizes for tiles
ELEM_SIZE_MAP = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
                 "ui8": 1, "ui16": 2, "ui32": 4, "ui64": 8}

def _est_ws_bytes(TM: int, TK: int, TN: int, elem_bytes: int) -> int:
    """
    Estimate the working-set size, in bytes, to process one GEMM tile at a given level.
    Current placeholder: A (TM*TK) + B (TK*TN) + C (TM*TN accum)
    """
    return elem_bytes * (TM * TK + TK * TN + TM * TN)

def _factor_pairs(n: int) -> List[Tuple[int, int]]:
    """
    Return unordered factor pairs (a, b) such that a * b == n with a >= b.

    Used to enumerate spatial parallelization splits (SPm, SPn) where order does not
    matter. Time complexity is O(sqrt(n)).
    """
    return [(n // b, b) for b in range(1, int(math.isqrt(n)) + 1) if n % b == 0]

def _divisors(n: int) -> List[int]:
    """All positive divisors of n (unordered)."""
    ds = set()
    for b in range(1, int(math.isqrt(n)) + 1):
        if n % b == 0:
            ds.add(b)
            ds.add(n // b)
    return sorted(ds)

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


def make_tc_cases(op_cases: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Placeholder: turn op cases into tc cases.
    """
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
            ct_limit = (CTILE_MEM_LIMIT // 2) if double_buffer else CTILE_MEM_LIMIT     # 30KB if enabled

            # ---------- numLastSpm: 4..32 ----------
            for numLastSpm in range(4, 32 + 1, 4):

                # ---------- # DRAM→L1: (SPm,SPn) ----------
                for SPm, SPn in _factor_pairs(numLastSpm):

                    # Must divide M and N to form CT-assigned block
                    if (M % SPm) or (N % SPn):
                        continue

                    # ---------- # DRAM→L1: (SPm,SPn,TPm,TPn,TPk) ----------
                    # CT-assigned block before temporal splitting
                    M0 = M // SPm
                    N0 = N // SPn
                    K0 = K

                    # Working-set for that block
                    ws_bytes = _est_ws_bytes(M0, K0, N0, elem_bytes)

                    # Required total temporal factor (>=1)
                    TPtotal_init = max(1, math.ceil(ws_bytes / ct_limit))
                    TPtotal_max  = max(1, M0 * K0 * N0)  # upper bound per spec

                    # Enumerate TP triples starting from TPtotal_init, increasing if needed
                    best = None  # (key, (TPm,TPk,TPn), tp_order, TM, TK, TN)
                    tie_rank = {"M": 2, "N": 1, "K": 0}

                    # Log header for this (SPm,SPn)
                    log_lines.append(
                        f"[CASE#{idx}] MKN=({M},{K},{N}) elemType={elem_type} numLastSpm={numLastSpm} SP=(m={SPm},n={SPn}) "
                        f"CTblock(M0,K0,N0)=({M0},{K0},{N0}) ct_limit={ct_limit}B ws_block={ws_bytes}B "
                        f"TPtotal_init={TPtotal_init}"
                    )

                    for TPtotal in range(TPtotal_init, TPtotal_max + 1):
                        triples = _triple_factorizations(TPtotal)
                        found_this_total = False
                        for (TPm, TPk, TPn) in triples:
                            # Divisibility constraints
                            if (M0 % TPm) or (K0 % TPk) or (N0 % TPn):
                                continue

                            # Per-step tile sizes
                            TM = M0 // TPm
                            TK = K0 // TPk
                            TN = N0 // TPn

                            ws_step = _est_ws_bytes(TM, TK, TN, elem_bytes)
                            if ws_step > ct_limit:
                                continue

                            # Reuse scores (same formula), at this level using (M0,K0,N0)
                            scores = {
                                "M": K0 * N0 * max(TPm - 1, 0) * SPm,
                                "N": M0 * K0 * max(TPn - 1, 0) * SPn,
                                "K": 2 * M0 * N0 * max(TPk - 1, 0),
                            }
                            
                            # Winner axis & extra transfer cost
                            winner = max(("M","N","K"), key=lambda ax: (scores[ax], tie_rank[ax]))
                            extra_cost = sum(v for ax, v in scores.items() if ax != winner)
                            winner_score = scores[winner]

                            # TP order: winner first, then default M->N->K
                            axis_id = {"M": 0, "N": 1, "K": 2}
                            default_order = [axis_id["M"], axis_id["N"], axis_id["K"]]
                            win_id = axis_id[winner]
                            tp_order = [win_id] + [ax for ax in default_order if ax != win_id]

                            # Selection key: minimize extra_cost; tie → larger winner_score;
                            # then winner priority M>N>K; then smaller TP sum for compactness.
                            active_cnt = (1 if TPm > 1 else 0) + (1 if TPk > 1 else 0) + (1 if TPn > 1 else 0)
                            if active_cnt == 1:
                                key = (extra_cost, -tie_rank[winner], (TPm + TPk + TPn))
                            else:
                                key = (extra_cost, -winner_score, -tie_rank[winner], (TPm + TPk + TPn))

                            # ---- LOG per valid candidate ----
                            log_lines.append(
                                "  TPtotal={}: TP=(m={},k={},n={}) Tiles(TM,TK,TN)=({},{},{}) ws_step={}B "
                                "scores{{M:{}, N:{}, K:{}}} winner={} reuse={} extra={} tpOrder={}".format(
                                    TPtotal, TPm, TPk, TPn, TM, TK, TN, ws_step,
                                    scores["M"], scores["N"], scores["K"],
                                    winner, winner_score, extra_cost, tp_order
                                )
                            )

                            if (best is None) or (key < best[0]):
                                best = (key, (TPm, TPk, TPn), tp_order, TM, TK, TN)
                                found_this_total = True

                        if found_this_total:
                            break  # stop increasing TPtotal once feasible found

                    if best is None:
                        log_lines.append("  -> No feasible TP for this SPm/SPn; continue")
                        continue

                    (_, (TPm, TPk, TPn), tp_order, TM, TK, TN) = best

                    # ---- LOG final selection for this (SPm,SPn) ----
                    log_lines.append(
                        "  [SELECT] TP=(m={},k={},n={}) Tiles(TM,TK,TN)=({},{},{}) tpOrder={} "
                        "reuse_axis={} extra={} reuse={}".format(
                            TPm, TPk, TPn, TM, TK, TN, tp_order,
                            ["M","N","K"][tp_order[0]],  # first in order is winner axis
                            # recompute for logging clarity
                            sum(v for ax, v in {
                                "M": K0 * N0 * max(TPm - 1, 0) * SPm,
                                "N": M0 * K0 * max(TPn - 1, 0) * SPn,
                                "K": 2 * M0 * N0 * max(TPk - 1, 0),
                            }.items() if ax != ["M","N","K"][tp_order[0]]),
                            {
                                "M": K0 * N0 * max(TPm - 1, 0) * SPm,
                                "N": M0 * K0 * max(TPn - 1, 0) * SPn,
                                "K": 2 * M0 * N0 * max(TPk - 1, 0),
                            }[["M","N","K"][tp_order[0]]]
                        )
                    )

                    # Emit one-level case
                    tc_cases.append({
                        "M": M, "K": K, "N": N,
                        "elemType": elem_type,
                        "numLevel": 1,
                        "numLastSpm": numLastSpm,
                        "doubleBuffer": double_buffer,
                        "levels": [
                            {
                                "label": "DRAM<->CT",
                                "numSpm": numLastSpm,
                                "SPm": SPm, "SPn": SPn,
                                "TPm": TPm, "TPk": TPk, "TPn": TPn,
                                "TM": TM, "TK": TK, "TN": TN,
                                "tpOrder": tp_order
                            }
                        ]
                    })

    return tc_cases, log_lines


# ---------- CLI ----------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate out/tc_list.json from data/op_list.json")
    p.add_argument("--op",  default=str(DEFAULT_OP_PATH), help="Path to input op_list.json (default: data/op_list.json)")
    p.add_argument("--out", default=str(DEFAULT_TC_PATH), help="Path to output tc_list.json (default: out/tc_list.json)")
    p.add_argument("--dry-run", action="store_true", help="Do not write, only print summary")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    op_path  = Path(args.op).resolve()
    out_path = Path(args.out).resolve()
    log_path = out_path.with_name("tc_list.log")

    # 1) Read op_list.json -> list of op cases
    try:
        op_cases = load_op_list(op_path)
    except Exception as e:
        print(f"[ERROR] Failed to load op_list: {e}", file=sys.stderr)
        return 2

    print(f"[INFO] Loaded {len(op_cases)} op cases from: {op_path}")

    # 2) Make tc cases from op cases
    tc_cases, log_lines = make_tc_cases(op_cases)

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
