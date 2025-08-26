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
from typing import List, Dict, Any
import tempfile
import shutil

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


def write_tc_list(tc_cases: List[Dict[str, Any]], out_json_path: Path) -> None:
    """
    Accept a list of testcase dicts and write to out/tc_list.json
    Structure: { "cases": [ ... ] }
    """
    payload = {"cases": tc_cases}
    atomic_write_json(payload, out_json_path)


# ---------- Tiling primitives ----------
CTILE_MEM_LIMIT = 64 * 1024 - 4 * 1024  # 64KB - 1KB (stack) - 1KB (heap) - 2KB (reserved)
MEMTILE_MEM_LIMIT = 512 * 1024          # 512KB
ELEM_BYTES = 4                          # f32 default. change if needed.


def est_ctile_bytes(TM: int, TK: int, TN: int, elem_bytes: int = ELEM_BYTES) -> int:
    """
    Estimate bytes required on a compute tile for one matmul tile.
    Current placeholder: A (TM*TK) + B (TK*TN) + C (TM*TN accum)
    """
    return elem_bytes * (TM * TK + TK * TN + TM * TN)


def est_memtile_bytes(MemTM: int, MemTK: int, MemTN: int, elem_bytes: int = ELEM_BYTES) -> int:
    """
    Estimate bytes required on a mem tile for one matmul tile.
    """
    return elem_bytes * (MemTM * MemTK + MemTK * MemTN + MemTM * MemTN)


def decide_memtile_shape(
    M: int, K: int, N: int,
    TM: int, TK: int, TN: int,
    num_ct: int = 4,
    num_last_spm: int = 1,
) -> Optional[Tuple[int, int, int]]:
    """
    Decide (MemTM, MemTK, MemTN) for the mem tile.
    Return None if not feasible (e.g., cannot split to CTs, etc.).
    """
    MemTMUnit = 4 * TM
    MemTKUnit = TK
    MemTNUnit = TN

    # base units
    memtm = MemTMUnit
    memtk = MemTKUnit
    memtn = MemTNUnit

    # quick feasibility check for base unit
    if est_memtile_bytes(memtm, memtk, memtn) > MEMTILE_MEM_LIMIT:
        return None

    if M % MemTMUnit != 0:
        return None

    # number of memtiles along M axis
    n_memtiles_M = M // MemTMUnit
    # must split evenly across num_last_spm
    if n_memtiles_M % num_last_spm != 0:
        return None

    # ---- 1) Expand K axis first (by MemTKUnit), require K % MemTK == 0 ----
    # monotonic in MemTK -> break on first overflow
    for cand in range(memtk + MemTKUnit, K + 1, MemTKUnit):
        if K % cand != 0:
            continue
        if est_memtile_bytes(memtm, cand, memtn) <= MEMTILE_MEM_LIMIT:
            memtk = cand
        else:
            break

    # if couldn't reach full K, stop here
    if memtk < K:
        return (memtm, memtk, memtn)

    # ---- 2) Expand N axis next (by MemTNUnit), require N % MemTN == 0 ----
    memtn_max = min(MemTNUnit * 8, N)   # AIE hardware constraint (maximum value of lock is 63)
    for cand in range(memtn + MemTNUnit, memtn_max + 1, MemTNUnit):
        if N % cand != 0:
            continue
        if est_memtile_bytes(memtm, memtk, cand) <= MEMTILE_MEM_LIMIT:
            memtn = cand
        else:
            break

    # ---- 3) Expand M axis last (by MemTMUnit), require M % MemTM == 0 ----
    for cand in range(memtm + MemTMUnit, M + 1, MemTMUnit):
        if M % cand != 0:
            continue
        if (M // cand) % num_last_spm:
            continue
        if est_memtile_bytes(cand, memtk, memtn) <= MEMTILE_MEM_LIMIT:
            memtm = cand
        else:
            break

    return (memtm, memtk, memtn)


def make_tc_cases(op_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Placeholder: turn op cases into tc cases.
    """
    tc_cases: List[Dict[str, Any]] = []

    # unit sizes for tiles
    TM_UNIT, TK_UNIT, TN_UNIT = 16, 16, 4

    for idx, base in enumerate(op_cases, start=1):
        try:
            M = int(base["M"])
            K = int(base["K"])
            N = int(base["N"])
        except (KeyError, TypeError, ValueError):
            print(f"[ERROR] op_case#{idx} incorrect format: {base}", file=sys.stderr)
            continue

        errs = []
        if M % TM_UNIT != 0:
            errs.append(f"M={M} (mod {TM_UNIT} != 0)")
        if K % TK_UNIT != 0:
            errs.append(f"K={K} (mod {TK_UNIT} != 0)")
        if N % TN_UNIT != 0:
            errs.append(f"N={N} (mod {TN_UNIT} != 0)")

        if errs:
            print(f"[ERROR] op_case#{idx} unit-size divisibility not satisfied: " + ", ".join(errs), file=sys.stderr)
            continue  # skip this op case

        # ---------- outermost: double buffer toggle ----------
        for double_buffer in (False, True):
            ct_limit = (CTILE_MEM_LIMIT // 2) if double_buffer else CTILE_MEM_LIMIT  # 32KB if enabled

            # ---------- numLastSpm: 1..8 ----------
            for numLastSpm in range(1, 8 + 1):
                compute_tiles = 4 * numLastSpm  # total CTs

                # ---- search TM/TK/TN (multiples of 8,8,2) ----
                for TM in range(TM_UNIT, M + 1, TM_UNIT):
                    # early-stop on TM by memory with minimal TK,TN
                    if est_ctile_bytes(TM, TK_UNIT, TN_UNIT) > ct_limit:
                        break

                    if M % TM != 0:
                        continue

                    # number of tiles along M axis
                    n_tiles_M = M // TM
                    # 1) must split evenly across numLastSpm (first stage)
                    if n_tiles_M % numLastSpm != 0:
                        continue
                    # 2) then, tiles assigned to each numLastSpm group must split
                    #    evenly across 4 CTs (total compute_tiles)
                    if n_tiles_M % compute_tiles != 0:
                        continue
                    
                    for TK in range(TK_UNIT, K + 1, TK_UNIT):
                        if est_ctile_bytes(TM, TK, TN_UNIT) > ct_limit:
                            break

                        if K % TK != 0:
                            continue

                        for TN in range(TN_UNIT, N + 1, TN_UNIT):
                            if est_ctile_bytes(TM, TK, TN) > ct_limit:
                                break

                            if N % TN != 0:
                                continue

                            # ---- decide mem-tile size; skip if not feasible ----
                            try:
                                mem_shape = decide_memtile_shape(
                                    M, K, N, TM, TK, TN,
                                    num_ct=compute_tiles,
                                    num_last_spm=numLastSpm,
                                )
                            except Exception as e:
                                print(
                                    f"[WARN] op_case#{idx} (TM,TK,TN)=({TM},{TK},{TN}) "
                                    f"memtile-shape decision raised: {e}; skipping.",
                                    file=sys.stderr,
                                )
                                continue
                            if mem_shape is None:
                                # e.g., cannot split to CTs/memtile constraints failed
                                continue

                            MemTM, MemTK, MemTN = mem_shape

                            # ---- append one tc record ----
                            tc_cases.append({
                                "M": M, "K": K, "N": N,
                                "TM": TM, "TK": TK, "TN": TN,
                                "MemTM": MemTM, "MemTK": MemTK, "MemTN": MemTN,
                                "numLevel": 2,
                                "numLastSpm": numLastSpm,
                                "doubleBuffer": double_buffer
                            })

    return tc_cases


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

    # 1) Read op_list.json -> list of op cases
    try:
        op_cases = load_op_list(op_path)
    except Exception as e:
        print(f"[ERROR] Failed to load op_list: {e}", file=sys.stderr)
        return 2

    print(f"[INFO] Loaded {len(op_cases)} op cases from: {op_path}")

    # 2) Make tc cases from op cases
    tc_cases = make_tc_cases(op_cases)

    print(f"[INFO] Makes {len(tc_cases)} tc cases.")

    # 3) Write tc_list.json
    if args.dry_run:
        print("[DRY-RUN] Would write tc_list.json to:", out_path)
    else:
        try:
            write_tc_list(tc_cases, out_path)
            print(f"[INFO] Wrote tc_list.json to: {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write tc_list.json: {e}", file=sys.stderr)
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
