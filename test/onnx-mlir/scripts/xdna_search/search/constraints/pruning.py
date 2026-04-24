"""STAR-Map pruning rules (promoted from paper/search_bench/baselines.py).

Three rules act as structural pruning on the configuration space:
  Rule 1 (exact):      TP monotonicity - only C1-boundary TP combinations
                       (no single tile dimension can grow further without
                       violating the memory constraint).
  Rule 2 (heuristic):  TP allocation priority - maximize non-inner tiles
                       first, then fill remaining memory with the inner tile.
  Rule 3 (heuristic):  SP pruning - keep only top-k (SPm, SPn) shapes per
                       (P, inner) by the inner-axis-specific N_dma key.

These functions expect aligned "align" parameters that match the existing
C5 mmul-shape constraint: TM % (2*MMUL_R) == 0, TN % (2*MMUL_T) == 0,
TK % MMUL_S == 0.
"""

from __future__ import annotations

from typing import List, Tuple

from xdna_search.hw_constants import (
    MMUL_R,
    MMUL_S,
    MMUL_T,
    TP_AXIS_K,
    TP_AXIS_M,
    TP_AXIS_N,
)


def valid_tile_sizes(dim: int, sp: int, align: int) -> List[int]:
    """All valid tile sizes T = dim/(sp*TP) with T >= align, T % align == 0.

    Returned in descending order (largest tile first), matching the ordering
    used by the original _valid_tile_sizes() in paper/search_bench/baselines.py.
    """
    tiles: List[int] = []
    for tp in range(1, dim // sp + 1):
        if dim % (sp * tp) != 0:
            continue
        t = dim // (sp * tp)
        if t < align or t % align != 0:
            continue
        tiles.append(t)
    return sorted(tiles, reverse=True)


def valid_tile_sizes_k(K: int, align: int) -> List[int]:
    """Valid tile sizes for K (SP_k = 1 implicitly)."""
    tiles: List[int] = []
    for tpk in range(1, K + 1):
        if K % tpk != 0:
            continue
        tk = K // tpk
        if tk < align or tk % align != 0:
            continue
        tiles.append(tk)
    return sorted(tiles, reverse=True)


def _tp_from_tile(dim: int, sp: int, T: int) -> int:
    return dim // (sp * T)


def enumerate_tp_rule1(
    M: int, K: int, N: int,
    SPm: int, SPn: int,
    elem_bytes: int, mem_bytes: int,
) -> List[Tuple[int, int, int]]:
    """Rule 1 (exact): TP monotonicity at the C1 memory boundary.

    Emits (TPm, TPk, TPn) triples for which no single tile dimension can
    be increased without overshooting the memory budget.
    """
    tm_align = 2 * MMUL_R
    tn_align = 2 * MMUL_T
    tk_align = MMUL_S

    TMs = valid_tile_sizes(M, SPm, tm_align)
    TNs = valid_tile_sizes(N, SPn, tn_align)
    TKs = valid_tile_sizes_k(K, tk_align)

    results: List[Tuple[int, int, int]] = []
    for TM in TMs:
        for TN in TNs:
            for TK in TKs:
                tile_bytes = elem_bytes * (TM * TK + TK * TN + TM * TN)
                if tile_bytes > mem_bytes:
                    continue

                at_boundary = True
                idx_m = TMs.index(TM)
                if idx_m > 0:
                    TM_next = TMs[idx_m - 1]
                    if elem_bytes * (TM_next * TK + TK * TN + TM_next * TN) <= mem_bytes:
                        at_boundary = False
                        continue

                idx_n = TNs.index(TN)
                if idx_n > 0:
                    TN_next = TNs[idx_n - 1]
                    if elem_bytes * (TM * TK + TK * TN_next + TM * TN_next) <= mem_bytes:
                        at_boundary = False
                        continue

                idx_k = TKs.index(TK)
                if idx_k > 0:
                    TK_next = TKs[idx_k - 1]
                    if elem_bytes * (TM * TK_next + TK_next * TN + TM * TN) <= mem_bytes:
                        at_boundary = False
                        continue

                if at_boundary:
                    results.append(
                        (_tp_from_tile(M, SPm, TM),
                         K // TK,
                         _tp_from_tile(N, SPn, TN))
                    )
    return results


def enumerate_tp_rule12(
    M: int, K: int, N: int,
    SPm: int, SPn: int, inner: int,
    elem_bytes: int, mem_bytes: int,
) -> List[Tuple[int, int, int]]:
    """Rule 1 + Rule 2: monotonicity + inner-axis allocation priority."""
    tm_align = 2 * MMUL_R
    tn_align = 2 * MMUL_T
    tk_align = MMUL_S

    TMs = valid_tile_sizes(M, SPm, tm_align)
    TNs = valid_tile_sizes(N, SPn, tn_align)
    TKs = valid_tile_sizes_k(K, tk_align)

    results: List[Tuple[int, int, int]] = []

    if inner == TP_AXIS_K:
        for TM in TMs:
            for TN in TNs:
                can_grow_m = False
                idx_m = TMs.index(TM)
                if idx_m > 0:
                    TM_next = TMs[idx_m - 1]
                    min_TK = TKs[-1] if TKs else None
                    if min_TK and elem_bytes * (TM_next * min_TK + min_TK * TN + TM_next * TN) <= mem_bytes:
                        can_grow_m = True
                can_grow_n = False
                idx_n = TNs.index(TN)
                if idx_n > 0:
                    TN_next = TNs[idx_n - 1]
                    min_TK = TKs[-1] if TKs else None
                    if min_TK and elem_bytes * (TM * min_TK + min_TK * TN_next + TM * TN_next) <= mem_bytes:
                        can_grow_n = True
                if can_grow_m or can_grow_n:
                    continue
                best_TK = None
                for TK in TKs:
                    if elem_bytes * (TM * TK + TK * TN + TM * TN) <= mem_bytes:
                        best_TK = TK
                        break
                if best_TK is not None:
                    results.append(
                        (_tp_from_tile(M, SPm, TM),
                         K // best_TK,
                         _tp_from_tile(N, SPn, TN))
                    )

    elif inner == TP_AXIS_M:
        for TN in TNs:
            for TK in TKs:
                can_grow_n = False
                idx_n = TNs.index(TN)
                if idx_n > 0:
                    TN_next = TNs[idx_n - 1]
                    min_TM = TMs[-1] if TMs else None
                    if min_TM and elem_bytes * (min_TM * TK + TK * TN_next + min_TM * TN_next) <= mem_bytes:
                        can_grow_n = True
                can_grow_k = False
                idx_k = TKs.index(TK)
                if idx_k > 0:
                    TK_next = TKs[idx_k - 1]
                    min_TM = TMs[-1] if TMs else None
                    if min_TM and elem_bytes * (min_TM * TK_next + TK_next * TN + min_TM * TN) <= mem_bytes:
                        can_grow_k = True
                if can_grow_n or can_grow_k:
                    continue
                best_TM = None
                for TM in TMs:
                    if elem_bytes * (TM * TK + TK * TN + TM * TN) <= mem_bytes:
                        best_TM = TM
                        break
                if best_TM is not None:
                    results.append(
                        (_tp_from_tile(M, SPm, best_TM),
                         K // TK,
                         _tp_from_tile(N, SPn, TN))
                    )

    elif inner == TP_AXIS_N:
        for TM in TMs:
            for TK in TKs:
                can_grow_m = False
                idx_m = TMs.index(TM)
                if idx_m > 0:
                    TM_next = TMs[idx_m - 1]
                    min_TN = TNs[-1] if TNs else None
                    if min_TN and elem_bytes * (TM_next * TK + TK * min_TN + TM_next * min_TN) <= mem_bytes:
                        can_grow_m = True
                can_grow_k = False
                idx_k = TKs.index(TK)
                if idx_k > 0:
                    TK_next = TKs[idx_k - 1]
                    min_TN = TNs[-1] if TNs else None
                    if min_TN and elem_bytes * (TM * TK_next + TK_next * min_TN + TM * min_TN) <= mem_bytes:
                        can_grow_k = True
                if can_grow_m or can_grow_k:
                    continue
                best_TN = None
                for TN in TNs:
                    if elem_bytes * (TM * TK + TK * TN + TM * TN) <= mem_bytes:
                        best_TN = TN
                        break
                if best_TN is not None:
                    results.append(
                        (_tp_from_tile(M, SPm, TM),
                         K // TK,
                         _tp_from_tile(N, SPn, best_TN))
                    )
    return results


def sp_shape_key(inner: int, SPm: int, SPn: int) -> int:
    """Rule 3 ranking key: minimize N_dma per inner axis.

    K-inner  (balanced SP is best):     SPm + SPn
    M-inner  (minimize SPm contribution): SPm + 2 * SPm * SPn
    N-inner  (minimize SPn contribution): SPn + 2 * SPm * SPn
    """
    if inner == TP_AXIS_K:
        return SPm + SPn
    if inner == TP_AXIS_M:
        return SPm + 2 * SPm * SPn
    return SPn + 2 * SPm * SPn
