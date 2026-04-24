"""Pure math utilities used by enumeration, filtering, and ranking.

divisors / factor_pairs / ws_bytes / build_tp_order /
spearman_rank_correlation. Moved from scripts/generate/tiling_common.py
without any change.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from xdna_search.hw_constants import TP_AXIS_K, TP_AXIS_M, TP_AXIS_N


def divisors(n: int) -> List[int]:
    """All positive divisors of n, sorted ascending."""
    ds = set()
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            ds.add(d)
            ds.add(n // d)
    return sorted(ds)


def factor_pairs(n: int) -> List[Tuple[int, int]]:
    """All (a, b) pairs with a * b == n, sorted by a."""
    return [(d, n // d) for d in divisors(n)]


def ws_bytes(TM: int, TK: int, TN: int, elem_bytes: int) -> int:
    """Working-set size for one compute tile: A(TM*TK) + B(TK*TN) + C(TM*TN)."""
    return elem_bytes * (TM * TK + TK * TN + TM * TN)


def build_tp_order(winner_axis: int) -> List[int]:
    """Build full tpOrder from the innermost (winner) axis.

    Returns [winner, middle, outermost] where remaining axes follow the
    K > M > N priority used by the existing pipeline.
    """
    default_order = [TP_AXIS_K, TP_AXIS_M, TP_AXIS_N]
    return [winner_axis] + [ax for ax in default_order if ax != winner_axis]


def spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation coefficient, averaging ranks for ties."""
    n = len(x)
    if n < 2:
        return float("nan")

    def _rank(vals: List[float]) -> List[float]:
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = sum((a - mean_rx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_ry) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)
