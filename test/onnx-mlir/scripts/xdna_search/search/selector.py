"""Selector Protocol + EdpSelector.

A Selector receives the flat list of evaluated CostResult items (one per
(candidate, tp_order) pair) and returns the ranking: at most one entry per
candidate, sorted ascending by the selector's primary objective. Moved
from the tail of scripts/generate/cost_model.py::select_optimal without
any behavioral change.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Protocol

from xdna_search.types import CostResult


class Selector(Protocol):
    """Aggregate per-candidate best and order by the primary objective."""

    name: str

    def rank(self, scored: List[CostResult]) -> List[CostResult]: ...


class EdpSelector:
    """Per-candidate argmin-EDP, then global ascending sort by EDP.

    The per-candidate grouping preserves the first-seen order from the
    scored list so that tie-break semantics match the legacy
    select_optimal() (stable sort on EDP).
    """

    name = "edp"

    def rank(self, scored: List[CostResult]) -> List[CostResult]:
        best_by_cand: "OrderedDict[int, CostResult]" = OrderedDict()
        for cr in scored:
            cid = id(cr.candidate)
            cur = best_by_cand.get(cid)
            if cur is None or cr.edp < cur.edp:
                best_by_cand[cid] = cr
        return sorted(best_by_cand.values(), key=lambda r: r.edp)
