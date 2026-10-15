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


class CycleSelector:
    """Per-candidate argmin-cycle, then global ascending sort by (cycle, P, TP_total).

    Designed for CHARM-CDSE whose objective is throughput-cycle, not EDP.
    Tie-breaks by increasing `num_cores` and `tp_total` so that simpler
    shapes win among cost-equivalent candidates — matching
    paper/fig/baselines_charm.py::find_charm_cdse's `_charm_key`.
    """

    name = "cycle"

    def rank(self, scored: List[CostResult]) -> List[CostResult]:
        best_by_cand: "OrderedDict[int, CostResult]" = OrderedDict()
        for cr in scored:
            cid = id(cr.candidate)
            cur = best_by_cand.get(cid)
            if cur is None or cr.t_total < cur.t_total:
                best_by_cand[cid] = cr
        return sorted(
            best_by_cand.values(),
            key=lambda r: (
                r.t_total,
                r.candidate.num_cores,
                r.candidate.tp_total,
            ),
        )


class MaxPEdpSelector:
    """Restrict to P == P_max, then per-candidate argmin-EDP + sort by EDP.

    "P_max" is computed from the CostResult list AFTER feasibility filtering,
    so it reflects the maximum feasible num_cores for the specific OpCase.
    This reproduces paper/fig/baselines.py::find_naive_max semantics (pick
    the configuration that uses the maximum feasible number of PEs, then
    minimize EDP among those) without requiring measured data.

    Primary objective remains EDP, and the V16 cost model is shared with
    STAR-Map so the two selectors contrast solely on the P=P_max constraint.
    """

    name = "max-p-edp"

    def rank(self, scored: List[CostResult]) -> List[CostResult]:
        if not scored:
            return []

        p_max = max(cr.candidate.num_cores for cr in scored)
        filtered = [cr for cr in scored if cr.candidate.num_cores == p_max]

        best_by_cand: "OrderedDict[int, CostResult]" = OrderedDict()
        for cr in filtered:
            cid = id(cr.candidate)
            cur = best_by_cand.get(cid)
            if cur is None or cr.edp < cur.edp:
                best_by_cand[cid] = cr
        return sorted(best_by_cand.values(), key=lambda r: r.edp)


class TimeloopEdpSelector:
    """Per-candidate argmin-EDP, then global sort by (EDP, P, TP_total).

    Tie-break matches paper/fig/baselines_timeloop.py::find_timeloop's
    `_timeloop_key`. The (P, TP_total) secondary key is an addition for
    determinism; Timeloop's activity-only energy often ties at fixed
    tile shape, and without a deterministic secondary we would pick by
    enumeration order.
    """

    name = "timeloop-edp"

    def rank(self, scored: List[CostResult]) -> List[CostResult]:
        best_by_cand: "OrderedDict[int, CostResult]" = OrderedDict()
        for cr in scored:
            cid = id(cr.candidate)
            cur = best_by_cand.get(cid)
            if cur is None or cr.edp < cur.edp:
                best_by_cand[cid] = cr
        return sorted(
            best_by_cand.values(),
            key=lambda r: (
                r.edp,
                r.candidate.num_cores,
                r.candidate.tp_total,
            ),
        )
