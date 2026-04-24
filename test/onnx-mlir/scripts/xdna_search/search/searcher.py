"""Searcher — concrete 4-component pipeline executor.

Composes (Enumerator, FilterSet, CostFunction, Selector) into a single
run: enumerate candidates, apply constraints, evaluate every surviving
candidate across the three innermost temporal axis choices, then rank.

Every baseline (STAR-Map, SM-exh, Max-P, Timeloop, ...) differs only in
which components get plugged in; see xdna_search.search.factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from xdna_search.cost.base import CostFunction
from xdna_search.search.constraints.filter_set import FilterSet
from xdna_search.search.enumerator import Enumerator
from xdna_search.search.selector import Selector
from xdna_search.types import (
    CalibCoeffs,
    Candidate,
    CostResult,
    FilterResult,
    OpCase,
    SystemInfo,
)

# Inner temporal axis choices (0=M, 1=N, 2=K) swept for each candidate.
TP_ORDER_CANDIDATES = (0, 1, 2)


@dataclass
class SearchOutput:
    """Per-op search result, including diagnostic counters.

    - ranked: selector output, one CostResult per valid candidate (best
      tp_order), sorted ascending by the selector's primary objective.
    - valid: candidates surviving every constraint.
    - filter_results: per-constraint before/after counts.
    - total_enumerated: size of the raw enumeration.
    - all_tporder_data: populated only when search(keep_all_tporders=True);
      maps valid-candidate index to the full 3-tpOrder CostResult list.
    """

    ranked: List[CostResult]
    valid: List[Candidate]
    filter_results: List[FilterResult]
    total_enumerated: int
    all_tporder_data: Optional[dict] = None


class Searcher:
    """4-component concrete searcher: enumerate -> filter -> cost -> rank."""

    def __init__(
        self,
        name: str,
        enumerator: Enumerator,
        filter_set: FilterSet,
        cost_fn: CostFunction,
        selector: Selector,
        *,
        sys_info: SystemInfo,
        coeffs: Optional[CalibCoeffs] = None,
        exclude_cores: Optional[set] = None,
    ) -> None:
        self.name = name
        self.enumerator = enumerator
        self.filter_set = filter_set
        self.cost_fn = cost_fn
        self.selector = selector
        self.sys_info = sys_info
        self.coeffs = coeffs
        self.exclude_cores = exclude_cores

    def search(self, op: OpCase, *, keep_all_tporders: bool = False) -> SearchOutput:
        cands = self.enumerator.generate(
            op, self.sys_info, exclude_cores=self.exclude_cores
        )
        valid, filter_results = self.filter_set.apply(cands, op, self.sys_info)

        scored: List[CostResult] = []
        all_tporder_data = {} if keep_all_tporders else None
        for i, c in enumerate(valid):
            # Candidates with a committed inner_axis (e.g. STAR-Map pruned
            # outputs) are evaluated only at that tpOrder; otherwise sweep
            # all three.
            tporders = (
                (c.inner_axis,) if c.inner_axis is not None
                else TP_ORDER_CANDIDATES
            )
            results = [
                self.cost_fn.evaluate(op, c, tpo, self.coeffs) for tpo in tporders
            ]
            scored.extend(results)
            if all_tporder_data is not None:
                all_tporder_data[i] = results

        ranked = self.selector.rank(scored)
        return SearchOutput(
            ranked=ranked,
            valid=valid,
            filter_results=filter_results,
            total_enumerated=len(cands),
            all_tporder_data=all_tporder_data,
        )

    def search_many(
        self, ops: Iterable[OpCase], *, keep_all_tporders: bool = False,
    ) -> List[SearchOutput]:
        return [self.search(op, keep_all_tporders=keep_all_tporders) for op in ops]
