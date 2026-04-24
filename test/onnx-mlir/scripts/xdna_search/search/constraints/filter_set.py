"""FilterSet — ordered list of Constraints applied sequentially.

Yields per-constraint statistics (FilterResult) so the caller can report
how many candidates each constraint removed.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from xdna_search.search.constraints.base import Constraint
from xdna_search.search.constraints.feasibility import (
    C1Memory,
    C2SpatialDiv,
    C3TemporalDiv,
    C4ColAlign,
    C5MmulShape,
)
from xdna_search.types import Candidate, FilterResult, OpCase, SystemInfo


class FilterSet:
    """Ordered composition of Constraint objects.

    FilterSet(constraints).apply(cands, op, sys_info) -> (valid, results)
    returns only the candidates that pass every constraint, along with the
    per-constraint before/after counts used for diagnostics (matches the
    output of the former filter_candidates() function).
    """

    def __init__(self, constraints: Iterable[Constraint]):
        self.constraints: List[Constraint] = list(constraints)

    def apply(
        self,
        candidates: List[Candidate],
        op: OpCase,
        sys_info: SystemInfo,
    ) -> Tuple[List[Candidate], List[FilterResult]]:
        results: List[FilterResult] = []
        current = candidates
        for cons in self.constraints:
            before = len(current)
            current = [c for c in current if cons.check(c, op, sys_info)]
            results.append(FilterResult(name=cons.name, before=before, after=len(current)))
        return current, results


def DefaultFeasibility() -> List[Constraint]:
    """Canonical C1-C5 feasibility constraint sequence (same order as the legacy filter_candidates())."""
    return [C1Memory(), C2SpatialDiv(), C3TemporalDiv(), C4ColAlign(), C5MmulShape()]
