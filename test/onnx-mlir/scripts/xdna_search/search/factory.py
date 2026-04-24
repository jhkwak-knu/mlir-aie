"""Searcher factories + registry.

Every baseline is just a different 4-component injection into the same
Searcher class. Factories receive (sys_info, coeffs, exclude_cores) and
produce a ready-to-run Searcher.

Registered factory names:
  "sm-exh"   STAR-Map w/o pruning = exhaustive enumerate + default feasibility
  "star-map" STAR-Map framework (stub until Step 9)
  "naive"    Naive P_max baseline (stub, Step 10)
  "timeloop" Timeloop / CHARM-CDSE adapter (stub, Step 10)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from xdna_search.cost.v16_edp import V16EdpCost
from xdna_search.search.constraints.filter_set import DefaultFeasibility, FilterSet
from xdna_search.search.enumerator import (
    ExhaustiveEnumerator,
    StarMapPrunedEnumerator,
)
from xdna_search.search.searcher import Searcher
from xdna_search.search.selector import EdpSelector
from xdna_search.types import CalibCoeffs, SystemInfo


SearcherFactory = Callable[..., Searcher]


def make_sm_exh_searcher(
    sys_info: SystemInfo,
    coeffs: Optional[CalibCoeffs] = None,
    *,
    exclude_cores: Optional[set] = None,
) -> Searcher:
    """STAR-Map w/o pruning: exhaustive enumerate + default feasibility."""
    return Searcher(
        name="sm-exh",
        enumerator=ExhaustiveEnumerator(),
        filter_set=FilterSet(DefaultFeasibility()),
        cost_fn=V16EdpCost(),
        selector=EdpSelector(),
        sys_info=sys_info,
        coeffs=coeffs,
        exclude_cores=exclude_cores,
    )


def make_starmap_searcher(
    sys_info: SystemInfo,
    coeffs: Optional[CalibCoeffs] = None,
    *,
    pruning_level: int = 123,
    exclude_cores: Optional[set] = None,
) -> Searcher:
    """STAR-Map framework: pruned enumerate + default feasibility.

    pruning_level selects which STAR-Map rules are applied during enumeration:
        1   = Rule 1 only
        12  = Rule 1 + Rule 2
        123 = Rule 1 + Rule 2 + Rule 3 (paper default)

    Every pruned candidate already commits to an inner axis, so the Searcher
    evaluates only that single tpOrder per candidate.
    """
    return Searcher(
        name=f"star-map-rule{pruning_level}",
        enumerator=StarMapPrunedEnumerator(pruning_level=pruning_level),
        filter_set=FilterSet(DefaultFeasibility()),
        cost_fn=V16EdpCost(),
        selector=EdpSelector(),
        sys_info=sys_info,
        coeffs=coeffs,
        exclude_cores=exclude_cores,
    )


def make_naive_searcher(
    sys_info: SystemInfo,
    coeffs: Optional[CalibCoeffs] = None,
    *,
    exclude_cores: Optional[set] = None,
) -> Searcher:
    """Naive (P_max / vendor-default) baseline. Not yet implemented."""
    raise NotImplementedError("make_naive_searcher: Naive P_max baseline is a stub")


def make_timeloop_searcher(
    sys_info: SystemInfo,
    coeffs: Optional[CalibCoeffs] = None,
    *,
    exclude_cores: Optional[set] = None,
) -> Searcher:
    """Timeloop / CHARM-CDSE baseline adapter. Not yet implemented."""
    raise NotImplementedError(
        "make_timeloop_searcher: Timeloop / CHARM-CDSE adapter is a stub"
    )


_REGISTRY: Dict[str, SearcherFactory] = {
    "sm-exh": make_sm_exh_searcher,
    "star-map": make_starmap_searcher,
    "naive": make_naive_searcher,
    "timeloop": make_timeloop_searcher,
}


def get_searcher_factory(name: str) -> SearcherFactory:
    """Look up a factory by searcher name. Raises ValueError for unknown names."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown searcher: {name!r} (choices: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[name]


def available_searchers() -> list:
    """Return the list of registered searcher names."""
    return sorted(_REGISTRY.keys())
