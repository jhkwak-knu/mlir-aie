"""Searcher factories + registry.

Every baseline is just a different 4-component injection into the same
Searcher class. Factories receive (sys_info, coeffs, exclude_cores) and
produce a ready-to-run Searcher.

Registered factory names:
  "sm-exh"   STAR-Map w/o pruning = exhaustive enumerate + default feasibility
  "star-map" STAR-Map framework (pruned: Rule 1/2/3)
  "charm"    CHARM-CDSE 1-level (throughput-cycle, SP_k=1)
  "timeloop" Timeloop EDP adapter (stub until Phase 3)
  "naive"    Naive P_max baseline (stub)
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, Optional

from xdna_search.cost.charm import CharmCost, StageMode
from xdna_search.cost.timeloop_edp import TimeloopCoeffs, TimeloopEdpCost
from xdna_search.cost.v16_edp import V16EdpCost
from xdna_search.search.constraints.filter_set import DefaultFeasibility, FilterSet
from xdna_search.search.enumerator import (
    ExhaustiveEnumerator,
    StarMapPrunedEnumerator,
)
from xdna_search.search.searcher import Searcher
from xdna_search.search.selector import (
    CycleSelector,
    EdpSelector,
    TimeloopEdpSelector,
)
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


def make_charm_searcher(
    sys_info: SystemInfo,
    coeffs: Optional[CalibCoeffs] = None,
    *,
    stage_mode: StageMode = "sum",
    exclude_cores: Optional[set] = None,
) -> Searcher:
    """CHARM-CDSE 1-level baseline: min-cycle over broadcast-reuse model.

    SP_k is implicitly 1 because xdna_search's Candidate has no SP_k field
    (the K axis is partitioned only temporally via TP_k). This matches the
    paper/fig CHARM "Variant B" adaptation exactly.
    """
    return Searcher(
        name=f"charm-{stage_mode}",
        enumerator=ExhaustiveEnumerator(),
        filter_set=FilterSet(DefaultFeasibility()),
        cost_fn=CharmCost(stage_mode=stage_mode),
        selector=CycleSelector(),
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
    tl_coeffs: Optional[TimeloopCoeffs] = None,
    exclude_cores: Optional[set] = None,
) -> Searcher:
    """Timeloop EDP adapter: min-EDP under activity-only energy model.

    Uses Timeloop's default goodness metric (EDP, §V-E) and our 3-component
    activity-proportional energy model (§VI-D) WITHOUT a P_BASE term. The
    structural absence of P_BASE is the paper's main claim about Cat B
    frameworks; `tl_coeffs` only tunes magnitudes, not that structural gap.
    """
    return Searcher(
        name="timeloop",
        enumerator=ExhaustiveEnumerator(),
        filter_set=FilterSet(DefaultFeasibility()),
        cost_fn=TimeloopEdpCost(tl_coeffs),
        selector=TimeloopEdpSelector(),
        sys_info=sys_info,
        coeffs=coeffs,
        exclude_cores=exclude_cores,
    )


_REGISTRY: Dict[str, SearcherFactory] = {
    "sm-exh": make_sm_exh_searcher,
    "star-map": make_starmap_searcher,
    "charm": make_charm_searcher,
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
