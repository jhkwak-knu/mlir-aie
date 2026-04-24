"""Semantic tests for the Max-P (naive P_max) searcher.

Verifies:
  1. Factory registration ("max-p" key resolves to make_max_p_searcher).
  2. MaxPEdpSelector restricts to CostResults whose num_cores == P_max.
  3. The returned best candidate uses P == P_max (feasible maximum).
  4. Returned candidate is the argmin-EDP within the P_max group.
  5. Search on an empty scored list returns an empty ranking (defensive).
  6. "naive" key is no longer registered (stub removal).

Uses a tiny (M, K, N) so exhaustive enumeration stays cheap.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from xdna_search.search.factory import (
    available_searchers,
    get_searcher_factory,
)
from xdna_search.search.selector import EdpSelector, MaxPEdpSelector
from xdna_search.types import (
    Candidate,
    CostResult,
    DEFAULT_COEFFS,
    OpCase,
    SystemInfo,
)


# A modest XDNA2 shape that still exercises the P axis (8 total cores is
# enough to see P_max != 1 while keeping candidate enumeration small).
TEST_SYS_INFO = SystemInfo(
    total_cores=8,
    comp_tiles_per_col=4,
    max_columns=8,
    spm_size_bytes=65536,
    mem_tile_mem_bytes=524288,
)

# 32x64x32 bf16: exercises SPm*SPn in [1,8], divisor-rich so feasibility
# does not reject the entire space.
TEST_OP = OpCase(M=32, K=64, N=32, elem_type="bf16")


def _make_searcher(name: str):
    factory = get_searcher_factory(name)
    return factory(TEST_SYS_INFO, DEFAULT_COEFFS)


def test_factory_registers_max_p():
    """'max-p' is registered and resolves to a working Searcher."""
    assert "max-p" in available_searchers()
    searcher = _make_searcher("max-p")
    assert searcher.name == "max-p"


def test_naive_key_removed():
    """'naive' key must no longer be in the registry (stub removed)."""
    assert "naive" not in available_searchers()
    with pytest.raises(ValueError, match="unknown searcher"):
        get_searcher_factory("naive")


def test_max_p_returns_p_max_config():
    """Best candidate uses the maximum feasible num_cores."""
    searcher = _make_searcher("max-p")
    output = searcher.search(TEST_OP)

    assert output.ranked, "ranked output should be non-empty for 32x64x32"
    # P_max is defined over feasibility-passing candidates (same set the
    # selector sees), so it equals max over output.valid.
    p_max = max(c.num_cores for c in output.valid)
    best = output.ranked[0]
    assert best.candidate.num_cores == p_max


def test_max_p_differs_from_star_map_on_p_choice():
    """Max-P and STAR-Map agree on cost model but can disagree on P.

    The only source of divergence is the P=P_max filter. If both setters
    happen to pick P_max on this shape, we still exercise the code paths —
    but we assert that Max-P NEVER picks a P below P_max.
    """
    max_p = _make_searcher("max-p").search(TEST_OP)
    star_map = _make_searcher("sm-exh").search(TEST_OP)

    p_max = max(c.num_cores for c in max_p.valid)
    assert max_p.ranked[0].candidate.num_cores == p_max
    assert star_map.ranked[0].candidate.num_cores <= p_max


def test_max_p_is_argmin_edp_within_p_max_group():
    """Within P == P_max, returned candidate has the minimum EDP."""
    searcher = _make_searcher("max-p")
    output = searcher.search(TEST_OP)

    p_max = max(c.num_cores for c in output.valid)
    p_max_group = [r for r in output.ranked if r.candidate.num_cores == p_max]
    # All entries in ranked must be in the P_max group (selector filter).
    assert len(p_max_group) == len(output.ranked)

    min_edp = min(r.edp for r in p_max_group)
    assert output.ranked[0].edp == pytest.approx(min_edp)


def test_max_p_selector_on_empty_scored_list():
    """Defensive: empty scored input yields empty ranking, no exception."""
    selector = MaxPEdpSelector()
    assert selector.rank([]) == []


def _fake_cost_result(num_cores: int, edp: float) -> CostResult:
    """Hand-rolled CostResult for selector unit tests (bypasses cost fn)."""
    cand = Candidate(
        num_cores=num_cores,
        num_columns=1,
        SPm=num_cores, SPn=1,
        TPm=1, TPk=1, TPn=1,
        TM=1, TK=1, TN=1,
        ws_bytes=0,
    )
    return CostResult(
        candidate=cand,
        tp_order=0,
        t_comp=0.0, t_comm=0.0, t_overhead=0.0, t_total=0.0,
        e_dynamic_comp=0.0, e_dynamic_comm=0.0, e_static=0.0, e_total=0.0,
        edp=edp,
    )


def test_max_p_selector_filters_and_sorts_by_edp():
    """Synthetic CostResults: selector drops below-P_max entries, sorts by EDP."""
    selector = MaxPEdpSelector()
    scored = [
        _fake_cost_result(num_cores=2, edp=10.0),   # below P_max -> dropped
        _fake_cost_result(num_cores=4, edp=50.0),   # P_max, higher EDP
        _fake_cost_result(num_cores=4, edp=30.0),   # P_max, lowest EDP
        _fake_cost_result(num_cores=4, edp=40.0),   # P_max, mid
        _fake_cost_result(num_cores=1, edp=1.0),    # lowest EDP overall, but dropped
    ]
    ranked = selector.rank(scored)
    assert [r.candidate.num_cores for r in ranked] == [4, 4, 4]
    assert [r.edp for r in ranked] == [30.0, 40.0, 50.0]
