"""Enumerator Protocol + ExhaustiveEnumerator.

ExhaustiveEnumerator contains the candidate enumeration loop moved verbatim
from scripts/generate/cost_model.py::enumerate_candidates (no logic change).
"""

from __future__ import annotations

import math
from typing import List, Optional, Protocol

from xdna_search.hw_constants import TP_AXIS_K, TP_AXIS_M, TP_AXIS_N
from xdna_search.math_utils import divisors, factor_pairs, ws_bytes
from xdna_search.search.constraints.pruning import (
    enumerate_tp_rule1,
    enumerate_tp_rule12,
    sp_shape_key,
)
from xdna_search.types import Candidate, OpCase, SystemInfo


class Enumerator(Protocol):
    """Produce the candidate set for a single op."""

    name: str

    def generate(
        self,
        op: OpCase,
        sys_info: SystemInfo,
        exclude_cores: Optional[set] = None,
    ) -> List[Candidate]: ...


class ExhaustiveEnumerator:
    """Enumerate every (num_cores, SPm, SPn, TPm, TPk, TPn) combination.

    Loop hierarchy (resource usage as the primary independent variable):
      1. num_cores in [1, C]         -- resource budget (PE count)
      2. SPm in [1, num_cores]        -- spatial M-axis split
         SPn = num_cores / SPm        (determined, not a free variable)
      3. TPm, TPn, TPk over divisors of M, N, K respectively.

    Loop order matches paper/fig/baselines.py::enumerate_configs
    (TPm -> TPn -> TPk) so tie-break results from stable sorts are
    identical to the paper reference implementation.

    Hardware constraints are deferred to Stage 2 (filters). Tile sizes
    (TM, TK, TN) are 0 whenever the split does not evenly divide the
    corresponding dimension so that constraint C3 removes them cleanly.
    """

    name = "exhaustive"

    def generate(
        self,
        op: OpCase,
        sys_info: SystemInfo,
        exclude_cores: Optional[set] = None,
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        eb = op.elem_bytes

        divs_M = divisors(op.M)
        divs_K = divisors(op.K)
        divs_N = divisors(op.N)

        for num_cores in range(1, sys_info.total_cores + 1):
            if exclude_cores and num_cores in exclude_cores:
                continue
            for SPm, SPn in factor_pairs(num_cores):
                for TPm in divs_M:
                    for TPn in divs_N:
                        for TPk in divs_K:
                            sp_tp_m = SPm * TPm
                            sp_tp_n = SPn * TPn
                            TM = op.M // sp_tp_m if op.M % sp_tp_m == 0 else 0
                            TK = op.K // TPk
                            TN = op.N // sp_tp_n if op.N % sp_tp_n == 0 else 0
                            ws = (
                                ws_bytes(TM, TK, TN, eb)
                                if (TM > 0 and TN > 0)
                                else 0
                            )
                            num_cols = math.ceil(
                                num_cores / sys_info.comp_tiles_per_col
                            )

                            candidates.append(
                                Candidate(
                                    num_cores=num_cores,
                                    num_columns=num_cols,
                                    SPm=SPm, SPn=SPn,
                                    TPm=TPm, TPk=TPk, TPn=TPn,
                                    TM=TM, TK=TK, TN=TN,
                                    ws_bytes=ws,
                                )
                            )

        return candidates


class StarMapPrunedEnumerator:
    """STAR-Map pruning: Rule 1 (exact) + Rule 2/3 (heuristic) from the paper.

    Produces one Candidate per surviving (P, SPm, SPn, TPm, TPk, TPn, inner)
    combination. Each Candidate carries `inner_axis` so the Searcher evaluates
    only that single tpOrder (not all three); pruning rules commit to an inner
    axis by construction.

    pruning_level:
        1   Rule 1 only (monotonicity)
        12  Rule 1 + Rule 2 (inner-axis allocation priority)
        123 Rule 1 + Rule 2 + Rule 3 (top-k SP per inner)   -- paper default
    """

    def __init__(self, pruning_level: int = 123, top_k_sp: int = 3):
        if pruning_level not in (1, 12, 123):
            raise ValueError(
                f"pruning_level must be 1, 12, or 123; got {pruning_level}"
            )
        self.pruning_level = pruning_level
        self.top_k_sp = top_k_sp
        self.name = f"star-map-pruned-rule{pruning_level}"

    def generate(
        self,
        op: OpCase,
        sys_info: SystemInfo,
        exclude_cores: Optional[set] = None,
    ) -> List[Candidate]:
        apply_rule2 = self.pruning_level >= 12
        apply_rule3 = self.pruning_level >= 123
        elem_bytes = op.elem_bytes
        mem_bytes = sys_info.ct_usable_bytes

        candidates: List[Candidate] = []

        for num_cores in range(1, sys_info.total_cores + 1):
            if exclude_cores and num_cores in exclude_cores:
                continue

            sp_pairs = factor_pairs(num_cores)
            num_cols = math.ceil(num_cores / sys_info.comp_tiles_per_col)

            # Rule 3: restrict (SPm, SPn) to top-k per inner, ranked by the
            # inner-specific N_dma key. Without Rule 3, every SP pair is kept.
            sp_keep = {}
            for inner in (TP_AXIS_M, TP_AXIS_N, TP_AXIS_K):
                if apply_rule3:
                    sorted_sp = sorted(
                        sp_pairs, key=lambda pr, _i=inner: sp_shape_key(_i, pr[0], pr[1])
                    )
                    sp_keep[inner] = set(sorted_sp[: self.top_k_sp])
                else:
                    sp_keep[inner] = set(sp_pairs)

            for SPm, SPn in sp_pairs:
                for inner in (TP_AXIS_M, TP_AXIS_N, TP_AXIS_K):
                    if (SPm, SPn) not in sp_keep[inner]:
                        continue

                    if apply_rule2:
                        tp_list = enumerate_tp_rule12(
                            op.M, op.K, op.N, SPm, SPn, inner,
                            elem_bytes, mem_bytes,
                        )
                    else:
                        tp_list = enumerate_tp_rule1(
                            op.M, op.K, op.N, SPm, SPn,
                            elem_bytes, mem_bytes,
                        )

                    for TPm, TPk, TPn in tp_list:
                        TM = op.M // (SPm * TPm)
                        TK = op.K // TPk
                        TN = op.N // (SPn * TPn)
                        candidates.append(
                            Candidate(
                                num_cores=num_cores,
                                num_columns=num_cols,
                                SPm=SPm, SPn=SPn,
                                TPm=TPm, TPk=TPk, TPn=TPn,
                                TM=TM, TK=TK, TN=TN,
                                ws_bytes=ws_bytes(TM, TK, TN, elem_bytes),
                                inner_axis=inner,
                            )
                        )

        return candidates
