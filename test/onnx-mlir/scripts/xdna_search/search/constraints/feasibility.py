"""Feasibility constraints C1-C5.

Logic moved verbatim from scripts/generate/cost_model.py (c1_memory through
c5_mmul_shape). Each constraint is a class implementing the Constraint
Protocol so it can be combined via FilterSet. Order of application is the
same as before: C1 -> C2 -> C3 -> C4 -> C5.
"""

from __future__ import annotations

from xdna_search.hw_constants import MMUL_R, MMUL_S, MMUL_T
from xdna_search.types import Candidate, OpCase, SystemInfo


class C1Memory:
    """Working set must fit in one compute tile's SPM budget."""
    name = "C1: memory"

    def check(self, c: Candidate, op: OpCase, sys_info: SystemInfo) -> bool:
        return c.ws_bytes <= sys_info.ct_usable_bytes


class C2SpatialDiv:
    """Spatial split must evenly divide M and N."""
    name = "C2: spatial divisibility"

    def check(self, c: Candidate, op: OpCase, sys_info: SystemInfo) -> bool:
        return (op.M % c.SPm == 0) and (op.N % c.SPn == 0)


class C3TemporalDiv:
    """Temporal split must evenly divide per-core block sizes."""
    name = "C3: temporal divisibility"

    def check(self, c: Candidate, op: OpCase, sys_info: SystemInfo) -> bool:
        M0 = op.M // c.SPm
        K0 = op.K
        N0 = op.N // c.SPn
        return (M0 % c.TPm == 0) and (K0 % c.TPk == 0) and (N0 % c.TPn == 0)


class C4ColAlign:
    """Core count must be a multiple of comp_tiles_per_col (column power gating)."""
    name = "C4: column alignment"

    def check(self, c: Candidate, op: OpCase, sys_info: SystemInfo) -> bool:
        return c.num_cores % sys_info.comp_tiles_per_col == 0


class C5MmulShape:
    """Tile dimensions must satisfy bf16 mmul<4,8,8> 2x2 expansion alignment."""
    name = "C5: mmul shape"

    def check(self, c: Candidate, op: OpCase, sys_info: SystemInfo) -> bool:
        return (
            (c.TM % (2 * MMUL_R) == 0)
            and (c.TK % MMUL_S == 0)
            and (c.TN % (2 * MMUL_T) == 0)
        )
