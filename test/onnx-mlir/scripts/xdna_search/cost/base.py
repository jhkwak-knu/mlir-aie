"""CostFunction Protocol.

Every cost function (v16 EDP, future Timeloop/CHARM-CDSE adapters, etc.)
returns a CostResult. The total fields (t_total, e_total, edp) are required;
breakdown fields (t_comp/t_comm/t_overhead/e_dynamic_*/e_static) may be NaN
when the underlying model does not decompose them.
"""

from __future__ import annotations

from typing import Optional, Protocol

from xdna_search.types import CalibCoeffs, Candidate, CostResult, OpCase


class CostFunction(Protocol):
    """Evaluate a (candidate, tpOrder) pair under a given calibration."""

    name: str

    def evaluate(
        self,
        op: OpCase,
        candidate: Candidate,
        tp_order: int,
        coeffs: Optional[CalibCoeffs] = None,
    ) -> CostResult: ...
