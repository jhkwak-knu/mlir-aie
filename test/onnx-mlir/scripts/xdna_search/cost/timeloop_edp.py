"""Timeloop EDP cost (activity-proportional, no P_BASE).

Reference:
    A. Parashar et al., "Timeloop: A Systematic Approach to DNN
    Accelerator Evaluation," ISPASS 2019.

Ported from paper/fig/baselines_timeloop.py without numerical change.

Performance model (MAX → SUM adaptation for XDNA2 sequential execution):

    T_DRAM = total_data_bytes / bw_eff_bpc            [cycles]
    T_MAC  = (M * N * K) / (P * eff_macs)             [cycles]
    T_L1   = 0                                        (non-bottlenecking at bf16)
    T      = T_DRAM + T_MAC

Energy model (activity-proportional, 3-component, NO P_BASE):

    E_DRAM = e_dram * total_data_bytes                [pJ]
    E_L1   = e_l1   * 4 * MACs * elem_bytes           [pJ]
    E_MAC  = e_mac  * MACs                            [pJ]
    E_uJ   = (E_DRAM + E_L1 + E_MAC) * 1e-6           [uJ]

    EDP = T (cycles) * E_uJ                           [cycles * uJ]

Unit matches paper/fig exactly; differs from V16EdpCost which returns
EDP in cycles * pJ. Absolute numeric values are thus 1e6 smaller than
V16EdpCost's, but argmin ordering is invariant to this global scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from xdna_search import models as _models
from xdna_search.cost.v16_edp import BANDWIDTH_BPC, PEAK_MACS
from xdna_search.types import CalibCoeffs, Candidate, CostResult, OpCase


@dataclass
class TimeloopCoeffs:
    """Per-access energy coefficients at 16 nm (Accelergy-class estimates, pJ).

    Defaults match paper/fig/baselines_timeloop.py::TimeloopCoeffs. Values
    are round-number estimates consistent with published Accelergy / CACTI
    16 nm figures. Treat as engineering estimates, not citation-ready
    Accelergy defaults.
    """
    e_dram_pJ_per_byte: float = 40.0
    e_l1_pJ_per_byte: float = 2.0
    e_mac_pJ: float = 0.5


class TimeloopEdpCost:
    """CostFunction: Timeloop EDP (activity-only energy).

    The structural claim this model makes (that Timeloop / Category B
    frameworks lack a P_BASE term) is invariant to coefficient values.
    """

    name = "timeloop-edp"

    def __init__(self, tcoef: Optional[TimeloopCoeffs] = None) -> None:
        self.tcoef = tcoef if tcoef is not None else TimeloopCoeffs()

    def evaluate(
        self,
        op: OpCase,
        candidate: Candidate,
        tp_order: int,
        coeffs: Optional[CalibCoeffs] = None,
    ) -> CostResult:
        # Calibrated eff_macs / bw take precedence over library defaults.
        eff_macs = coeffs.eff_macs if coeffs else PEAK_MACS
        bw = coeffs.bw_eff_bpc if coeffs else BANDWIDTH_BPC

        dram_bytes = _models.total_data_bytes(
            op.M, op.K, op.N, op.elem_bytes,
            candidate.SPm, candidate.SPn,
            candidate.TPm, candidate.TPk, candidate.TPn,
            tp_order,
        )
        macs = op.M * op.N * op.K
        l1_bytes = 4 * macs * op.elem_bytes

        t_dram_cy = dram_bytes / bw
        t_mac_cy = macs / (candidate.num_cores * eff_macs)
        t_total = t_dram_cy + t_mac_cy  # sum-aggregation (T_L1=0)

        e_dram_pJ = self.tcoef.e_dram_pJ_per_byte * dram_bytes
        e_l1_pJ = self.tcoef.e_l1_pJ_per_byte * l1_bytes
        e_mac_pJ = self.tcoef.e_mac_pJ * macs
        # Convert pJ -> uJ to match paper/fig's edp scale (cycles * uJ).
        e_total_uJ = (e_dram_pJ + e_l1_pJ + e_mac_pJ) * 1e-6

        return CostResult(
            candidate=candidate, tp_order=tp_order,
            t_comp=t_mac_cy,
            t_comm=t_dram_cy,
            t_overhead=0.0,
            t_total=t_total,
            e_dynamic_comp=e_mac_pJ * 1e-6,
            e_dynamic_comm=e_dram_pJ * 1e-6,
            # Stash L1 in the "static" slot for reporting even though L1
            # here is activity-proportional, not leakage.
            e_static=e_l1_pJ * 1e-6,
            e_total=e_total_uJ,
            edp=t_total * e_total_uJ,
        )
