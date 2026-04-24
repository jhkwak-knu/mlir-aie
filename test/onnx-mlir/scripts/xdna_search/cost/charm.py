"""CHARM-CDSE throughput-cycle cost (1-level adaptation for XDNA2).

Reference:
    Jinming Zhuang et al., "CHARM: Composing Heterogeneous AcceleRators
    for Matrix Multiply on Versal ACAP Architecture," FPGA 2023.

Ported from paper/fig/baselines_charm.py::charm_cycle_cost without
numerical change. Only the interface is adapted to xdna_search's
CostFunction Protocol.

CHARM's objective is *throughput-cycle*, not EDP. The total cycle count is
stored in CostResult.t_total and also copied to CostResult.edp so that
downstream selectors (CycleSelector, EdpSelector) can rank by either field
without knowing which cost model produced the result. Energy fields are
NaN because CHARM is cycle-only.

Broadcast-reuse bytes (preserved from the original CHARM paper):

    load_A_iter  = SP_m * TM * TK * elem_bytes     (A broadcast across SP_n)
    load_B_iter  = SP_n * TK * TN * elem_bytes     (B broadcast across SP_m)
    store_C_iter = P    * TM * TN * elem_bytes     (C unique per core)

Per-iteration cycles:

    compute_cyc = TM * TK * TN / eff_macs          (parallel across cores)
    load_cyc    = (load_A_iter + load_B_iter) / bw_eff_bpc
    store_cyc   = store_C_iter / bw_eff_bpc

Per-iteration aggregation (stage_mode):

    "sum" (default)   : aie_cycle = compute_cyc + load_cyc + store_cyc
    "max"             : aie_cycle = max(compute_cyc, load_cyc, store_cyc)

Total cycles:

    total_cycle = aie_cycle * TP_total             (TP_m * TP_k * TP_n)
"""
from __future__ import annotations

import math
from typing import Literal, Optional

from xdna_search.cost.v16_edp import BANDWIDTH_BPC, PEAK_MACS
from xdna_search.types import CalibCoeffs, Candidate, CostResult, OpCase


StageMode = Literal["sum", "max"]


class CharmCost:
    """CHARM-CDSE throughput-cycle objective.

    Variant B: SP_k is implicitly 1 (xdna_search's Candidate has no SP_k
    field; the K axis is only partitioned temporally via TP_k). This matches
    the paper/fig CHARM adaptation exactly.
    """

    def __init__(self, stage_mode: StageMode = "sum") -> None:
        if stage_mode not in ("sum", "max"):
            raise ValueError(f"Unknown stage_mode: {stage_mode!r}")
        self.stage_mode = stage_mode
        self.name = f"charm-{stage_mode}"

    def evaluate(
        self,
        op: OpCase,
        candidate: Candidate,
        tp_order: int,
        coeffs: Optional[CalibCoeffs] = None,
    ) -> CostResult:
        # Calibrated coefficients take precedence; fall back to the library
        # defaults used by V16EdpCost when the calibration file is absent.
        eff_macs = coeffs.eff_macs if coeffs else PEAK_MACS
        bw = coeffs.bw_eff_bpc if coeffs else BANDWIDTH_BPC

        eb = op.elem_bytes
        SPm, SPn = candidate.SPm, candidate.SPn
        P = candidate.num_cores
        TM, TK, TN = candidate.TM, candidate.TK, candidate.TN

        load_A = SPm * TM * TK * eb
        load_B = SPn * TK * TN * eb
        store_C = P * TM * TN * eb

        compute_cyc = (TM * TK * TN) / eff_macs
        load_cyc = (load_A + load_B) / bw
        store_cyc = store_C / bw

        if self.stage_mode == "sum":
            aie_cycle = compute_cyc + load_cyc + store_cyc
        else:  # "max"
            aie_cycle = max(compute_cyc, load_cyc, store_cyc)

        total_cycle = aie_cycle * candidate.tp_total

        return CostResult(
            candidate=candidate, tp_order=tp_order,
            t_comp=compute_cyc * candidate.tp_total,
            t_comm=(load_cyc + store_cyc) * candidate.tp_total,
            t_overhead=0.0,
            t_total=total_cycle,
            e_dynamic_comp=math.nan,
            e_dynamic_comm=math.nan,
            e_static=math.nan,
            e_total=math.nan,
            # Expose cycle as edp so EdpSelector can also consume CharmCost
            # outputs (CycleSelector is the canonical pair).
            edp=total_cycle,
        )
