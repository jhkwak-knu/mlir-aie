"""V16 EDP cost function.

Performance + energy + EDP computation moved verbatim from
scripts/generate/cost_model.py (no numerical change). The module still
exposes the underlying helper functions (total_data_bytes, perf_compute,
perf_comm, perf_overhead, energy_*, energy_total_calibrated) so callers
migrating incrementally can import them.

V16EdpCost adapts this logic to the CostFunction Protocol.
"""

from __future__ import annotations

from typing import Optional, Tuple

from xdna_search import models as _models
from xdna_search.hw_constants import TP_AXIS_K
from xdna_search.types import CalibCoeffs, Candidate, CostResult, OpCase


# Hardware coefficients for XDNA2 (Ryzen AI 9 HX 370, Strix Point, TSMC N4P).
# Sources: AMD XDNA2 spec (256 MACs/cycle BF16 per tile, 32 tiles, ~1.5 GHz),
#          LPDDR5X-7500 measured bandwidth ~80 GB/s (Chips and Cheese),
#          Horowitz 2014 scaled to N4P for energy estimates.
PEAK_MACS = 256        # MACs/Cycle/Tile (BF16)
BANDWIDTH_BPC = 4      # Bytes/Cycle (NoC stream bandwidth per channel)
ALPHA_CYCLES = 20      # Cycles per temporal iteration (pipeline drain/fill)
E_MAC_PJ = 0.2         # pJ per MAC (TSMC N4P estimate)
E_DRAM_PJ = 40         # pJ per Byte DRAM access (LPDDR5X estimate)
P_STATIC_PJ = 27       # pJ/Cycle per Tile (default mode, 0.04W @ 1.5 GHz)


# ===== helpers =====================================================

def total_data_bytes(op: OpCase, c: Candidate, tp_order: int) -> int:
    """Total data transfer volume (bytes). Delegates to xdna_search.models."""
    return _models.total_data_bytes(
        op.M, op.K, op.N, op.elem_bytes,
        c.SPm, c.SPn, c.TPm, c.TPk, c.TPn, tp_order,
    )


def _d_total(c: Candidate, tp_order: int) -> float:
    """Refined total DMA descriptor setups (D_total) for a Candidate."""
    return _models.d_total(
        c.SPm, c.SPn, c.num_cores,
        c.TPm, c.TPk, c.TPn, c.tp_total, tp_order,
    )


def _dma_ops_per_step(c: Candidate, tp_order: int) -> int:
    """Delegate to models.dma_ops_per_step()."""
    return _models.dma_ops_per_step(c.SPm, c.SPn, c.num_cores, tp_order)


# ===== performance (cycles) ========================================

def perf_compute(
    op: OpCase, c: Candidate, coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """T_comp: compute time assuming all cores run in parallel."""
    eff = coeffs.eff_macs if coeffs else PEAK_MACS
    return (op.M * op.N * op.K) / (c.SPm * c.SPn * eff)


def perf_comm(
    op: OpCase, c: Candidate, tp_order: int,
    coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """T_comm: pure data transfer time at stream bandwidth."""
    bw = coeffs.bw_eff_bpc if coeffs else BANDWIDTH_BPC
    return total_data_bytes(op, c, tp_order) / bw


def perf_overhead(
    c: Candidate, coeffs: Optional[CalibCoeffs] = None,
    tp_order: int = TP_AXIS_K,
) -> float:
    """T_overhead: delegates to the appropriate PerfModel variant.

    DMA-Bottleneck (v16): D * max(L_SETUP, avg_tile/BW) + sync overhead
    DMA-Refined (v13+):   L_DMA * D_total + sync overhead
    Core-Sync (v9):       L_DMA * N_dma * TP_total + sync overhead
    """
    if coeffs and coeffs.calibrated:
        if coeffs.perf_model == "DMA-Bottleneck":
            d_tot = _d_total(c, tp_order)
            l_setup = coeffs.l_setup_cy if hasattr(coeffs, "l_setup_cy") else coeffs.l_dma_cy
            _, t_dma, t_sync = _models.PerfModel.components_dma_bottleneck(
                macs=0, data_bytes=0,
                n_cores=c.num_cores, tp_total=c.tp_total, d_total_val=d_tot,
                eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
                l_sync=coeffs.l_sync_cy, l_pe=coeffs.l_pe_cy,
                l_setup=l_setup, l_startup=coeffs.l_startup_cy,
            )
            return t_dma + t_sync
        elif coeffs.perf_model == "DMA-Refined":
            d_tot = _d_total(c, tp_order)
            _, _, t_ovh = _models.PerfModel.components_dma_refined(
                macs=0, data_bytes=0,
                n_cores=c.num_cores, tp_total=c.tp_total, d_total_val=d_tot,
                eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
                l_sync=coeffs.l_sync_cy, l_pe=coeffs.l_pe_cy,
                l_dma=coeffs.l_dma_cy, l_startup=coeffs.l_startup_cy,
            )
            return t_ovh
        else:
            n_dma = _dma_ops_per_step(c, tp_order)
            _, _, t_ovh = _models.PerfModel.components_v9(
                macs=0, data_bytes=0,
                n_cores=c.num_cores, tp_total=c.tp_total, n_dma=n_dma,
                eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
                l_sync=coeffs.l_sync_cy, l_pe=coeffs.l_pe_cy,
                l_dma=coeffs.l_dma_cy, l_startup=coeffs.l_startup_cy,
            )
            return t_ovh
    return ALPHA_CYCLES * (c.TPm * c.TPn * c.TPk)


# ===== energy (pJ) =================================================

def energy_dynamic_comp(
    op: OpCase, coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """E_dynamic_comp: total MAC energy (constant across candidates)."""
    if coeffs and coeffs.energy_calibrated:
        e_mac = coeffs.energy_params.get("e_mac_pj", E_MAC_PJ)
    else:
        e_mac = E_MAC_PJ
    return op.M * op.N * op.K * e_mac


def energy_dynamic_comm(
    op: OpCase, c: Candidate, tp_order: int,
    coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """E_dynamic_comm: DRAM access energy proportional to transfer volume."""
    if coeffs and coeffs.energy_calibrated:
        e_dram = coeffs.energy_params.get("e_byte_pj", E_DRAM_PJ)
    else:
        e_dram = E_DRAM_PJ
    return total_data_bytes(op, c, tp_order) * e_dram


def energy_static(
    c: Candidate, t_total: float,
    coeffs: Optional[CalibCoeffs] = None,
) -> float:
    """E_static: leakage energy for active tiles over total execution time."""
    if coeffs and coeffs.energy_calibrated:
        p_static = coeffs.energy_params.get("p_static_pj", P_STATIC_PJ)
    else:
        p_static = P_STATIC_PJ
    return (c.SPm * c.SPn) * p_static * t_total


def energy_total_calibrated(
    op: OpCase, c: Candidate, tp_order: int,
    t_comp: float, t_comm: float, t_overhead: float, t_total: float,
    coeffs: CalibCoeffs,
) -> Tuple[float, float, float, float]:
    """Compute energy using the calibrated model; delegates to models.EnergyModel.

    Returns (e_comp_pj, e_comm_pj, e_static_pj, e_total_pj). The component
    breakdown is approximate for models that do not decompose.
    """
    features = {
        "macs": op.M * op.N * op.K,
        "data_bytes": total_data_bytes(op, c, tp_order),
        "n_cores": c.SPm * c.SPn,
        "tp_total": c.tp_total,
        "n_dma": _dma_ops_per_step(c, tp_order),
        "d_total": _d_total(c, tp_order),
        "t_total_cy": t_total,
        "t_comp_cy": t_comp,
        "t_comm_cy": t_comm,
        "t_overhead_cy": t_overhead,
    }
    e_total = _models.EnergyModel.predict(
        coeffs.energy_model, coeffs.energy_params, features,
    )

    # Approximate breakdown for reporting.
    e_mac_pj = coeffs.energy_params.get("e_mac_pj", E_MAC_PJ)
    e_dram_pj = coeffs.energy_params.get(
        "e_dram_pj",
        coeffs.energy_params.get("e_byte_pj", E_DRAM_PJ),
    )
    e_comp = features["macs"] * e_mac_pj
    e_comm = features["data_bytes"] * e_dram_pj
    e_static = max(0.0, e_total - e_comp - e_comm)
    return e_comp, e_comm, e_static, e_total


# ===== combined evaluation =========================================

def evaluate_candidate(
    op: OpCase, c: Candidate, tp_order: int,
    coeffs: Optional[CalibCoeffs] = None,
) -> CostResult:
    """Evaluate a single candidate for a specific tpOrder."""
    if coeffs and coeffs.calibrated and coeffs.perf_model == "DMA-Bottleneck":
        # v16: T = T_comp + D*max(L_SETUP, avg_tile/BW) + sync + startup.
        # T_comm is integrated into the DMA bottleneck term (no separate T_comm).
        l_setup = coeffs.l_setup_cy if hasattr(coeffs, "l_setup_cy") else coeffs.l_dma_cy
        data_bytes = total_data_bytes(op, c, tp_order)
        d_tot = _d_total(c, tp_order)
        tc_cy, td_cy, to_cy = _models.PerfModel.components_dma_bottleneck(
            macs=op.M * op.N * op.K,
            data_bytes=data_bytes,
            n_cores=c.num_cores, tp_total=c.tp_total, d_total_val=d_tot,
            eff_macs=coeffs.eff_macs, bw_bpc=coeffs.bw_eff_bpc,
            l_sync=coeffs.l_sync_cy, l_pe=coeffs.l_pe_cy,
            l_setup=l_setup, l_startup=coeffs.l_startup_cy,
        )
        tc = tc_cy  # T_comp in cycles
        tm = td_cy  # T_dma in cycles (replaces T_comm)
        to = to_cy  # T_sync + T_startup in cycles
        tt = tc + tm + to
    else:
        tc = perf_compute(op, c, coeffs)
        tm = perf_comm(op, c, tp_order, coeffs)
        to = perf_overhead(c, coeffs, tp_order)
        tt = tc + tm + to

    if coeffs and coeffs.energy_calibrated:
        edc, edm, es, et = energy_total_calibrated(
            op, c, tp_order, tc, tm, to, tt, coeffs,
        )
    else:
        edc = energy_dynamic_comp(op, coeffs)
        edm = energy_dynamic_comm(op, c, tp_order, coeffs)
        es = energy_static(c, tt, coeffs)
        et = edc + edm + es

    return CostResult(
        candidate=c, tp_order=tp_order,
        t_comp=tc, t_comm=tm, t_overhead=to, t_total=tt,
        e_dynamic_comp=edc, e_dynamic_comm=edm, e_static=es, e_total=et,
        edp=tt * et,
    )


class V16EdpCost:
    """CostFunction: XDNA2 v16 DMA-Bottleneck perf + Power-Time-Byte energy + EDP."""

    name = "v16-edp"

    def evaluate(
        self,
        op: OpCase,
        candidate: Candidate,
        tp_order: int,
        coeffs: Optional[CalibCoeffs] = None,
    ) -> CostResult:
        return evaluate_candidate(op, candidate, tp_order, coeffs)
