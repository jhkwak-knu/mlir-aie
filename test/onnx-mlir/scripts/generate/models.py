#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models.py -- Single source of truth for performance and energy model formulas.

All functions accept scalar (int/float) or numpy array inputs.
NumPy broadcasting handles both cases transparently.

No file I/O, no CalibCoeffs dependency -- pure math only.
Callers (cost_model.py, recalibrate_model.py, etc.) extract raw values
from their own data structures before calling these functions.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Union

# Type alias: scalar or array
Numeric = Union[int, float, np.ndarray]

# tpOrder axis indices (must match tiling_common.py)
TP_AXIS_M, TP_AXIS_N, TP_AXIS_K = 0, 1, 2

# NPU clock frequency (MHz)
CLOCK_MHZ = 1500


# ============================================================
# Common helper functions
# ============================================================

def dma_ops_per_step(
    spm: Numeric, spn: Numeric, num_cores: Numeric, tp_order: int,
) -> Numeric:
    """Number of unique DMA descriptor setups per temporal iteration.

    Balanced SP (SPm ~= SPn) minimizes this count via AM-GM inequality:
      SPm + SPn >= 2*sqrt(N_cores), equality at SPm = SPn.

    For scalar tp_order only (tp_order does not vary within a single call).
    """
    if tp_order == TP_AXIS_M:
        return spm + 2 * num_cores
    elif tp_order == TP_AXIS_N:
        return spn + 2 * num_cores
    else:  # TP_AXIS_K
        return spm + spn


def dma_ops_decomposed(
    spm: Numeric, spn: Numeric, num_cores: Numeric, tp_order: int,
) -> Tuple[Numeric, Numeric]:
    """Decompose DMA ops into 'every iteration' and 'reused' categories.

    Returns (N_dma_every, N_dma_reused) where:
      - N_dma_every: DMA descriptors needed every temporal iteration
      - N_dma_reused: DMA descriptors that benefit from inner-loop reuse
        (only reloaded when outer loop advances: TP_total / TP_inner times)

    D_total = N_dma_every * TP_total + N_dma_reused * (TP_total / TP_inner)

    Physical basis: inner-loop temporal reuse means some operands stay in
    tile memory across consecutive iterations of the innermost temporal axis.
    """
    if tp_order == TP_AXIS_M:
        # M-inner: LHS + OUT reload every iter; RHS reused across TPm
        return spm + 2 * num_cores, spn
    elif tp_order == TP_AXIS_N:
        # N-inner: RHS + OUT reload every iter; LHS reused across TPn
        return spn + 2 * num_cores, spm
    else:  # TP_AXIS_K
        # K-inner: LHS + RHS reload every iter; OUT reused (local accum)
        return spm + spn, 2 * num_cores


def tp_inner_value(
    tpm: Numeric, tpk: Numeric, tpn: Numeric, tp_order: int,
) -> Numeric:
    """Return the TP value for the innermost temporal axis."""
    if tp_order == TP_AXIS_M:
        return tpm
    elif tp_order == TP_AXIS_N:
        return tpn
    else:
        return tpk


def d_total(
    spm: Numeric, spn: Numeric, num_cores: Numeric,
    tpm: Numeric, tpk: Numeric, tpn: Numeric,
    tp_total: Numeric, tp_order: int,
) -> Numeric:
    """Refined total DMA descriptor setups across all temporal iterations.

    D_total = N_dma_every * TP_total + N_dma_reused * (TP_total / TP_inner)

    Degeneracy: when TP_inner == 1, D_total == N_dma * TP_total (baseline).
    """
    n_every, n_reused = dma_ops_decomposed(spm, spn, num_cores, tp_order)
    tp_inn = tp_inner_value(tpm, tpk, tpn, tp_order)
    return n_every * tp_total + n_reused * (tp_total / tp_inn)


def total_data_bytes(
    M: Numeric, K: Numeric, N: Numeric, elem_bytes: Numeric,
    spm: Numeric, spn: Numeric,
    tpm: Numeric, tpk: Numeric, tpn: Numeric,
    tp_order: int,
) -> Numeric:
    """Total data transfer volume (bytes) with temporal/spatial reuse.

    The innermost temporal axis determines which operands stay in tile memory
    (temporal reuse) and which are shared across spatial cores (spatial reuse).
    """
    if tp_order == TP_AXIS_M:
        # M innermost: RHS reused across TPm iterations and SPm cores.
        lhs = M * K * tpn
        rhs = K * N
        out = 2 * M * N * tpk
    elif tp_order == TP_AXIS_N:
        # N innermost: LHS reused across TPn iterations and SPn cores.
        lhs = M * K
        rhs = K * N * tpm
        out = 2 * M * N * tpk
    else:
        # K innermost: OUT reused across TPk iterations (local accumulation).
        lhs = M * K * tpn
        rhs = K * N * tpm
        out = 2 * M * N

    return (lhs + rhs + out) * elem_bytes


# ============================================================
# Performance Model
# ============================================================

class PerfModel:
    """Performance prediction formulas for XDNA2 NPU."""

    @staticmethod
    def components_v9(
        macs: Numeric, data_bytes: Numeric,
        n_cores: Numeric, tp_total: Numeric, n_dma: Numeric,
        eff_macs: float, bw_bpc: float,
        l_sync: float, l_sync2: float, l_dma: float,
        l_startup: float,
    ) -> Tuple[Numeric, Numeric, Numeric]:
        """v9 Core-Sync model: returns (T_comp, T_comm, T_overhead) in cycles.

        T_comp = MACs / (N_cores * eff_macs)
        T_comm = data_bytes / bw_bpc
        T_overhead = L_SYNC*TP + L_SYNC2*P*TP + L_DMA*N_dma*TP + L_STARTUP
        """
        t_comp = macs / (n_cores * eff_macs)
        t_comm = data_bytes / bw_bpc

        if l_sync2 != 0:
            # v9: per-core per-iteration barrier cost
            t_overhead = (l_sync * tp_total
                          + l_sync2 * n_cores * tp_total
                          + l_dma * n_dma * tp_total
                          + l_startup)
        else:
            # v8 compat: l_core as per-core fixed cost (no TP scaling)
            t_overhead = (l_sync * tp_total
                          + l_dma * n_dma * tp_total
                          + l_startup)

        return t_comp, t_comm, t_overhead

    @staticmethod
    def predict_v9(
        macs: Numeric, data_bytes: Numeric,
        n_cores: Numeric, tp_total: Numeric, n_dma: Numeric,
        eff_macs: float, bw_bpc: float,
        l_sync: float, l_sync2: float, l_dma: float,
        l_startup: float,
    ) -> Numeric:
        """v9 Core-Sync model: returns T_total in cycles."""
        t_comp, t_comm, t_overhead = PerfModel.components_v9(
            macs, data_bytes, n_cores, tp_total, n_dma,
            eff_macs, bw_bpc, l_sync, l_sync2, l_dma, l_startup)
        return t_comp + t_comm + t_overhead


# ============================================================
# Energy Model
# ============================================================

class EnergyModel:
    """Energy prediction formulas. All outputs in pJ unless noted."""

    @staticmethod
    def predict(
        model_name: str, params: Dict[str, float],
        features: Dict[str, Numeric],
    ) -> Numeric:
        """Dispatch to the appropriate energy model by name.

        Required features keys depend on model; common ones:
          macs, data_bytes, n_cores, tp_total, n_dma,
          t_total_cy, t_comp_cy, t_comm_cy, t_overhead_cy
        """
        dispatch = {
            "E-A": EnergyModel._predict_ea,
            "E-B": EnergyModel._predict_eb,
            "E-C": EnergyModel._predict_ec,
            "E-D": EnergyModel._predict_ed,
            "E-F": EnergyModel._predict_ef,
            "T-A": EnergyModel._predict_ta,
            "T-B": EnergyModel._predict_tb,
            "T-C": EnergyModel._predict_tc,
            "T-E": EnergyModel._predict_te,
            "T-3": EnergyModel._predict_t3,
            "T-3S": EnergyModel._predict_t3s,
            "L-B": EnergyModel._predict_lb,
        }
        fn = dispatch.get(model_name)
        if fn is None:
            return EnergyModel._predict_fallback(features)
        return fn(params, features)

    # --- Theory-calibrated models (T-*) ---

    @staticmethod
    def predict_te(
        features: Dict[str, Numeric],
        e_mac: float, e_dram: float,
        e_dma: float, e_sync: float,
        p_base: float, p_core: float,
    ) -> Numeric:
        """T-E: 6-param theoretical + DMA/sync + power model.

        E = e_mac*MACs + e_dram*bytes + e_dma*N_dma*TP + e_sync*TP
            + (p_base + p_core*P) * T_total_us

        Units: e_mac/e_dram in pJ, e_dma/e_sync in uJ,
               p_base/p_core in uW, T in us -> uW*us = pJ
        """
        macs = features["macs"]
        data_bytes = features["data_bytes"]
        n_dma = features["n_dma"]
        tp_total = features["tp_total"]
        n_cores = features["n_cores"]
        t_total_us = features["t_total_cy"] / CLOCK_MHZ

        e_comp = e_mac * macs
        e_comm = e_dram * data_bytes
        e_dma_term = e_dma * n_dma * tp_total * 1e6    # uJ -> pJ
        e_sync_term = e_sync * tp_total * 1e6           # uJ -> pJ
        e_power = (p_base + p_core * n_cores) * t_total_us  # uW*us = pJ

        return e_comp + e_comm + e_dma_term + e_sync_term + e_power

    @staticmethod
    def predict_tb(
        features: Dict[str, Numeric],
        e_mac: float, e_dram: float,
        p_base: float, p_core: float,
    ) -> Numeric:
        """T-B: 4-param base + per-core power model.

        E = e_mac*MACs + e_dram*bytes + (p_base + p_core*P) * T_total_us
        """
        e_comp = e_mac * features["macs"]
        e_comm = e_dram * features["data_bytes"]
        t_total_us = features["t_total_cy"] / CLOCK_MHZ
        e_power = (p_base + p_core * features["n_cores"]) * t_total_us
        return e_comp + e_comm + e_power

    # --- T-3: 3-parameter compact model ---

    @staticmethod
    def predict_compact(
        features: Dict[str, Numeric],
        p_sys: float, p_core: float, e_dma: float,
    ) -> Numeric:
        """T-3: 3-parameter compact energy model.

        E = P_sys*T + P_core*P*T + E_DMA*D_total

        Physical interpretation:
          - P_sys*T: system power independent of active core count
            (CPU, DRAM refresh, uncore, NPU leakage).
            Absorbs P_BASE*T, E_SYNC*TP (TP~T corr=0.959).
          - P_core*P*T: per-core active power (power gating).
            Absorbs P_CORE*P*T, E_MAC*MACs (~P*T), E_DRAM*bytes (~T).
          - E_DMA*D_total: per-DMA-descriptor energy (time-independent).

        Units: p_sys/p_core in uW, e_dma in uJ, T in us -> pJ output.
        """
        t_us = features["t_total_cy"] / CLOCK_MHZ
        n_cores = features["n_cores"]
        d_tot = features["d_total"]

        e_sys = p_sys * t_us                    # uW * us = pJ
        e_core = p_core * n_cores * t_us        # uW * us = pJ
        e_dma_term = e_dma * d_tot * 1e6        # uJ -> pJ

        return e_sys + e_core + e_dma_term

    @staticmethod
    def predict_compact_sync(
        features: Dict[str, Numeric],
        p_sys: float, p_core: float, e_dma: float, e_sync: float,
    ) -> Numeric:
        """T-3S: 4-parameter variant with E_SYNC separated.

        E = P_sys*T + P_core*P*T + E_DMA*D_total + E_SYNC*TP_total

        Rationale: TP-T correlation is high (0.959) but not perfect;
        separating sync energy may improve P_core identifiability.
        """
        t_us = features["t_total_cy"] / CLOCK_MHZ
        n_cores = features["n_cores"]
        d_tot = features["d_total"]
        tp_total = features["tp_total"]

        e_sys = p_sys * t_us
        e_core = p_core * n_cores * t_us
        e_dma_term = e_dma * d_tot * 1e6
        e_sync_term = e_sync * tp_total * 1e6   # uJ -> pJ

        return e_sys + e_core + e_dma_term + e_sync_term

    # --- Dispatcher helpers (params dict -> keyword args) ---

    @staticmethod
    def _predict_te(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        return EnergyModel.predict_te(
            features,
            e_mac=params.get("e_mac_pj", 0),
            e_dram=params.get("e_dram_pj", 0),
            e_dma=params.get("e_dma_uj", 0),
            e_sync=params.get("e_sync_uj", 0),
            p_base=params.get("p_base_uw", 0),
            p_core=params.get("p_core_uw", 0),
        )

    @staticmethod
    def _predict_tb(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        return EnergyModel.predict_tb(
            features,
            e_mac=params.get("e_mac_pj", 0),
            e_dram=params.get("e_dram_pj", 0),
            p_base=params.get("p_base_uw", 0),
            p_core=params.get("p_core_uw", 0),
        )

    @staticmethod
    def _predict_t3(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        return EnergyModel.predict_compact(
            features,
            p_sys=params.get("p_sys_uw", 0),
            p_core=params.get("p_core_uw", 0),
            e_dma=params.get("e_dma_uj", 0),
        )

    @staticmethod
    def _predict_t3s(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        return EnergyModel.predict_compact_sync(
            features,
            p_sys=params.get("p_sys_uw", 0),
            p_core=params.get("p_core_uw", 0),
            e_dma=params.get("e_dma_uj", 0),
            e_sync=params.get("e_sync_uj", 0),
        )

    @staticmethod
    def _predict_ta(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        """T-A: E = e_mac*MACs + e_dram*bytes + p_static*P*T"""
        e_comp = params.get("e_mac_pj", 0) * features["macs"]
        e_comm = params.get("e_dram_pj", 0) * features["data_bytes"]
        p_static = params.get("p_static_pj", 0)
        e_st = features["n_cores"] * p_static * features["t_total_cy"]
        return e_comp + e_comm + e_st

    @staticmethod
    def _predict_tc(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        """T-C: T-B + startup energy."""
        e_base = EnergyModel._predict_tb(params, features)
        e_startup = params.get("e_startup_uj", 0) * 1e6  # uJ -> pJ
        return e_base + e_startup

    @staticmethod
    def _predict_ea(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        """E-A: E = P_active * T + E_startup"""
        p_active = params.get("p_active_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        t_us = features["t_total_cy"] / CLOCK_MHZ
        return (p_active * t_us / 1e6 + e_startup) * 1e6

    @staticmethod
    def _predict_eb(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        """E-B: E = P_comp*t_comp + P_dma*t_comm + P_idle*t_overhead + E_startup"""
        p_comp = params.get("p_comp_uw", 0)
        p_dma = params.get("p_dma_uw", 0)
        p_idle = params.get("p_idle_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        e_comp = p_comp * (features["t_comp_cy"] / CLOCK_MHZ)     # uW*us = pJ
        e_comm = p_dma * (features["t_comm_cy"] / CLOCK_MHZ)
        e_static = p_idle * (features["t_overhead_cy"] / CLOCK_MHZ)
        return e_comp + e_comm + e_static + e_startup * 1e6

    @staticmethod
    def _predict_ec(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        """E-C: E = e_mac*MACs + e_byte*bytes + p_static*N*T + E_startup"""
        e_comp = params.get("e_mac_pj", 0) * features["macs"]
        e_comm = params.get("e_byte_pj", 0) * features["data_bytes"]
        p_static = params.get("p_static_pj", 0)
        e_st = features["n_cores"] * p_static * features["t_total_cy"]
        e_startup = params.get("e_startup_uj", 0) * 1e6
        return e_comp + e_comm + e_st + e_startup

    @staticmethod
    def _predict_ed(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        """E-D: E = P_core*N*T + E_startup"""
        p_core = params.get("p_core_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        t_us = features["t_total_cy"] / CLOCK_MHZ
        return (p_core * features["n_cores"] * t_us / 1e6 + e_startup) * 1e6

    @staticmethod
    def _predict_ef(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        """E-F: E = (P_base + P_core*N)*T + E_startup"""
        p_base = params.get("p_base_uw", 0)
        p_core = params.get("p_core_uw", 0)
        e_startup = params.get("e_startup_uj", 0)
        t_us = features["t_total_cy"] / CLOCK_MHZ
        return ((p_base + p_core * features["n_cores"]) * t_us / 1e6
                + e_startup) * 1e6

    @staticmethod
    def _predict_lb(params: Dict[str, float], features: Dict[str, Numeric]) -> Numeric:
        """L-B: E = C * t^a * N^c  (log-space power law, empirical)."""
        a_time = params.get("a_time", 1.0)
        c_cores = params.get("c_cores", 0.0)
        C = params.get("C", 1.0)
        t_us = features["t_total_cy"] / CLOCK_MHZ
        # L-B output is in uJ; convert to pJ for consistency
        e_uj = C * (t_us ** a_time) * (features["n_cores"] ** c_cores)
        return e_uj * 1e6

    @staticmethod
    def _predict_fallback(features: Dict[str, Numeric]) -> Numeric:
        """Fallback: theoretical constants (no calibration)."""
        E_MAC_PJ = 0.2
        E_DRAM_PJ = 40.0
        P_STATIC_PJ = 27.0
        e_comp = features["macs"] * E_MAC_PJ
        e_comm = features["data_bytes"] * E_DRAM_PJ
        e_static = features["n_cores"] * P_STATIC_PJ * features["t_total_cy"]
        return e_comp + e_comm + e_static

    # -----------------------------------------------------------------
    # Optimization-unit interface (uJ-based internal units)
    #
    # scipy optimization operates in uJ-based units to keep parameter
    # magnitudes reasonable. These functions use the SAME formulas as
    # the deployment-unit versions above, just in different units:
    #   e_mac/e_dram: uJ (not pJ)
    #   p_base/p_core: uJ/cycle (not uW)
    #   e_dma/e_sync: uJ (same as deployment)
    #   output: uJ (not pJ)
    # -----------------------------------------------------------------

    @staticmethod
    def _to_arr(v):
        """Ensure value is a numpy array (handles Python lists)."""
        return v if isinstance(v, np.ndarray) else np.asarray(v, dtype=float)

    @staticmethod
    def predict_te_optim(
        params: Tuple, features: Dict[str, Numeric],
    ) -> Numeric:
        """T-E in optimization units (uJ). Same formula, different scale.

        params = (e_mac, e_dram, e_dma, e_sync, p_base, p_core) all in uJ-based.
        output in uJ.
        """
        e_mac, e_dram, e_dma, e_sync, p_base, p_core = params
        _a = EnergyModel._to_arr
        return (e_mac * _a(features["macs"]) + e_dram * _a(features["data_bytes"])
                + e_dma * _a(features["ndma_tp"]) + e_sync * _a(features["tp_total"])
                + p_base * _a(features["t_total_cy"]) + p_core * _a(features["nt"]))

    @staticmethod
    def predict_tb_optim(
        params: Tuple, features: Dict[str, Numeric],
    ) -> Numeric:
        """T-B in optimization units (uJ)."""
        e_mac, e_dram, p_base, p_core = params
        _a = EnergyModel._to_arr
        return (e_mac * _a(features["macs"]) + e_dram * _a(features["data_bytes"])
                + p_base * _a(features["t_total_cy"]) + p_core * _a(features["nt"]))

    @staticmethod
    def predict_td_optim(
        params: Tuple, features: Dict[str, Numeric],
    ) -> Numeric:
        """T-D in optimization units (uJ)."""
        e_mac, e_dram, e_dma, p_base, p_core = params
        _a = EnergyModel._to_arr
        return (e_mac * _a(features["macs"]) + e_dram * _a(features["data_bytes"])
                + e_dma * _a(features["ndma_tp"])
                + p_base * _a(features["t_total_cy"]) + p_core * _a(features["nt"]))

    @staticmethod
    def predict_ta_optim(
        params: Tuple, features: Dict[str, Numeric],
    ) -> Numeric:
        """T-A in optimization units (uJ)."""
        e_mac, e_dram, p_static = params
        _a = EnergyModel._to_arr
        return (e_mac * _a(features["macs"]) + e_dram * _a(features["data_bytes"])
                + p_static * _a(features["nt"]))

    @staticmethod
    def predict_tc_optim(
        params: Tuple, features: Dict[str, Numeric],
    ) -> Numeric:
        """T-C in optimization units (uJ). T-B + startup."""
        e_mac, e_dram, p_base, p_core, e_startup = params
        _a = EnergyModel._to_arr
        return (e_mac * _a(features["macs"]) + e_dram * _a(features["data_bytes"])
                + p_base * _a(features["t_total_cy"]) + p_core * _a(features["nt"])
                + e_startup)

    @staticmethod
    def predict_t3_optim(
        params: Tuple, features: Dict[str, Numeric],
    ) -> Numeric:
        """T-3 in optimization units (uJ).

        params = (p_sys, p_core, e_dma) in uJ-cycle / uJ units.
        output in uJ.
        """
        p_sys, p_core, e_dma = params
        _a = EnergyModel._to_arr
        return (p_sys * _a(features["t_total_cy"])
                + p_core * _a(features["nt"])
                + e_dma * _a(features["d_total"]))

    @staticmethod
    def predict_t3s_optim(
        params: Tuple, features: Dict[str, Numeric],
    ) -> Numeric:
        """T-3S in optimization units (uJ). T-3 + sync."""
        p_sys, p_core, e_dma, e_sync = params
        _a = EnergyModel._to_arr
        return (p_sys * _a(features["t_total_cy"])
                + p_core * _a(features["nt"])
                + e_dma * _a(features["d_total"])
                + e_sync * _a(features["tp_total"]))

    @staticmethod
    def predict_tf_optim(
        params: Tuple, features: Dict[str, Numeric],
    ) -> Numeric:
        """T-F in optimization units (uJ). Separate comp/comm power."""
        p_comp, p_comm, e_dma, e_sync, p_idle = params
        _a = EnergyModel._to_arr
        return (p_comp * _a(features["t_comp_cy"]) + p_comm * _a(features["t_comm_cy"])
                + e_dma * _a(features["ndma_tp"]) + e_sync * _a(features["tp_total"])
                + p_idle * _a(features["nt"]))

    @staticmethod
    def optim_params_to_calib(model_name: str, params: Tuple) -> Dict[str, float]:
        """Convert optimization-internal params to calibration.json units.

        Optimization uses uJ-based units; calibration uses pJ/uW units.
        """
        if model_name == "T-E":
            e_mac, e_dram, e_dma, e_sync, p_base, p_core = params
            return {
                "e_mac_pj": e_mac * 1e6,
                "e_dram_pj": e_dram * 1e6,
                "e_dma_uj": e_dma,
                "e_sync_uj": e_sync,
                "p_base_uw": p_base * CLOCK_MHZ * 1e6,
                "p_core_uw": p_core * CLOCK_MHZ * 1e6,
            }
        elif model_name == "T-B":
            e_mac, e_dram, p_base, p_core = params
            return {
                "e_mac_pj": e_mac * 1e6,
                "e_dram_pj": e_dram * 1e6,
                "p_base_uw": p_base * CLOCK_MHZ * 1e6,
                "p_core_uw": p_core * CLOCK_MHZ * 1e6,
            }
        elif model_name == "T-D":
            e_mac, e_dram, e_dma, p_base, p_core = params
            return {
                "e_mac_pj": e_mac * 1e6,
                "e_dram_pj": e_dram * 1e6,
                "e_dma_uj": e_dma,
                "p_base_uw": p_base * CLOCK_MHZ * 1e6,
                "p_core_uw": p_core * CLOCK_MHZ * 1e6,
            }
        elif model_name == "T-A":
            e_mac, e_dram, p_static = params
            return {
                "e_mac_pj": e_mac * 1e6,
                "e_dram_pj": e_dram * 1e6,
                "p_static_pj": p_static * 1e6,
            }
        elif model_name == "T-C":
            e_mac, e_dram, p_base, p_core, e_startup = params
            return {
                "e_mac_pj": e_mac * 1e6,
                "e_dram_pj": e_dram * 1e6,
                "p_base_uw": p_base * CLOCK_MHZ * 1e6,
                "p_core_uw": p_core * CLOCK_MHZ * 1e6,
                "e_startup_uj": e_startup,
            }
        elif model_name == "T-3":
            p_sys, p_core, e_dma = params
            return {
                "p_sys_uw": p_sys * CLOCK_MHZ * 1e6,
                "p_core_uw": p_core * CLOCK_MHZ * 1e6,
                "e_dma_uj": e_dma,
            }
        elif model_name == "T-3S":
            p_sys, p_core, e_dma, e_sync = params
            return {
                "p_sys_uw": p_sys * CLOCK_MHZ * 1e6,
                "p_core_uw": p_core * CLOCK_MHZ * 1e6,
                "e_dma_uj": e_dma,
                "e_sync_uj": e_sync,
            }
        else:
            return {}
