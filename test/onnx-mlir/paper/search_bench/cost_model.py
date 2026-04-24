"""
cost_model.py — Performance and energy models from Methodology Sections 3.4 & 3.5.

Performance Model (v16 DMA-Bottleneck):
    T_total = T_comp + T_comm + T_overhead
    T_comp  = (M * N * K) / (P * eff_macs)
    T_comm  = N_dma * max(L_DMA, b_bar / bw_eff)
              where b_bar = Total_Data / N_dma
              (per-descriptor cost is the larger of setup latency and payload transfer)
    T_overhead = L_SYNC * TP_total + L_CORE * P * TP_total + L_STARTUP
              (L_DMA moved into T_comm; no longer duplicated here)

Energy Model (v16 4-term decomposition):
    E_total = E_active + E_comm + E_startup + E_static
            = P_CORE * P * T_total  (active core power, mW × µs)
            + E_BYTE * Total_Data   (per-byte data-movement energy, µJ/B × B)
            + E_STARTUP             (one-time dispatch energy, µJ)
            + P_BASE * T_total      (system base power, W × µs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config import Config

# ─── tpOrder encoding ────────────────────────────────────────────────────────
# In the JSON, tpOrder is a 3-element list [innermost, ..., outermost]
# where 0=M, 1=N, 2=K.
# M-inner: inner axis = 0, N-inner: inner axis = 1, K-inner: inner axis = 2
TPORDER_M_INNER = 0
TPORDER_N_INNER = 1
TPORDER_K_INNER = 2

TpOrderName = Literal["M-inner", "N-inner", "K-inner"]

def tporder_from_list(tp_list: list[int]) -> int:
    """Extract inner axis index from tpOrder list [innermost, ..., outermost]."""
    return tp_list[0]

def tporder_name(inner_axis: int) -> TpOrderName:
    return {0: "M-inner", 1: "N-inner", 2: "K-inner"}[inner_axis]


# ─── Mapping configuration ──────────────────────────────────────────────────
@dataclass
class MappingConfig:
    M: int; K: int; N: int
    SPm: int; SPn: int
    TPm: int; TPk: int; TPn: int
    tpOrder_inner: int  # 0=M, 1=N, 2=K

    @property
    def P(self) -> int:
        return self.SPm * self.SPn

    @property
    def TM(self) -> int:
        return self.M // (self.SPm * self.TPm)

    @property
    def TK(self) -> int:
        return self.K // self.TPk  # SP_k = 1

    @property
    def TN(self) -> int:
        return self.N // (self.SPn * self.TPn)

    @property
    def TP_total(self) -> int:
        return self.TPm * self.TPn * self.TPk

    @property
    def MACs(self) -> int:
        return 2 * self.M * self.N * self.K


# ─── Data movement (Section 3.3) ────────────────────────────────────────────
def total_data_elements(mc: MappingConfig) -> int:
    """Total data transfer in elements (Section 3.3.2, Step 3).
    
    Returns the sum of LHS + RHS + OUT transfer counts.
    SP-independent (Property 1): spatial reuse cancels spatial duplication.
    """
    M, K, N = mc.M, mc.K, mc.N
    TPm, TPn, TPk = mc.TPm, mc.TPn, mc.TPk
    inner = mc.tpOrder_inner

    if inner == TPORDER_M_INNER:
        lhs = M * K * TPn
        rhs = K * N
        out = 2 * M * N * TPk
    elif inner == TPORDER_N_INNER:
        lhs = M * K
        rhs = K * N * TPm
        out = 2 * M * N * TPk
    elif inner == TPORDER_K_INNER:
        lhs = M * K * TPn
        rhs = K * N * TPm
        # v16: partial-sum accumulation is local (no write-back per K step),
        # so the OUT transfer count is 1× (write final result), not 2×.
        out = M * N
    else:
        raise ValueError(f"Invalid tpOrder_inner: {inner}")

    return lhs + rhs + out


def total_data_bytes(mc: MappingConfig, elem_bytes: int = 2) -> int:
    """Total_Data in bytes."""
    return total_data_elements(mc) * elem_bytes


def n_dma(mc: MappingConfig) -> int:
    """Total DMA descriptor count over the entire execution (Section 3.3.3).

    Accounts for inner-loop temporal reuse: the operand resident in local
    memory during the inner loop is transferred only at outer-loop
    transitions (TP_total / TP_inner times), while non-resident operands
    are transferred at every time step (TP_total times).
    """
    inner = mc.tpOrder_inner
    P = mc.P
    SPm, SPn = mc.SPm, mc.SPn
    tp_total = mc.TP_total

    if inner == TPORDER_K_INNER:
        tp_inner = mc.TPk
        n_every = SPm + SPn          # LHS + RHS
        n_reused = 2 * P             # OUT (partial sum accumulation)
    elif inner == TPORDER_M_INNER:
        tp_inner = mc.TPm
        n_every = SPm + 2 * P        # LHS + OUT
        n_reused = SPn                # RHS (resident)
    elif inner == TPORDER_N_INNER:
        tp_inner = mc.TPn
        n_every = SPn + 2 * P        # RHS + OUT
        n_reused = SPm                # LHS (resident)
    else:
        raise ValueError(f"Invalid tpOrder_inner: {inner}")

    return n_every * tp_total + n_reused * (tp_total // tp_inner)


# ─── Performance model (Section 3.4) ────────────────────────────────────────
def t_comp(mc: MappingConfig, cfg: Config) -> float:
    """Compute time in cycles."""
    return (mc.M * mc.N * mc.K) / (mc.P * cfg.perf.eff_macs)


def t_comm(mc: MappingConfig, cfg: Config) -> float:
    """Communication time in cycles (v16 DMA-bottleneck).

    Per-descriptor cost is max(L_DMA setup latency, payload transfer time).
    The communication phase issues N_dma descriptors, each carrying an average
    payload of b_bar = Total_Data / N_dma bytes, so:

        T_comm = N_dma × max(L_DMA, b_bar / bw_eff)
    """
    p = cfg.perf
    nd = n_dma(mc)
    td = total_data_bytes(mc, cfg.hw.elem_bytes)
    if nd <= 0:
        return 0.0
    b_bar = td / nd
    per_desc_cycles = max(p.L_DMA, b_bar / p.bw_eff_bpc)
    return nd * per_desc_cycles


def t_overhead(mc: MappingConfig, cfg: Config) -> float:
    """Overhead time in cycles (v16: L_DMA term folded into T_comm)."""
    p = cfg.perf
    tp_total = mc.TP_total
    P = mc.P
    return (p.L_SYNC * tp_total
            + p.L_CORE * P * tp_total
            + p.L_STARTUP)


def t_total(mc: MappingConfig, cfg: Config) -> float:
    """Total execution time in cycles (non-overlapping)."""
    return t_comp(mc, cfg) + t_comm(mc, cfg) + t_overhead(mc, cfg)


# ─── Energy model (Section 3.5) ─────────────────────────────────────────────
def e_total(mc: MappingConfig, cfg: Config) -> float:
    """Total energy in µJ (v16 4-term decomposition).

    E_total = E_active + E_comm + E_startup + E_static
            = P_CORE·P·T + E_BYTE·Total_Data + E_STARTUP + P_BASE·T

    Unit conversions (all terms → µJ):
      P_CORE (mW) × P × T (µs) × 1e-3   → µJ
      E_BYTE (µJ/B) × Total_Data (B)     → µJ
      E_STARTUP (µJ)                     → µJ
      P_BASE (W) × T (µs)                → µJ
    """
    e = cfg.energy
    P = mc.P
    t_cy = t_total(mc, cfg)                     # cycles
    t_us = t_cy / cfg.hw.clock_mhz              # µs
    td = total_data_bytes(mc, cfg.hw.elem_bytes)  # bytes

    return (e.P_CORE * P * t_us * 1e-3          # mW × µs × 1e-3 = µJ
            + e.E_BYTE * td                      # µJ/B × B = µJ
            + e.E_STARTUP                        # µJ
            + e.P_BASE * t_us)                   # W × µs = µJ


def edp(mc: MappingConfig, cfg: Config) -> float:
    """Energy-Delay Product."""
    return t_total(mc, cfg) * e_total(mc, cfg)


# ─── Convenience: predict from raw parameters ───────────────────────────────
def predict(M: int, K: int, N: int,
            SPm: int, SPn: int,
            TPm: int, TPk: int, TPn: int,
            tpOrder_inner: int,
            cfg: Config) -> dict:
    """Compute all cost model outputs for a given configuration."""
    mc = MappingConfig(M, K, N, SPm, SPn, TPm, TPk, TPn, tpOrder_inner)
    tt = t_total(mc, cfg)
    et = e_total(mc, cfg)
    return {
        "T_total": tt,
        "E_total": et,
        "EDP": tt * et,
        "T_comp": t_comp(mc, cfg),
        "T_comm": t_comm(mc, cfg),
        "T_overhead": t_overhead(mc, cfg),
        "Total_Data": total_data_bytes(mc, cfg.hw.elem_bytes),
        "N_dma": n_dma(mc),
        "TP_total": mc.TP_total,
        "P": mc.P,
    }
