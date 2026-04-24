"""
baselines.py — Compute GT, Naive-Max, and Framework optimal configurations.

- GT (Ground Truth): best measured EDP from exhaustive measurement data
- Naive-Max: best configuration using all 32 cores (vendor compiler default)
- Framework: best configuration found by cost-model-based exhaustive search
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config
from cost_model import (
    MappingConfig, edp, n_dma, t_total, e_total, total_data_bytes,
    TPORDER_M_INNER, TPORDER_N_INNER, TPORDER_K_INNER,
)
from data_loader import WorkloadKey


@dataclass
class BaselineResult:
    """Result of a baseline search for one workload."""
    workload: WorkloadKey
    P: int
    SPm: int; SPn: int
    TPm: int; TPk: int; TPn: int
    tpOrder_inner: int
    # Measured values (NaN if not available in measurement data)
    measured_time_us: float
    measured_energy_uj: float
    measured_edp: float
    # Predicted values from cost model
    pred_time: float
    pred_energy: float
    pred_edp: float
    # Whether this config exists in measurement data
    in_measurements: bool
    # Whether this is a fallback (original FW pick not in measurements)
    is_fallback: bool = False


# ─── GT: Ground Truth from measurements ─────────────────────────────────────
def find_gt(
    wl_df: pd.DataFrame,
    workload: WorkloadKey,
) -> Optional[BaselineResult]:
    """Find the configuration with minimum measured EDP."""
    valid = wl_df[wl_df["energy_valid"]].copy()
    if valid.empty:
        return None
    
    best_idx = valid["edp_measured"].idxmin()
    row = valid.loc[best_idx]
    
    return BaselineResult(
        workload=workload,
        P=int(row["numSpm"]),
        SPm=int(row["SPm"]), SPn=int(row["SPn"]),
        TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
        tpOrder_inner=int(row["tpOrder_inner"]),
        measured_time_us=float(row["time_us"]),
        measured_energy_uj=float(row["energy_uj"]),
        measured_edp=float(row["edp_measured"]),
        pred_time=float(row["t_total_pred"]),
        pred_energy=float(row.get("e_total_pred", np.nan)),
        pred_edp=float(row.get("edp_pred", np.nan)),
        in_measurements=True,
    )


# ─── P_max-GT: Ground-truth optimum constrained to P = P_max ───────────────
# NOTE: Function name `find_naive_max` kept for backwards compatibility with
# main.py / tables.py / figures.py references, but the semantics are now
# "P_max-GT" — the measurement-optimal oracle restricted to P = P_max.
def find_naive_max(
    M: int, K: int, N: int,
    cfg: Config,
    wl_df: Optional[pd.DataFrame] = None,
) -> Optional[BaselineResult]:
    """P_max-GT baseline: ground-truth optimum constrained to P = P_max.

    Semantics (contextual reference bound):
      Among all *measured* configurations at P = max feasible PE count,
      select the one with minimum *measured* EDP.

    Represents the best EDP achievable when the core count is forced to
    P_max with optimal per-workload tile (SP×TP×tpOrder) selection. The
    gap between GT and P_max-GT quantifies the pure contribution of
    adaptive P selection (resource elasticity) — i.e., the value added
    by *not* restricting P.

    Fallback (wl_df has no measurements at P_max): reverts to the legacy
    rule-based heuristic (min data transfer, balanced SP) so that the
    function remains total on synthetic / partial measurement sets.
    """
    all_configs = enumerate_configs(M, K, N, cfg)
    if not all_configs:
        return None

    max_p = max(mc.P for mc in all_configs)

    # ── Primary: measurement-optimal at P = max_p ───────────────────
    if wl_df is not None:
        valid_at_pmax = wl_df[
            (wl_df["numSpm"] == max_p)
            & wl_df["energy_valid"]
            & (wl_df["edp_measured"] > 0)
        ]
        if not valid_at_pmax.empty:
            min_idx = valid_at_pmax["edp_measured"].idxmin()
            row = valid_at_pmax.loc[min_idx]
            best_mc = MappingConfig(
                M, K, N,
                int(row["SPm"]), int(row["SPn"]),
                int(row["TPm"]), int(row["TPk"]), int(row["TPn"]),
                int(row["tpOrder_inner"]),
            )
            return BaselineResult(
                workload=(M, K, N),
                P=best_mc.P,
                SPm=best_mc.SPm, SPn=best_mc.SPn,
                TPm=best_mc.TPm, TPk=best_mc.TPk, TPn=best_mc.TPn,
                tpOrder_inner=best_mc.tpOrder_inner,
                measured_time_us=float(row["time_us"]),
                measured_energy_uj=float(row["energy_uj"]),
                measured_edp=float(row["edp_measured"]),
                pred_time=t_total(best_mc, cfg),
                pred_energy=e_total(best_mc, cfg),
                pred_edp=edp(best_mc, cfg),
                in_measurements=True,
                is_fallback=False,
            )

    # ── Fallback: legacy rule-based heuristic ───────────────────────
    # Used only when no measurements are available at P = max_p.
    max_p_configs = [mc for mc in all_configs if mc.P == max_p]
    max_p_configs.sort(key=lambda mc: (
        total_data_bytes(mc),
        abs(mc.SPm - mc.SPn),
    ))
    best_mc = max_p_configs[0]

    measured_time = np.nan
    measured_energy = np.nan
    measured_edp_val = np.nan
    in_meas = False
    is_fb = True

    return BaselineResult(
        workload=(M, K, N),
        P=best_mc.P,
        SPm=best_mc.SPm, SPn=best_mc.SPn,
        TPm=best_mc.TPm, TPk=best_mc.TPk, TPn=best_mc.TPn,
        tpOrder_inner=best_mc.tpOrder_inner,
        measured_time_us=measured_time,
        measured_energy_uj=measured_energy,
        measured_edp=measured_edp_val,
        pred_time=t_total(best_mc, cfg),
        pred_energy=e_total(best_mc, cfg),
        pred_edp=edp(best_mc, cfg),
        in_measurements=in_meas,
        is_fallback=is_fb,
    )


# ─── Framework: cost-model exhaustive search ─────────────────────────────────
def _divisors(n: int) -> List[int]:
    """All divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def _aligned_divisors(n: int, align: int) -> List[int]:
    """Divisors of n that are multiples of align."""
    return [d for d in _divisors(n) if d % align == 0]


def enumerate_configs(
    M: int, K: int, N: int, cfg: Config,
) -> List[MappingConfig]:
    """Enumerate all feasible (P, SP, TP, tpOrder) configurations."""
    hw = cfg.hw
    configs = []

    for P in hw.P_set:
        # SP factorizations: SPm * SPn = P
        for SPm in _divisors(P):
            SPn = P // SPm
            if SPm * SPn != P:
                continue

            # TP ranges: M / (SPm * TPm) must be aligned and fit memory
            for TPm in range(1, M // SPm + 1):
                TM = M // (SPm * TPm)
                if M % (SPm * TPm) != 0:
                    continue
                if TM % hw.TM_align != 0:
                    continue
                if TM < hw.TM_align:
                    continue

                for TPn in range(1, N // SPn + 1):
                    TN = N // (SPn * TPn)
                    if N % (SPn * TPn) != 0:
                        continue
                    if TN % hw.TN_align != 0:
                        continue
                    if TN < hw.TN_align:
                        continue

                    for TPk in range(1, K + 1):
                        TK = K // TPk
                        if K % TPk != 0:
                            continue
                        if TK % hw.TK_align != 0:
                            continue
                        if TK < hw.TK_align:
                            continue

                        # C1: local memory constraint
                        tile_bytes = hw.elem_bytes * (TM * TK + TK * TN + TM * TN)
                        if tile_bytes > hw.L_mem_bytes:
                            continue

                        # All 3 tpOrders
                        for inner in [TPORDER_M_INNER, TPORDER_N_INNER, TPORDER_K_INNER]:
                            configs.append(MappingConfig(
                                M, K, N, SPm, SPn, TPm, TPk, TPn, inner
                            ))

    return configs


# ─── Pruned configuration generation (Rules 1–3) ──────────────────────────
def _valid_tile_sizes(dim: int, sp: int, align: int) -> List[int]:
    """Return all valid tile sizes T for a dimension: T = dim/(sp*TP),
    T >= align, T % align == 0, dim % (sp*TP) == 0.
    Returns T values in descending order (largest tile first)."""
    tiles = []
    for tp in range(1, dim // sp + 1):
        if dim % (sp * tp) != 0:
            continue
        t = dim // (sp * tp)
        if t < align or t % align != 0:
            continue
        tiles.append(t)
    return sorted(tiles, reverse=True)  # largest first


def _valid_tile_sizes_k(K: int, align: int) -> List[int]:
    """Valid tile sizes for K dimension (SP_k = 1)."""
    tiles = []
    for tpk in range(1, K + 1):
        if K % tpk != 0:
            continue
        tk = K // tpk
        if tk < align or tk % align != 0:
            continue
        tiles.append(tk)
    return sorted(tiles, reverse=True)


def _tp_from_tile(dim: int, sp: int, T: int) -> int:
    """TP = dim / (sp * T)."""
    return dim // (sp * T)


def _enumerate_tp_rule1(
    M: int, K: int, N: int,
    SPm: int, SPn: int,
    hw,
) -> List[Tuple[int, int, int]]:
    """Rule 1: TP monotonicity pruning (exact).

    Only keep TP combinations at the C1 boundary — where no single
    tile dimension can be increased without violating memory constraint.
    """
    TMs = _valid_tile_sizes(M, SPm, hw.TM_align)
    TNs = _valid_tile_sizes(N, SPn, hw.TN_align)
    TKs = _valid_tile_sizes_k(K, hw.TK_align)

    results = []
    for TM in TMs:
        for TN in TNs:
            for TK in TKs:
                tile_bytes = hw.elem_bytes * (TM * TK + TK * TN + TM * TN)
                if tile_bytes > hw.L_mem_bytes:
                    continue

                # Check C1 boundary: no single tile can grow further
                at_boundary = True

                # Can TM grow?
                idx_m = TMs.index(TM)
                if idx_m > 0:  # larger TM exists
                    TM_next = TMs[idx_m - 1]
                    if hw.elem_bytes * (TM_next * TK + TK * TN + TM_next * TN) <= hw.L_mem_bytes:
                        at_boundary = False
                        continue

                # Can TN grow?
                idx_n = TNs.index(TN)
                if idx_n > 0:
                    TN_next = TNs[idx_n - 1]
                    if hw.elem_bytes * (TM * TK + TK * TN_next + TM * TN_next) <= hw.L_mem_bytes:
                        at_boundary = False
                        continue

                # Can TK grow?
                idx_k = TKs.index(TK)
                if idx_k > 0:
                    TK_next = TKs[idx_k - 1]
                    if hw.elem_bytes * (TM * TK_next + TK_next * TN + TM * TN) <= hw.L_mem_bytes:
                        at_boundary = False
                        continue

                if at_boundary:
                    TPm = _tp_from_tile(M, SPm, TM)
                    TPn = _tp_from_tile(N, SPn, TN)
                    TPk = K // TK
                    results.append((TPm, TPk, TPn))

    return results


def _enumerate_tp_rule12(
    M: int, K: int, N: int,
    SPm: int, SPn: int,
    inner: int,
    hw,
) -> List[Tuple[int, int, int]]:
    """Rule 1 + Rule 2: TP monotonicity + allocation priority.

    Rule 2: Maximize non-inner axis tiles first (they reduce Total_Data),
    then fill remaining memory with inner-axis tile.
    """
    TMs = _valid_tile_sizes(M, SPm, hw.TM_align)
    TNs = _valid_tile_sizes(N, SPn, hw.TN_align)
    TKs = _valid_tile_sizes_k(K, hw.TK_align)

    results = []

    if inner == TPORDER_K_INNER:
        # Prioritize TM, TN; TK gets remaining memory
        for TM in TMs:
            for TN in TNs:
                # Check if at boundary for TM and TN (non-inner)
                can_grow_m = False
                idx_m = TMs.index(TM)
                if idx_m > 0:
                    TM_next = TMs[idx_m - 1]
                    # Check with smallest TK
                    min_TK = TKs[-1] if TKs else None
                    if min_TK and hw.elem_bytes * (TM_next * min_TK + min_TK * TN + TM_next * TN) <= hw.L_mem_bytes:
                        can_grow_m = True

                can_grow_n = False
                idx_n = TNs.index(TN)
                if idx_n > 0:
                    TN_next = TNs[idx_n - 1]
                    min_TK = TKs[-1] if TKs else None
                    if min_TK and hw.elem_bytes * (TM * min_TK + min_TK * TN_next + TM * TN_next) <= hw.L_mem_bytes:
                        can_grow_n = True

                if can_grow_m or can_grow_n:
                    continue  # non-inner tiles not maximized yet

                # Find largest TK that fits
                best_TK = None
                for TK in TKs:
                    if hw.elem_bytes * (TM * TK + TK * TN + TM * TN) <= hw.L_mem_bytes:
                        best_TK = TK
                        break  # TKs sorted descending
                if best_TK is not None:
                    TPm = _tp_from_tile(M, SPm, TM)
                    TPn = _tp_from_tile(N, SPn, TN)
                    TPk = K // best_TK
                    results.append((TPm, TPk, TPn))

    elif inner == TPORDER_M_INNER:
        # Prioritize TN, TK; TM gets remaining memory
        for TN in TNs:
            for TK in TKs:
                can_grow_n = False
                idx_n = TNs.index(TN)
                if idx_n > 0:
                    TN_next = TNs[idx_n - 1]
                    min_TM = TMs[-1] if TMs else None
                    if min_TM and hw.elem_bytes * (min_TM * TK + TK * TN_next + min_TM * TN_next) <= hw.L_mem_bytes:
                        can_grow_n = True

                can_grow_k = False
                idx_k = TKs.index(TK)
                if idx_k > 0:
                    TK_next = TKs[idx_k - 1]
                    min_TM = TMs[-1] if TMs else None
                    if min_TM and hw.elem_bytes * (min_TM * TK_next + TK_next * TN + min_TM * TN) <= hw.L_mem_bytes:
                        can_grow_k = True

                if can_grow_n or can_grow_k:
                    continue

                best_TM = None
                for TM in TMs:
                    if hw.elem_bytes * (TM * TK + TK * TN + TM * TN) <= hw.L_mem_bytes:
                        best_TM = TM
                        break
                if best_TM is not None:
                    TPm = _tp_from_tile(M, SPm, best_TM)
                    TPn = _tp_from_tile(N, SPn, TN)
                    TPk = K // TK
                    results.append((TPm, TPk, TPn))

    elif inner == TPORDER_N_INNER:
        # Prioritize TM, TK; TN gets remaining memory
        for TM in TMs:
            for TK in TKs:
                can_grow_m = False
                idx_m = TMs.index(TM)
                if idx_m > 0:
                    TM_next = TMs[idx_m - 1]
                    min_TN = TNs[-1] if TNs else None
                    if min_TN and hw.elem_bytes * (TM_next * TK + TK * min_TN + TM_next * min_TN) <= hw.L_mem_bytes:
                        can_grow_m = True

                can_grow_k = False
                idx_k = TKs.index(TK)
                if idx_k > 0:
                    TK_next = TKs[idx_k - 1]
                    min_TN = TNs[-1] if TNs else None
                    if min_TN and hw.elem_bytes * (TM * TK_next + TK_next * min_TN + TM * min_TN) <= hw.L_mem_bytes:
                        can_grow_k = True

                if can_grow_m or can_grow_k:
                    continue

                best_TN = None
                for TN in TNs:
                    if hw.elem_bytes * (TM * TK + TK * TN + TM * TN) <= hw.L_mem_bytes:
                        best_TN = TN
                        break
                if best_TN is not None:
                    TPm = _tp_from_tile(M, SPm, TM)
                    TPn = _tp_from_tile(N, SPn, best_TN)
                    TPk = K // TK
                    results.append((TPm, TPk, TPn))

    return results


def enumerate_configs_pruned(
    M: int, K: int, N: int, cfg: Config,
    rule1: bool = True,
    rule2: bool = True,
    rule3: bool = True,
    top_k_sp: int = 3,
) -> List[MappingConfig]:
    """Enumerate feasible configurations with structural pruning (Rules 1–3).

    Rule 1 (exact):     TP monotonicity — only C1-boundary TP combinations.
    Rule 2 (heuristic): TP allocation priority — maximize non-inner tiles first.
    Rule 3 (heuristic): SP pruning — top-k SP shapes per (P, d_inner) by N_dma.

    Pruning semantics: Rules 1/2/3 act as *filters* on the configuration space;
    configurations that survive all filters are emitted in the same canonical
    order as `enumerate_configs` (P → SPm (divisor order) → inner → TP).
    This guarantees that when the unpruned argmin is also in the pruned space,
    both searches yield the identical MappingConfig (no enumeration-order
    dependency; see also the tie-break key in `find_framework_optimal`).

    When all rules are disabled, equivalent to enumerate_configs.
    """
    hw = cfg.hw
    configs = []

    for P in hw.P_set:
        sp_pairs = [(SPm, P // SPm) for SPm in _divisors(P) if SPm * (P // SPm) == P]

        # ── Rule 3: precompute per-inner keep set (filter, not reorder) ──
        sp_keep: Dict[int, set] = {}
        for inner in [TPORDER_M_INNER, TPORDER_N_INNER, TPORDER_K_INNER]:
            if rule3:
                def _sp_sort_key(sp_pair, _inner=inner):
                    spm, spn = sp_pair
                    if _inner == TPORDER_K_INNER:
                        return spm + spn  # AM-GM: balanced is best
                    elif _inner == TPORDER_M_INNER:
                        return spm + 2 * spm * spn  # minimize SPm contribution
                    else:  # N_INNER
                        return spn + 2 * spm * spn  # minimize SPn contribution

                sp_ranked = sorted(sp_pairs, key=_sp_sort_key)
                sp_keep[inner] = set(sp_ranked[:top_k_sp])
            else:
                sp_keep[inner] = set(sp_pairs)

        # ── Canonical iteration: SPm in divisor order (matches enumerate_configs) ──
        for SPm in _divisors(P):
            SPn = P // SPm
            if SPm * SPn != P:
                continue

            for inner in [TPORDER_M_INNER, TPORDER_N_INNER, TPORDER_K_INNER]:
                if (SPm, SPn) not in sp_keep[inner]:
                    continue

                # ── Rules 1+2 / Rule 1 / No pruning for TP ─────────
                if rule1 and rule2:
                    tp_list = _enumerate_tp_rule12(M, K, N, SPm, SPn, inner, hw)
                elif rule1:
                    tp_list = _enumerate_tp_rule1(M, K, N, SPm, SPn, hw)
                else:
                    # No TP pruning: enumerate all feasible
                    tp_list = []
                    for TPm in range(1, M // SPm + 1):
                        TM = M // (SPm * TPm)
                        if M % (SPm * TPm) != 0 or TM % hw.TM_align != 0 or TM < hw.TM_align:
                            continue
                        for TPn in range(1, N // SPn + 1):
                            TN = N // (SPn * TPn)
                            if N % (SPn * TPn) != 0 or TN % hw.TN_align != 0 or TN < hw.TN_align:
                                continue
                            for TPk in range(1, K + 1):
                                TK = K // TPk
                                if K % TPk != 0 or TK % hw.TK_align != 0 or TK < hw.TK_align:
                                    continue
                                if hw.elem_bytes * (TM * TK + TK * TN + TM * TN) > hw.L_mem_bytes:
                                    continue
                                tp_list.append((TPm, TPk, TPn))

                for TPm, TPk, TPn in tp_list:
                    configs.append(MappingConfig(
                        M, K, N, SPm, SPn, TPm, TPk, TPn, inner
                    ))

    return configs


def find_framework_optimal(
    M: int, K: int, N: int,
    cfg: Config,
    wl_df: Optional[pd.DataFrame] = None,
    pruned: bool = False,
    rule_kwargs: Optional[Dict] = None,
) -> Optional[BaselineResult]:
    """Find the cost-model-optimal configuration.

    Args:
        pruned: If True, use structural pruning to reduce the search space.
                If False, exhaustive enumeration (rule_kwargs is ignored).
        rule_kwargs: Optional dict passed to ``enumerate_configs_pruned``
                when ``pruned=True``.  Lets callers evaluate arbitrary
                rule subsets (e.g. ``{rule1: True, rule2: False,
                rule3: False}`` to isolate Rule 1).  When None, all rules
                are enabled (matches the STAR-Map default).

    If wl_df is provided, looks up measured values for the chosen config.
    If not found in measurements, uses F3+D2 nearest-neighbor fallback and
    marks is_fallback=True (in_measurements remains True since a measured
    proxy is substituted).
    """
    if pruned:
        configs = enumerate_configs_pruned(M, K, N, cfg, **(rule_kwargs or {}))
    else:
        configs = enumerate_configs(M, K, N, cfg)
    if not configs:
        return None

    # Deterministic argmin with canonical tie-breaking.
    # Primary: predicted EDP. Secondary: lexicographic on intrinsic config
    # fields so that argmin depends only on the config, not on enumeration
    # order. Guarantees: when the full-space argmin is also in the pruned
    # space, both searches return the identical config — resolving
    # tie-induced asymmetries between SM-exh and STAR-Map.
    def _argmin_key(mc):
        return (
            edp(mc, cfg),
            mc.P, mc.SPm, mc.SPn,
            mc.TPm, mc.TPk, mc.TPn,
            mc.tpOrder_inner,
        )

    best_mc = min(configs, key=_argmin_key)
    best_edp = edp(best_mc, cfg)

    # Try to look up in measurements
    measured_time = np.nan
    measured_energy = np.nan
    measured_edp_val = np.nan
    in_meas = False
    is_fallback = False

    if wl_df is not None:
        match = wl_df[
            (wl_df["SPm"] == best_mc.SPm) &
            (wl_df["SPn"] == best_mc.SPn) &
            (wl_df["TPm"] == best_mc.TPm) &
            (wl_df["TPk"] == best_mc.TPk) &
            (wl_df["TPn"] == best_mc.TPn) &
            (wl_df["tpOrder_inner"] == best_mc.tpOrder_inner)
        ]
        if not match.empty:
            row = match.iloc[0]
            measured_time = float(row["time_us"])
            measured_energy = float(row["energy_uj"]) if row["energy_valid"] else np.nan
            measured_edp_val = float(row["edp_measured"]) if row["energy_valid"] else np.nan
            in_meas = True
        else:
            # Fallback: find the measured case with closest predicted EDP.
            # Sort by distance with canonical tie-breakers (numSpm, SPm, SPn,
            # TPm, TPk, TPn, tpOrder_inner) so the fallback pick is reproducible
            # regardless of the underlying DataFrame row order.
            valid_fb = wl_df[wl_df["energy_valid"] & (wl_df["edp_pred"] > 0)].copy()
            if not valid_fb.empty:
                valid_fb["_edp_dist"] = np.abs(valid_fb["edp_pred"] - best_edp)
                valid_fb = valid_fb.sort_values(
                    by=["_edp_dist", "numSpm", "SPm", "SPn",
                        "TPm", "TPk", "TPn", "tpOrder_inner"],
                    kind="mergesort",
                ).reset_index(drop=True)
                fb_row = valid_fb.iloc[0]
                measured_time = float(fb_row["time_us"])
                measured_energy = float(fb_row["energy_uj"])
                measured_edp_val = float(fb_row["edp_measured"])
                # Override config to match the fallback case
                best_mc = MappingConfig(
                    M, K, N,
                    int(fb_row["SPm"]), int(fb_row["SPn"]),
                    int(fb_row["TPm"]), int(fb_row["TPk"]), int(fb_row["TPn"]),
                    int(fb_row["tpOrder_inner"]),
                )
                best_edp = edp(best_mc, cfg)
                in_meas = True
                is_fallback = True

    return BaselineResult(
        workload=(M, K, N),
        P=best_mc.P,
        SPm=best_mc.SPm, SPn=best_mc.SPn,
        TPm=best_mc.TPm, TPk=best_mc.TPk, TPn=best_mc.TPn,
        tpOrder_inner=best_mc.tpOrder_inner,
        measured_time_us=measured_time,
        measured_energy_uj=measured_energy,
        measured_edp=measured_edp_val,
        pred_time=t_total(best_mc, cfg),
        pred_energy=e_total(best_mc, cfg),
        pred_edp=best_edp,
        in_measurements=in_meas,
        is_fallback=is_fallback,
    )


# ─── Batch processing ───────────────────────────────────────────────────────
@dataclass
class WorkloadBaselines:
    workload: WorkloadKey
    gt: Optional[BaselineResult]
    naive_max: Optional[BaselineResult]
    framework: Optional[BaselineResult]                # pruned (final)
    framework_exhaustive: Optional[BaselineResult]     # exhaustive (for comparison)
    charm_spk1: Optional[BaselineResult] = None        # CHARM-CDSE 1-level, SP_k=1
    timeloop: Optional[BaselineResult] = None          # Timeloop EDP (Cat B baseline)


def compute_all_baselines(
    groups: Dict[WorkloadKey, pd.DataFrame],
    cfg: Config,
) -> Dict[WorkloadKey, WorkloadBaselines]:
    """Compute GT, Naive-Max, Framework, CHARM-CDSE, and Timeloop baselines."""
    # Local imports to avoid circular dependency (baseline files import
    # enumerate_configs / BaselineResult from this module).
    from baselines_charm import find_charm_cdse
    from baselines_timeloop import find_timeloop

    results = {}
    for wl in cfg.WORKLOADS:
        wl_df = groups.get(wl)
        if wl_df is None or wl_df.empty:
            results[wl] = WorkloadBaselines(
                wl, None, None, None, None, None, None
            )
            continue

        gt = find_gt(wl_df, wl)
        naive = find_naive_max(wl[0], wl[1], wl[2], cfg, wl_df)
        fw = find_framework_optimal(wl[0], wl[1], wl[2], cfg, wl_df, pruned=True)
        fw_exh = find_framework_optimal(wl[0], wl[1], wl[2], cfg, wl_df, pruned=False)
        charm = find_charm_cdse(wl[0], wl[1], wl[2], cfg, wl_df, spk_mode="fixed")
        timeloop = find_timeloop(wl[0], wl[1], wl[2], cfg, wl_df)
        results[wl] = WorkloadBaselines(
            wl, gt, naive, fw, fw_exh, charm, timeloop
        )

    return results