"""
baselines_charm.py — CHARM-CDSE 1-level baseline adapted to XDNA2.

Reference:
    Jinming Zhuang et al., "CHARM: Composing Heterogeneous AcceleRators
    for Matrix Multiply on Versal ACAP Architecture," FPGA 2023.
    GitHub: arc-research-lab/CHARM

Source verification status (as of 2026-04-18)
---------------------------------------------
  CROSS-REFERENCED against arc-research-lab/CHARM@main on 2026-04-18.
  Files inspected: CDSE/cdse.py (main DSE driver and cost model).
  Files still UN-inspected (lower priority for core cost-model claims):
    * CDSE/broadcast_tuning.py
    * CDSE/buffer_sel.py

  Confirmed against source:
    * AIE-array variable names (a, b, c) and inner-tile names (x, y, z).
    * Existence of b (SPk, K-dimension split) in the original CHARM
      search space — our Variant B restriction b = 1 is a genuine
      reduction of CHARM's search space, not a cosmetic simplification.
    * Broadcast-reuse byte formulas (TILEL_SIZE = a·b·H1·W1·DT etc.)
      reduce to our formulas exactly when b = 1.
    * Feasibility constraint form a·b·c ≤ AIE_NUM (we replaced 400
      with P_total = 32).

  Corrected against source (PRIOR DOCSTRING HAD INACCURACIES):
    * CHARM is actually 4-level tiling, not 2-level:
        (i)   single-AIE inner loops (x, y, z) around COMPUTE_CYCLE
        (ii)  AIE-array spatial partition (a, b, c)
        (iii) DDR→L1 buffering (TILEL/TILER/TILEO)
        (iv)  outer tile counts (X_TILE, Y_TILE, Z_TILE) and
              multi-workload weighting (MODEL_IN[:,3])
      Our collapse to 1-level drops levels (i), (iii), (iv) — more than
      the "just memtile" story the prior docstring suggested.
    * CHARM's per-workload aggregation is NOT a uniform max(c, l, s).
      It is a case-split over (X_TILE, Y_TILE, Z_TILE). The single-tile
      case is already partially sum-like:
          temp = max(load_L_DR, load_R_DL) + AIE_CYCLE + store_O_S
      Multi-tile cases use explicit pipeline patterns with T-versions
      (middle-tile reuse). Our "max → sum" narrative was a simplification
      of the original, not a reversal of a pure max formulation.
    * Overheads that we zeroed (Adaptation #5) are specifically:
      (a) PACK_IN / PACK_OUT packing terms in AIE_CYCLE, (b) the
      COMPUTE_CYCLE +40 accumulator-stall baseline, (c) the multi-tile
      T-version pipeline costs.
    * Tie-break: CHARM uses np.argsort by throughput (descending) and
      picks [0]; there is no explicit secondary tie-break. Our
      (P, TP_total) secondary key is an ADDITION for determinism.

  Items still to verify if time permits:
    * What broadcast_tuning.py does (it may affect effective
      BW_L_DR / BW_R_DL values we haven't modelled).
    * What buffer_sel.py does (likely L1 buffer size selection that
      interacts with feasibility constraints).
    * Whether MODEL_IN[:,3] is multi-workload weight in the single-op
      entry point we're comparing against (we assume weight = 1).

  The "Adapted CHARM-CDSE" description (what THIS file actually
  implements) is exact — the above corrections refine our description
  of the original CHARM, not our adaptation.

=============================================================================
ORIGINAL vs ADAPTED — for paper methodology / reviewer defense
=============================================================================

Original CHARM-CDSE (Versal VCK190, AIE1) — from CDSE/cdse.py
-------------------------------------------------------------
  * Target      : Versal VCK190 (400 AIE tiles, freq 230/250 MHz)
  * Memory path : DDR → PLIO → Memtile (L2 staging) → AIE L1 → compute
  * Tiling      : 4-LEVEL:
                    (i)   single-AIE inner loops (x, y, z) around
                          COMPUTE_CYCLE + L1 buffer loads
                    (ii)  AIE-array spatial partition (a, b, c) along
                          (M, K, N) axes
                    (iii) DDR→L1 buffering of TILEL/TILER/TILEO blocks
                    (iv)  outer tile counts (X_TILE, Y_TILE, Z_TILE)
                          and multi-workload weighting (MODEL_IN[:,3])
  * Objective   : min weighted throughput-cycle over workload set
  * Constants   : AIE_NUM=400, BRAM≤867, URAM≤420, PLIO_IN≤100,
                  PLIO_OUT≤80, placement_verify ≤ 50, per-dtype
                  alignment (a%2==0 int16; b%PACK_IN==0; c%PACK_OUT==0),
                  freq 230/250 MHz

  * Variable inventory:
        a, b, c       AIE-array sizes along (M, K, N) — spatial
        x, y, z       inner tile counts per AIE along (M, K, N)
        H1, W1, W2    single-AIE physical tile sizes (one inner iter)
        X_TILE        = ceil(M / (x·a·H1))    outer M-tile count
        Y_TILE        = ceil(K / (y·b·W1))    outer K-tile count
        Z_TILE        = ceil(N / (z·c·W2))    outer N-tile count
        DATA_TYPE     elem_bytes (e.g. int16 → 2)
        COMPUTE_CYCLE kernel cycles per inner iter (dtype-specific,
                      e.g. int8 ≈ 5040, int16 ≈ 3840, fp32 ≈ 2080;
                      already includes +40 accumulator-stall baseline)
        PACK_IN/OUT   input/output packing factors (2–4)
        MODEL_IN[:,3] per-workload weight used for weighted sum

  * Cost model (verbatim structure from cdse.py):

        # Inner AIE cycle — max over (L-buffer load, R-buffer load,
        # compute) × per-AIE inner loops, + packing amortisation.
        AIE_CYCLE = ceil(max(H1·W1·DATA_TYPE//4,
                             W1·W2·DATA_TYPE//4,
                             COMPUTE_CYCLE)) · x·y·z
                  + (H1·W1·DATA_TYPE//4) · PACK_IN
                  + (H1·W2·DATA_TYPE//4) · PACK_OUT

        # DDR→L1 block movement (bytes → cycles).
        TILEL_SIZE = a·b · LEFT_SIZE  · DATA_TYPE · NUM_PER_PORT_A
        TILER_SIZE = b·c · RIGHT_SIZE · DATA_TYPE · NUM_PER_PORT_B
        TILEO_SIZE = a·c · OUT_SIZE   · DATA_TYPE · NUM_PER_PORT_C

        load_L_DR = ceil(TILEL_SIZE · x·y / BW_L_DR)
        load_R_DL = ceil(TILER_SIZE · y·z / BW_R_DL)
        store_O_S = ceil(TILEO_SIZE · x·z / BW_O_S)

        # Per-workload temp_cycle — CASE-SPLIT, not uniform max:
        # Case A (single tile, X_TILE==Y_TILE==Z_TILE==1):
        temp = max(load_L_DR, load_R_DL) + AIE_CYCLE + store_O_S

        # Case B (multi X or Z tile, Y_TILE==1) — pipelined with
        # T-versions (middle-tile reuse):
        temp = max(load_L_DR, load_R_DL)
             + max(load_L_DR, load_R_DL, AIE_CYCLE)
             + max(AIE_CYCLE, store_O_S)
             + max(load_L_T, load_R_T, AIE_CYCLE, store_O_T)
               · (X_TILE·Y_TILE·Z_TILE - 2)
             + store_O_S

        # Case C (Y_TILE > 1) — K-accumulation pipelining:
        temp = max(load_L_DR, load_R_DL)
             + max(load_L_DR, load_R_DL, AIE_CYCLE)
               · ((X_TILE·Y_TILE·Z_TILE - 1) - (X_TILE·Z_TILE - 1))
             + max(load_L_T, load_R_T, AIE_CYCLE, store_O_T)
               · (X_TILE·Z_TILE - 1)
             + AIE_CYCLE + store_O_S

        # Multi-workload aggregation:
        total_cycle = sum(temp_cycle · MODEL_IN[:,3])

  * Search procedure (cdse.py):
        1. Enumerate feasible (a, b, c, x, y, z) under
           a·b·c ≤ AIE_NUM, BRAM/URAM/PLIO, placement_verify,
           dtype-alignment, factorisation over (M, K, N).
        2. Compute total_cycle for each tuple.
        3. Sort by throughput = total_ops / total_cycle DESCENDING.
        4. Pick config[0, :]  (no explicit secondary tie-break).

Adapted CHARM-CDSE (this paper, XDNA2 Ryzen AI 9 HX 370)
--------------------------------------------------------
  * Target      : XDNA2 (32 compute tiles, 1500 MHz)
  * Memory path : System memory → compute tile  (memtile bypassed)
  * Tiling      : 1-LEVEL — only (TP_m, TP_k, TP_n) outer count,
                  collapsed from CHARM's 4-level:
                    (x, y, z) inner + (X_TILE, Y_TILE, Z_TILE) outer
                    and DDR→L1 block buffering are all folded into
                    TP_total = TP_m · TP_k · TP_n. Only single-workload
                    evaluation (no MODEL_IN weighting).
  * Objective   : PRESERVED — min throughput-cycle, energy-agnostic,
                  single workload.
  * Per-iter    : cycle = compute + load + store
                  (sequential; matches our t_total execution model).
                  This is CHARM's Case A single-tile form with the
                  max(load_L_DR, load_R_DL) inner max replaced by
                  load_L_DR + load_R_DL — i.e. further simplified.
  * Constants   : from cfg.hw / cfg.perf — eff_macs=24.28, bw_eff_bpc=4.0,
                  L_mem_bytes=61440, align {TM=8, TK=8, TN=16}

Variable mapping
----------------
    CHARM (4-level, VCK190)         Ours (1-level, XDNA2)
      a, b, c                        SP_m, SP_k, SP_n   (spatial)
                                     † SP_k is forced to 1 (Variant B)
      x · H1, y · W1, z · W2         TM, TK, TN         (per-core tile
                                     size; we fold inner counts and
                                     physical tile size into one TM/TK/TN)
      X_TILE · Y_TILE · Z_TILE       TP_total           (outer iter count)
      MODEL_IN[:,3] weighted sum     single workload    (we score one
                                                         workload at a time)
      AIE_NUM = 400                  cfg.hw.P_total = 32
      BW_L_DR / BW_R_DL / BW_O_S     cfg.perf.bw_eff_bpc = 4.0
      BRAM / URAM / PLIO             cfg.hw.L_mem_bytes = 61440
      COMPUTE_CYCLE (incl. +40       cfg.perf.eff_macs = 24.28 MAC/cycle
          accumulator stall)         (stall is absorbed into calibrated
                                      eff_macs; not modelled separately)
      PACK_IN / PACK_OUT, H1/W1/W2   not modelled       (zeroed — see #5)

What is PRESERVED from the original CHARM
------------------------------------------
  * Optimisation objective — minimise predicted throughput-cycle
    (energy-agnostic); this is CDSE's core design choice.
  * Search space — same SP × TP enumeration via enumerate_configs (the
    same space STAR-Map and Naive-Max use).
  * Broadcast reuse assumption — A shared across SP_n cores in a row;
    B shared across SP_m cores in a column; C unique per core.
  * Per-iteration byte formulas:
        load_A  = SP_m · TM · TK · elem_bytes
        load_B  = SP_n · TK · TN · elem_bytes
        store_C =   P  · TM · TN · elem_bytes
  * CDSE philosophy — configuration-space exploration scored by a
    predicted cycle cost function.

What is ADAPTED (and why)
-------------------------
  1. Tiling levels (4-level → 1-level)
     Reason: XDNA2 bypasses the Versal memtile-based 2-stage DDR→L1
     pipeline that CHARM exploits. We also drop CHARM's inner (x, y, z)
     decomposition around a single AIE kernel, because our per-tile
     throughput is characterised directly by cfg.perf.eff_macs. The
     remaining outer iteration count becomes TP_total. Effect: CHARM's
     Case-B and Case-C pipelined formulas collapse to Case-A plus
     further simplification (see #2).

  2. Per-iteration cycle aggregation                        [stage_mode]
     CHARM original Case A (single tile):
         temp = max(load_L_DR, load_R_DL) + AIE_CYCLE + store_O_S
     CHARM original Case B / C (multi-tile): pipelined with T-versions.
     Our adaptation (default, stage_mode="sum"):
         temp = load_L_DR + load_R_DL + compute + store
     So we drop two things:
       (a) the inner max(·,·) over the two DDR loads
           (CHARM assumes L and R DMAs can overlap; we charge them
            sequentially because XDNA2's current single-BD streaming
            path doesn't expose dual-stream overlap in our execution
            pipeline),
       (b) the multi-tile T-version pipeline (middle-tile reuse),
           because we have no memtile to hold the middle tile.
     Effect: sum aggregation matches STAR-Map's cost_model.t_total
     (= t_comp + t_comm + t_overhead), so the comparison isolates the
     optimisation objective (throughput-cycle vs EDP) rather than
     mixing in a different hardware assumption.
     Switch: stage_mode="max" replaces our default sum with the flat
     `max(c, l, s)` (not CHARM's exact original formula — a strictly
     more-overlap-friendly variant). Retained for sensitivity /
     rebuttal; not reported in the main paper.

  3. SP_k fixed to 1 (Variant B only)                       [spk_mode]
     Reason: our MappingConfig assumes SP_k = 1 (§ 3.3.1 of the paper).
     CHARM's original DSE allows b > 1 (K-dimension spatial split with
     partial-sum reduction); restricting b = 1 genuinely reduces
     CHARM's search space. Variant A (SP_k > 1 allowed) would require
     extending enumerate_configs and is left as future work.

  4. Constants replaced with XDNA2-calibrated values
     Reason: using VCK190 constants on an XDNA2 target would produce
     nonsensical cycle predictions. We use our measured values so
     that CHARM's pick is evaluated under the same hardware model
     STAR-Map uses. Note: the COMPUTE_CYCLE +40 accumulator-stall
     baseline is absorbed implicitly into our calibrated eff_macs;
     we do not track a separate stall term.

  5. Original CHARM overhead terms zeroed
     The CHARM overheads that are explicitly absent from our model:
       (a) PACK_IN / PACK_OUT packing amortisation terms added to
           AIE_CYCLE — XDNA2 kernel-throughput already sits in a
           post-packing regime, so re-applying them would double-count.
       (b) T-version pipeline costs in multi-tile cases — absent in
           1-level flattening.
       (c) placement_verify cost and plio constraint residuals.
     Reason: these are Versal-specific overheads without direct XDNA2
     analogues. Zeroing is the most favourable interpretation for
     CHARM — it is never charged an overhead that STAR-Map pays.

Tie-break policy
----------------
  CHARM original: `config = config[config[:,0].argsort()[::-1]]; config[0,:]`
      — sort by throughput (ops/cycle) descending and take [0]. Ties
      are resolved by the (unstable-in-general) argsort stability
      order, which depends on enumeration order. There is NO explicit
      secondary key.

  Our adaptation: (min cycle, min P, min TP_total). The (P, TP_total)
      secondary key is an ADDITION for determinism — it favours
      simpler configurations on ties. Effect on final pick is small
      in most workloads but non-zero; it should be stated explicitly
      in the paper methodology.

Limitations / Honest caveats
----------------------------
  * Because sum aggregation is used, CHARM's predicted cycle model
    becomes structurally similar to our t_total (differing only in the
    overhead term, which CHARM sets to 0). The principal remaining
    difference between CHARM and STAR-Map is therefore the
    optimisation objective (throughput-cycle vs EDP), not the cycle
    predictor. This is intentional for fair comparison.
  * SP_k = 1 restriction may understate CHARM's search space; Variant A
    is marked as future work.
  * CHARM's original max formulation is retained as a switch (not
    deleted) so the adaptation is reversible and auditable.

Pending verification (remaining, arc-research-lab/CHARM)
--------------------------------------------------------
  Resolved against CDSE/cdse.py on 2026-04-18:
    ✓ Cost-function expression (documented above).
    ✓ Broadcast-reuse byte formulas (match ours when b = 1).
    ✓ Tie-breaking rule (argsort; ours adds (P, TP_total)).
    ✓ Variable-name mapping (documented above).
    ✓ AIE_NUM constraint form (a·b·c ≤ 400).
    ✓ Capacity budget (BRAM≤867, URAM≤420, PLIO_IN≤100, PLIO_OUT≤80).

  Still outstanding (lower priority, affects only peripheral detail):
    1. CDSE/broadcast_tuning.py — may add effective-BW adjustment
       terms we haven't modelled (BW_L_DR / BW_R_DL calibration).
    2. CDSE/buffer_sel.py — L1 buffer size selection; likely only
       affects feasibility enumeration, not the cost function itself.
    3. Whether the single-op entry path we're comparing against uses
       MODEL_IN[:,3] = 1 (our implicit assumption) or a different
       weight. This matters if CHARM's published single-GEMM numbers
       were produced with a weight ≠ 1 in the paper.

Variant policy (spk_mode)
-------------------------
  * spk_mode="fixed" (Variant B) — SP_k = 1, reuses enumerate_configs.
  * spk_mode="free"  (Variant A) — not implemented. Requires extending
    MappingConfig with SP_k and generalising the enumerator.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from config import Config
from cost_model import MappingConfig, edp, e_total, t_total


# ─── CHARM 1-level cycle model ────────────────────────────────────────────────
def charm_cycle_cost(
    mc: MappingConfig,
    cfg: Config,
    stage_mode: Literal["sum", "max"] = "sum",
) -> float:
    """CHARM-CDSE 1-level cycle prediction.

    Per-iteration (one tile step over the GEMM) bytes to move from
    system memory, assuming broadcast reuse of A across SP_n cores in a
    row and B across SP_m cores in a column, and unique partial-output
    C per core (CHARM's original formulation — no K-inner partial-sum
    amortisation):

        load_A_iter  = SP_m * TM * TK * elem_bytes   (A broadcast)
        load_B_iter  = SP_n * TK * TN * elem_bytes   (B broadcast)
        store_C_iter = P   * TM * TN * elem_bytes    (C per core)

    Per-iteration cycle cost of each stage (aggregate DMA BW):

        compute_cyc = TM * TK * TN / eff_macs        (per core, parallel)
        load_cyc    = (load_A_iter + load_B_iter) / bw_eff_bpc
        store_cyc   = store_C_iter / bw_eff_bpc

    Per-iteration aggregation:

        stage_mode="sum"  (default, matches our sequential XDNA2
            execution; same philosophy as cost_model.t_total which
            uses t_comp + t_comm + t_overhead):

                AIE_CYCLE = compute_cyc + load_cyc + store_cyc

        stage_mode="max"  (CHARM's original assumption of
            double-buffered overlap on Versal with memtile staging):

                AIE_CYCLE = max(compute_cyc, load_cyc, store_cyc)

    Total cycles over TP_m * TP_k * TP_n outer iterations (overhead=0):

        total_cycle = AIE_CYCLE * TP_m * TP_k * TP_n

    Rationale for "sum" default:
        Our target bypasses memtile and lacks the pipelined L2 staging
        that CHARM relied on.  Using "sum" gives CHARM the same
        execution model as STAR-Map, so the comparison isolates the
        optimisation objective (throughput-cycle vs EDP) rather than
        mixing in a different hardware assumption.
    """
    hw = cfg.hw
    perf = cfg.perf

    TM, TK, TN = mc.TM, mc.TK, mc.TN
    SPm, SPn, P = mc.SPm, mc.SPn, mc.P

    # Per-iteration bytes — PRESERVED from CHARM (broadcast reuse: A across
    # SP_n cores in a row, B across SP_m cores in a column, C unique per core).
    load_A = SPm * TM * TK * hw.elem_bytes
    load_B = SPn * TK * TN * hw.elem_bytes
    store_C = P * TM * TN * hw.elem_bytes

    # Per-iteration cycles — ADAPTED: constants from cfg.perf (eff_macs,
    # bw_eff_bpc) replace CHARM's Versal-specific PLIO/AIE_NUM/freq values.
    compute_cyc = (TM * TK * TN) / perf.eff_macs
    load_cyc = (load_A + load_B) / perf.bw_eff_bpc
    store_cyc = store_C / perf.bw_eff_bpc

    # Stage aggregation — ADAPTED (see module docstring, Adaptation #2):
    #   "sum"  → sequential execution (XDNA2 memtile-bypass, default for paper)
    #   "max"  → CHARM's original pipelined assumption (retained for
    #            sensitivity / rebuttal; not used in main paper because it
    #            assumes memtile L2 double-buffering that XDNA2 lacks).
    if stage_mode == "sum":
        aie_cycle = compute_cyc + load_cyc + store_cyc
    elif stage_mode == "max":
        aie_cycle = max(compute_cyc, load_cyc, store_cyc)
    else:
        raise ValueError(f"Unknown stage_mode: {stage_mode!r}")

    # Outer iteration count — ADAPTED: CHARM's 2-level (Memtile × AIE) is
    # collapsed to 1-level TP_total. Per-iter overhead and stream-count
    # compensation are set to 0 (most favourable to CHARM; see #5).
    total_cycle = aie_cycle * mc.TP_total
    return total_cycle


# ─── CHARM-CDSE selector ──────────────────────────────────────────────────────
def find_charm_cdse(
    M: int, K: int, N: int,
    cfg: Config,
    wl_df: Optional[pd.DataFrame] = None,
    spk_mode: Literal["fixed", "free"] = "fixed",
    stage_mode: Literal["sum", "max"] = "sum",
) -> Optional["BaselineResult"]:
    """CHARM-CDSE baseline: minimise throughput-cycle under own cost model.

    Algorithm (variant B, spk_mode="fixed"):
      1. Enumerate the same feasible configuration space used by
         `enumerate_configs` (SP_k = 1, memory / alignment constraints
         identical to our other baselines).
      2. Score every candidate with `charm_cycle_cost` and pick the one
         that minimises the CHARM cycle prediction (ties broken by
         fewest P, then fewest TP_total — favours simpler configs).
      3. Look up measured T/E/EDP for the chosen config in the result
         CSV, falling back to the nearest-predicted-EDP measured case
         when the exact config is missing (same pattern as
         `find_naive_max`).

    Returns a BaselineResult compatible with the rest of the pipeline,
    populated with CHARM's own pick but scored using *our* T / E / EDP
    cost model so the comparison plot evaluates all baselines on the
    same yardstick.
    """
    # Local import to avoid circular dependency (baselines imports this).
    from baselines import enumerate_configs, BaselineResult

    # ADAPTED (see module docstring, Adaptation #3): only Variant B is
    # supported — SP_k fixed to 1 because our MappingConfig assumes SP_k = 1
    # (§ 3.3.1). Variant A would require generalising enumerate_configs.
    if spk_mode != "fixed":
        raise NotImplementedError(
            "CHARM-CDSE variant A (SP_k > 1 allowed) is not implemented yet. "
            "Current enumerate_configs/MappingConfig assume SP_k = 1."
        )

    all_configs = enumerate_configs(M, K, N, cfg)
    if not all_configs:
        return None

    # CHARM objective: min total_cycle under its own cost model.
    # Tie-break: smaller P, then smaller TP_total (prefer simpler shape).
    def _charm_key(mc: MappingConfig):
        return (charm_cycle_cost(mc, cfg, stage_mode), mc.P, mc.TP_total)

    best_mc = min(all_configs, key=_charm_key)

    # ── Measurement lookup (mirror of find_naive_max) ────────────────────
    measured_time = np.nan
    measured_energy = np.nan
    measured_edp_val = np.nan
    in_meas = False
    is_fb = False

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
            measured_energy = (
                float(row["energy_uj"]) if row["energy_valid"] else np.nan
            )
            measured_edp_val = (
                float(row["edp_measured"]) if row["energy_valid"] else np.nan
            )
            in_meas = True
        else:
            # Fallback (F3+D2 policy, 2026-04-18):
            #   Preserve CHARM's core structural decisions (P, SPm, SPn)
            #   and snap to the measured config with the lowest
            #   CHARM-cycle cost within that constrained subspace.  Uses
            #   CHARM's OWN cost model (`charm_cycle_cost`) for the
            #   distance metric instead of STAR-Map's `edp_pred`, keeping
            #   the fallback consistent with CHARM's 1-stage objective.
            #
            #   Two-stage policy:
            #     Stage 1: (P, SPm, SPn) fixed — preserves SP shape.
            #     Stage 2: (P) fixed only — relaxes SP when Stage 1
            #              yields no measured config (avoids NaN).
            #
            #   Prior policy (nearest `edp_pred` at fixed P) is retained
            #   in git history; it let the fallback substitute degenerate
            #   SP shapes (e.g., SP=32×1 for 1024×1024×1024) because
            #   STAR-Map's EDP surface flattened toward K-inner, which
            #   CHARM never would have chosen.
            def _charm_score_row(row) -> float:
                row_mc = MappingConfig(
                    M, K, N,
                    int(row["SPm"]), int(row["SPn"]),
                    int(row["TPm"]), int(row["TPk"]), int(row["TPn"]),
                    int(row["tpOrder_inner"]),
                )
                return charm_cycle_cost(row_mc, cfg, stage_mode)

            # Stage 1: (P, SPm, SPn) fixed
            valid_fb = wl_df[
                wl_df["energy_valid"]
                & (wl_df["edp_measured"] > 0)
                & (wl_df["numSpm"] == best_mc.P)
                & (wl_df["SPm"] == best_mc.SPm)
                & (wl_df["SPn"] == best_mc.SPn)
            ].copy()

            # Stage 2: relax to (P) fixed if Stage 1 empty
            if valid_fb.empty:
                valid_fb = wl_df[
                    wl_df["energy_valid"]
                    & (wl_df["edp_measured"] > 0)
                    & (wl_df["numSpm"] == best_mc.P)
                ].copy()

            if not valid_fb.empty:
                valid_fb["_charm_cyc"] = valid_fb.apply(_charm_score_row, axis=1)
                fb_idx = valid_fb["_charm_cyc"].idxmin()
                fb_row = valid_fb.loc[fb_idx]
                measured_time = float(fb_row["time_us"])
                measured_energy = float(fb_row["energy_uj"])
                measured_edp_val = float(fb_row["edp_measured"])
                best_mc = MappingConfig(
                    M, K, N,
                    int(fb_row["SPm"]), int(fb_row["SPn"]),
                    int(fb_row["TPm"]), int(fb_row["TPk"]), int(fb_row["TPn"]),
                    int(fb_row["tpOrder_inner"]),
                )
                in_meas = True
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
