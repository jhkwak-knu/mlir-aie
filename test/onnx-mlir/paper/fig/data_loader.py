"""
data_loader.py — Load and parse measurement data from result CSV and tc_list JSON.

Handles CSV quirks (extra columns, duplicate headers, status strings)
and groups data by workload (M, K, N).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config
from cost_model import tporder_from_list


WorkloadKey = Tuple[int, int, int]  # (M, K, N)


def load_csv(path: Path) -> pd.DataFrame:
    """Load result CSV, handling its quirks.

    Supports two header styles:
      - v11: header is the very first line; comments are data rows with -1 padding.
      - v12+: comments start with '#'; header is the first non-comment line.
    """
    with open(path) as f:
        lines = f.readlines()

    # Find header: first non-empty line that starts with "case_index"
    header_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("case_index"):
            header_idx = i
            break

    header_line = lines[header_idx].strip()
    n_cols = len(header_line.split(","))
    header = header_line.split(",")[:n_cols]

    # Parse data rows (skip comments and duplicate header rows)
    data_rows = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("case_index"):
            continue
        vals = stripped.split(",")[:n_cols]
        data_rows.append(vals)

    df = pd.DataFrame(data_rows, columns=header)

    # Type conversion
    str_cols = {"status", "doubleBuffer"}
    for col in df.columns:
        if col in str_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # NOTE: PASS filtering is deferred to load_all() so that enrich_csv_with_json()
    # can use positional matching with tc_list before rows are removed.

    # Compute measured EDP using step_min (wall time including host overhead)
    # and batch_min_energy_per_iter_uj (energy per iteration, batch-min filtered)
    # Fallback to npu_energy_per_iter_uj if batch_min is unavailable
    df["energy_uj"] = df["batch_min_energy_per_iter_uj"].where(
        df["batch_min_energy_per_iter_uj"] > 0,
        df["npu_energy_per_iter_uj"]
    )
    df["time_us"] = df["batch_min_avg_us"].where(
        df["batch_min_avg_us"] > 0,
        df["min_us"]
    )
    df["energy_valid"] = df["energy_uj"] > 0
    df["edp_measured"] = np.where(
        df["energy_valid"],
        df["time_us"] * df["energy_uj"],
        np.nan,
    )

    # Cast workload keys to int
    for col in ["M", "K", "N", "numSpm", "SPm", "SPn", "TPm", "TPk", "TPn"]:
        df[col] = df[col].astype(int)

    return df


def load_tc_list(path: Path) -> Tuple[dict, List[dict]]:
    """Load tc_list JSON, return (metadata, cases)."""
    with open(path) as f:
        data = json.load(f)
    return data["metadata"], data["cases"]


def enrich_csv_with_json(df: pd.DataFrame, cases: List[dict]) -> pd.DataFrame:
    """Add tpOrder_inner from JSON to CSV dataframe.

    Only extracts tpOrder_inner (loop ordering) from JSON, which is not
    recorded in the CSV.  Prediction values are no longer read from JSON
    because they may be stale after re-calibration; use
    ``compute_predictions`` to generate them from the current model.
    """
    # Build lookup: case_index (1-based position) -> tpOrder_inner
    tp_by_ci = {
        i + 1: tporder_from_list(case["levels"][0]["tpOrder"])
        for i, case in enumerate(cases)
    }
    df["tpOrder_inner"] = df["case_index"].map(tp_by_ci)
    n_missing = df["tpOrder_inner"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} CSV rows have no matching tc_list entry (dropped)")
        df = df.dropna(subset=["tpOrder_inner"])
    df["tpOrder_inner"] = df["tpOrder_inner"].astype(int)
    return df


def compute_predictions(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute t/e/edp predictions from the current calibration coefficients.

    Replaces stale CSV/JSON predictions with fresh values calculated via
    the cost model, ensuring consistency with the latest calibration.
    """
    from cost_model import MappingConfig, t_total, e_total

    t_preds = []
    e_preds = []
    for _, row in df.iterrows():
        try:
            mc = MappingConfig(
                M=int(row["M"]), K=int(row["K"]), N=int(row["N"]),
                SPm=int(row["SPm"]), SPn=int(row["SPn"]),
                TPm=int(row["TPm"]), TPk=int(row["TPk"]), TPn=int(row["TPn"]),
                tpOrder_inner=int(row["tpOrder_inner"]),
            )
            t_cy = t_total(mc, cfg)   # cycles
            e_uj = e_total(mc, cfg)   # µJ
            t_preds.append(t_cy)
            e_preds.append(e_uj)
        except (ValueError, KeyError):
            t_preds.append(np.nan)
            e_preds.append(np.nan)

    df["t_total_pred"] = t_preds              # cycles (현재 모델)
    df["e_total_pred"] = e_preds              # µJ    (현재 모델)
    df["edp_pred"] = df["t_total_pred"] * df["e_total_pred"]
    return df


def group_by_workload(df: pd.DataFrame) -> Dict[WorkloadKey, pd.DataFrame]:
    """Group dataframe by (M, K, N) workload."""
    groups = {}
    for (m, k, n), grp in df.groupby(["M", "K", "N"]):
        groups[(int(m), int(k), int(n))] = grp.copy()
    return groups


# ─── High-level loader ───────────────────────────────────────────────────────
def load_all(cfg: Config) -> Tuple[pd.DataFrame, Dict[WorkloadKey, pd.DataFrame]]:
    """Load all data, enrich with JSON, compute predictions, group by workload.

    Returns:
        df: Full dataframe with all measurements + current-model predictions
        groups: Dict mapping (M,K,N) -> per-workload DataFrame
    """
    df = load_csv(cfg.result_csv)
    _, cases = load_tc_list(cfg.tc_list_json)
    df = enrich_csv_with_json(df, cases)   # matches by case_index
    df = df[df["status"] == "PASS"].copy()  # filter PASS only

    # Exclude tpOrder verification cases (added post-hoc for tpOrder sweep;
    # not part of the original calibration / evaluation set).
    verify_ci = {i + 1 for i, c in enumerate(cases) if c.get("tporder_verify", False)}
    df = df[~df["case_index"].isin(verify_ci)].copy()

    df = compute_predictions(df, cfg)
    groups = group_by_workload(df)
    return df, groups
