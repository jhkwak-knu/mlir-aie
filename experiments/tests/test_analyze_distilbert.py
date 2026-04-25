"""Unit tests for analyze_distilbert_step6.py.

Cover the three pure aggregations on a synthetic measurements.csv:
  * aggregate_per_setter_kernel (min over batches per kernel row)
  * edp_decomposition_by_gemm_type (sum across layers per gemm_type)
  * per_setter_summary (model-level T / E / EDP, sorted by EDP)

The sample fixture is small but exercises every grouping key the
production script depends on (setter, gemm_type, layer_idx, sub_type,
n_inner, batch_idx).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import analyze_distilbert_step6 as mod


_COLUMNS = [
    "measurement_type", "setter", "backend", "layer", "batch_idx", "n_inner",
    "time_us_min", "time_us_mean", "energy_uj_min_package", "wall_s",
    "idle_power_pre_mw", "idle_power_post_mw",
    "batch_energy_cv_pct", "batch_time_cv_pct",
    "gemm_type", "layer_idx", "sub_type",
]


def _row(measurement_type, setter, layer, batch_idx, n_inner,
         t_us_min, e_uj_min, gemm_type="", layer_idx=-1, sub_type=""):
    return {
        "measurement_type": measurement_type, "setter": setter, "backend": "npu",
        "layer": layer, "batch_idx": batch_idx, "n_inner": n_inner,
        "time_us_min": t_us_min, "time_us_mean": t_us_min * 1.05,
        "energy_uj_min_package": e_uj_min, "wall_s": 0.5,
        "idle_power_pre_mw": 3000.0, "idle_power_post_mw": 3000.0,
        "batch_energy_cv_pct": 2.0, "batch_time_cv_pct": 1.5,
        "gemm_type": gemm_type, "layer_idx": layer_idx, "sub_type": sub_type,
    }


@pytest.fixture
def synthetic_measurements_csv(tmp_path: Path) -> Path:
    """Two setters, two layers, two gemm_types, two batches.

    Construction:
      * star_map model  : t=100us, e=1000uJ at batch 0; t=110us, e=1100uJ at batch 1
                          n_inner=2 -> e_per_inf_uj = energy / n_inner
                          min across batches: t_min=100, e_min=500 (batch 0)
      * max_p model     : t=120us, e=1300uJ at batch 0; t=125, e=1320 at batch 1
                          n_inner=2 -> min: t_min=120, e_min=650
      * star_map L0/attn_qkv/Q (gemm_type=attention_qkv): 5us/100uJ over 2 batches
      * star_map L1/attn_qkv/Q                          : 6us/110uJ
      * star_map L0/ffn_expand                          : 20us/400uJ
      * max_p   L0/attn_qkv/Q                           : 7us/130uJ
      * max_p   L1/attn_qkv/Q                           : 8us/140uJ
      * max_p   L0/ffn_expand                           : 22us/430uJ
    """
    rows = []
    # Model rows (two batches each setter)
    rows.append(_row("model", "star_map", "_", 0, 2, 100.0, 1000.0))
    rows.append(_row("model", "star_map", "_", 1, 2, 110.0, 1100.0))
    rows.append(_row("model", "max_p",    "_", 0, 2, 120.0, 1300.0))
    rows.append(_row("model", "max_p",    "_", 1, 2, 125.0, 1320.0))

    # Kernel rows: 3 layers x 2 setters x 2 batches
    kernel_specs = [
        ("star_map", "L0/attention_qkv/Q", "attention_qkv", 0, "Q",
         [(0, 5.0,  100.0), (1, 6.0, 110.0)]),
        ("star_map", "L1/attention_qkv/Q", "attention_qkv", 1, "Q",
         [(0, 6.0,  110.0), (1, 6.5, 115.0)]),
        ("star_map", "L0/ffn_expand",      "ffn_expand",    0, "",
         [(0, 20.0, 400.0), (1, 21.0, 410.0)]),
        ("max_p",    "L0/attention_qkv/Q", "attention_qkv", 0, "Q",
         [(0, 7.0,  130.0), (1, 7.5, 135.0)]),
        ("max_p",    "L1/attention_qkv/Q", "attention_qkv", 1, "Q",
         [(0, 8.0,  140.0), (1, 8.5, 145.0)]),
        ("max_p",    "L0/ffn_expand",      "ffn_expand",    0, "",
         [(0, 22.0, 430.0), (1, 23.0, 440.0)]),
    ]
    for setter, layer, gtype, lidx, sub, batches in kernel_specs:
        for b, t, e in batches:
            rows.append(_row("kernel", setter, layer, b, 2, t, e,
                             gemm_type=gtype, layer_idx=lidx, sub_type=sub))

    df = pd.DataFrame(rows, columns=_COLUMNS)
    csv_path = tmp_path / "measurements.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_results_dir(synthetic_measurements_csv: Path) -> Path:
    return synthetic_measurements_csv.parent


# ---- load_measurements ------------------------------------------------------


def test_load_measurements_adds_e_per_inf_uj(synthetic_results_dir: Path):
    df = mod.load_measurements(synthetic_results_dir)
    # n_inner=2 across the fixture
    assert "e_per_inf_uj" in df.columns
    # Star_map model batch 0: 1000 / 2 = 500
    sm0 = df[(df["measurement_type"] == "model")
             & (df["setter"] == "star_map")
             & (df["batch_idx"] == 0)].iloc[0]
    assert sm0["e_per_inf_uj"] == pytest.approx(500.0)


def test_load_measurements_handles_zero_n_inner(tmp_path: Path):
    rows = [_row("model", "star_map", "_", 0, 0, 100.0, 1000.0)]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df.to_csv(tmp_path / "measurements.csv", index=False)
    out = mod.load_measurements(tmp_path)
    assert out.iloc[0]["e_per_inf_uj"] == 0.0


# ---- aggregate_per_setter_kernel -------------------------------------------


def test_aggregate_per_setter_kernel_takes_min_across_batches(
    synthetic_results_dir: Path,
):
    df = mod.load_measurements(synthetic_results_dir)
    kagg = mod.aggregate_per_setter_kernel(df)

    sm_l0 = kagg[(kagg["setter"] == "star_map")
                 & (kagg["layer"] == "L0/attention_qkv/Q")].iloc[0]
    # min over batches: t=5.0, energy=100uJ -> per-inf 50uJ
    assert sm_l0["t_min"] == pytest.approx(5.0)
    assert sm_l0["e_min"] == pytest.approx(50.0)
    assert sm_l0["edp"] == pytest.approx(5.0 * 50.0)
    assert sm_l0["gemm_type"] == "attention_qkv"
    assert sm_l0["layer_idx"] == 0
    assert sm_l0["sub_type"] == "Q"


def test_aggregate_per_setter_kernel_excludes_model_rows(
    synthetic_results_dir: Path,
):
    df = mod.load_measurements(synthetic_results_dir)
    kagg = mod.aggregate_per_setter_kernel(df)
    # Model rows have layer == "_"; aggregation must filter them out.
    assert (kagg["layer"] == "_").sum() == 0


def test_aggregate_per_setter_kernel_keeps_grouping_columns(
    synthetic_results_dir: Path,
):
    df = mod.load_measurements(synthetic_results_dir)
    kagg = mod.aggregate_per_setter_kernel(df)
    for col in ["setter", "layer", "gemm_type", "layer_idx", "sub_type",
                "t_min", "e_min", "edp"]:
        assert col in kagg.columns


# ---- edp_decomposition_by_gemm_type ----------------------------------------


def test_edp_decomposition_sums_across_layers(synthetic_results_dir: Path):
    df = mod.load_measurements(synthetic_results_dir)
    kagg = mod.aggregate_per_setter_kernel(df)
    decomp = mod.edp_decomposition_by_gemm_type(kagg)

    sm_qkv = decomp[(decomp["setter"] == "star_map")
                    & (decomp["gemm_type"] == "attention_qkv")].iloc[0]
    # L0 (5us/50uJ) + L1 (6us/55uJ) = 11us/105uJ
    assert sm_qkv["t_us_sum"] == pytest.approx(5.0 + 6.0)
    assert sm_qkv["e_uj_sum"] == pytest.approx(50.0 + 55.0)
    assert sm_qkv["edp"] == pytest.approx((5.0 + 6.0) * (50.0 + 55.0))


def test_edp_decomposition_avg_pwr_mw_units(synthetic_results_dir: Path):
    df = mod.load_measurements(synthetic_results_dir)
    kagg = mod.aggregate_per_setter_kernel(df)
    decomp = mod.edp_decomposition_by_gemm_type(kagg)
    # avg_pwr_mW = (e_uj/t_us) * 1000 = uJ/us * 1000 = mW
    # star_map ffn_expand: 200uJ / 20us = 10 W = 10000 mW
    smf = decomp[(decomp["setter"] == "star_map")
                 & (decomp["gemm_type"] == "ffn_expand")].iloc[0]
    assert smf["avg_pwr_mW"] == pytest.approx(200.0 / 20.0 * 1000)


# ---- per_setter_summary ----------------------------------------------------


def test_per_setter_summary_min_across_batches(synthetic_results_dir: Path):
    df = mod.load_measurements(synthetic_results_dir)
    summary = mod.per_setter_summary(df)
    # star_map: t_min=100, e_min=500 (batch 0); max_p: 120, 650
    sm = summary[summary["setter"] == "star_map"].iloc[0]
    assert sm["t_us_min"] == pytest.approx(100.0)
    assert sm["e_uj_min"] == pytest.approx(500.0)


def test_per_setter_summary_sorted_by_edp_with_relative(
    synthetic_results_dir: Path,
):
    df = mod.load_measurements(synthetic_results_dir)
    summary = mod.per_setter_summary(df)
    # star_map EDP = 100*500 = 50000; max_p = 120*650 = 78000
    assert list(summary["setter"]) == ["star_map", "max_p"]
    assert summary.iloc[0]["edp_rel"] == pytest.approx(1.0)
    assert summary.iloc[1]["edp_rel"] == pytest.approx(78000.0 / 50000.0)
    # Unit conversions
    assert summary.iloc[0]["t_ms"] == pytest.approx(0.1)
    assert summary.iloc[0]["e_mJ"] == pytest.approx(0.5)


def test_per_setter_summary_excludes_kernel_rows(synthetic_results_dir: Path):
    df = mod.load_measurements(synthetic_results_dir)
    summary = mod.per_setter_summary(df)
    # 2 setters -> 2 rows; if kernel rows leaked in we'd see more.
    assert len(summary) == 2


# ---- end-to-end pipeline through write_markdown_report ---------------------


def test_pipeline_writes_markdown_report(synthetic_results_dir: Path):
    """Run the same orchestration the CLI does and check the report exists.

    write_markdown_report needs a configs frame; build a minimal stub that
    matches the columns Section 3-7 read. The aim is to surface any future
    schema drift, not to assert the report's prose verbatim.
    """
    df = mod.load_measurements(synthetic_results_dir)
    kagg = mod.aggregate_per_setter_kernel(df)
    decomp = mod.edp_decomposition_by_gemm_type(kagg)
    summary = mod.per_setter_summary(df)

    # Minimal configs frame (the report only reads a handful of columns).
    configs = pd.DataFrame([
        {"shape_label": "attention_qkv", "M": 128, "K": 512, "N": 512,
         "setter": "star_map", "P_cores": 16, "SPm": 4, "SPn": 4,
         "TPm": 1, "TPk": 4, "TPn": 1, "TM": 32, "TK": 128, "TN": 128,
         "tpOrder": [2, 0, 1], "npu_dispatched": True},
        {"shape_label": "attention_qkv", "M": 128, "K": 512, "N": 512,
         "setter": "max_p", "P_cores": 32, "SPm": 4, "SPn": 8,
         "TPm": 1, "TPk": 2, "TPn": 1, "TM": 32, "TK": 256, "TN": 64,
         "tpOrder": [2, 0, 1], "npu_dispatched": True},
        {"shape_label": "ffn_expand", "M": 128, "K": 512, "N": 2048,
         "setter": "star_map", "P_cores": 32, "SPm": 2, "SPn": 16,
         "TPm": 1, "TPk": 8, "TPn": 1, "TM": 64, "TK": 64, "TN": 128,
         "tpOrder": [2, 0, 1], "npu_dispatched": True},
        {"shape_label": "ffn_expand", "M": 128, "K": 512, "N": 2048,
         "setter": "max_p", "P_cores": 32, "SPm": 2, "SPn": 16,
         "TPm": 1, "TPk": 8, "TPn": 1, "TM": 64, "TK": 64, "TN": 128,
         "tpOrder": [2, 0, 1], "npu_dispatched": True},
    ])

    out_md = synthetic_results_dir / "setter_configs.md"
    mod.write_markdown_report(out_md, configs, decomp, summary)
    text = out_md.read_text()
    assert "Section 7. Key finding" not in text  # was removed in 41a2d43e
    assert "Key finding (data-derived)" in text
    assert "star_map" in text
    assert "attention_qkv" in text
