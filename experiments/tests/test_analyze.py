"""Unit tests for experiments/scripts/analyze.py.

The goal is to verify aggregation math and file-IO wiring without depending
on the real NPU CSV schema. We build synthetic DataFrames that mirror the
columns mlp_runner writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import analyze


# ----- Fixtures -----------------------------------------------------------


@pytest.fixture
def synthetic_measurements() -> pd.DataFrame:
    """3 setters x 2 batches x (1 model + 3 layers) rows = 24."""
    rows = []
    setters = {
        "star_map": {"model": 100.0, "fc1": 40.0, "fc2": 30.0, "fc3": 5.0,
                     "model_e": 1000.0, "fc1_e": 400.0, "fc2_e": 300.0,
                     "fc3_e": 50.0},
        "max_p":    {"model": 120.0, "fc1": 50.0, "fc2": 35.0, "fc3": 6.0,
                     "model_e": 1200.0, "fc1_e": 500.0, "fc2_e": 350.0,
                     "fc3_e": 60.0},
        "charm_cdse": {"model": 90.0, "fc1": 35.0, "fc2": 28.0, "fc3": 4.0,
                       "model_e": 900.0, "fc1_e": 380.0, "fc2_e": 290.0,
                       "fc3_e": 40.0},
    }
    for setter, spec in setters.items():
        for batch in range(2):
            rows.append({
                "measurement_type": "model",
                "setter": setter, "backend": "npu", "layer": "_",
                "batch_idx": batch, "n_inner": 3,
                "time_us_min": spec["model"] + batch * 0.1,
                "time_us_mean": spec["model"] + 1.0,
                "energy_uj_min_package": spec["model_e"] + batch * 2,
                "wall_s": 0.1, "idle_power_pre_mw": 1000.0,
                "idle_power_post_mw": 1000.0,
                "batch_energy_cv_pct": 1.0, "batch_time_cv_pct": 0.5,
            })
            for layer in ("fc1", "fc2", "fc3"):
                rows.append({
                    "measurement_type": "kernel",
                    "setter": setter, "backend": "npu", "layer": layer,
                    "batch_idx": batch, "n_inner": 3,
                    "time_us_min": spec[layer] + batch * 0.05,
                    "time_us_mean": spec[layer] + 0.5,
                    "energy_uj_min_package": spec[layer + "_e"] + batch,
                    "wall_s": 0.1, "idle_power_pre_mw": 1000.0,
                    "idle_power_post_mw": 1000.0,
                    "batch_energy_cv_pct": 1.0, "batch_time_cv_pct": 0.5,
                })
    return pd.DataFrame(rows)


# ----- aggregate_per_setter -----------------------------------------------


def test_aggregate_per_setter_min_across_batches(synthetic_measurements):
    out = analyze.aggregate_per_setter(synthetic_measurements)
    star_fc1 = out[(out.setter == "star_map") & (out.measurement_type == "kernel")
                   & (out.layer == "fc1")].iloc[0]
    # batch=0 is lower in our fixture -> min = base value.
    # `energy_uj_min_package` is per-batch TOTAL (n_inner=3 inferences); the
    # per-inference normalization divides by n_inner before taking the min.
    assert star_fc1["time_us_min"] == pytest.approx(40.0)
    assert star_fc1["energy_per_inference_uj_min"] == pytest.approx(400.0 / 3)
    assert star_fc1["edp_min"] == pytest.approx(40.0 * 400.0 / 3)
    assert star_fc1["n_batches"] == 2


def test_aggregate_per_setter_model_rows(synthetic_measurements):
    out = analyze.aggregate_per_setter(synthetic_measurements)
    model = out[out.measurement_type == "model"]
    assert set(model.setter.unique()) == {"star_map", "max_p", "charm_cdse"}
    # layer for model rows stored verbatim as "_" (header null placeholder).
    assert (model["layer"] == "_").all()


# ----- dilution_factors ---------------------------------------------------


def test_dilution_factors_baseline_is_one(synthetic_measurements):
    per_setter = analyze.aggregate_per_setter(synthetic_measurements)
    d = analyze.dilution_factors(per_setter, "star_map")
    star_row = d[d.setter == "star_map"].iloc[0]
    assert star_row["level1_gain"] == pytest.approx(1.0)
    assert star_row["level2_gain"] == pytest.approx(1.0)
    assert star_row["dilution"] == pytest.approx(1.0)


def test_dilution_factors_charm_is_faster(synthetic_measurements):
    per_setter = analyze.aggregate_per_setter(synthetic_measurements)
    d = analyze.dilution_factors(per_setter, "star_map")
    charm = d[d.setter == "charm_cdse"].iloc[0]
    # charm_cdse has lower EDP in the fixture -> both gain ratios > 1.
    assert charm["level1_gain"] > 1.0
    assert charm["level2_gain"] > 1.0
    # Dilution around 1.0 since fixture layers proportional to model time.
    assert 0.5 < charm["dilution"] < 2.0


def test_dilution_factors_max_p_is_slower(synthetic_measurements):
    per_setter = analyze.aggregate_per_setter(synthetic_measurements)
    d = analyze.dilution_factors(per_setter, "star_map")
    max_p = d[d.setter == "max_p"].iloc[0]
    assert max_p["level1_gain"] < 1.0
    assert max_p["level2_gain"] < 1.0


# ----- gemm_time_fraction -------------------------------------------------


def test_gemm_time_fraction_bounded(synthetic_measurements):
    per_setter = analyze.aggregate_per_setter(synthetic_measurements)
    f = analyze.gemm_time_fraction(per_setter)
    # Each setter's layer time sum should be < model time (host overhead present).
    for _, row in f.iterrows():
        assert 0 < row["gemm_time_fraction"] < 1.0


# ----- End-to-end CLI -----------------------------------------------------


def test_cli_end_to_end_smoke(tmp_path, synthetic_measurements):
    csv_path = tmp_path / "measurements.csv"
    synthetic_measurements.to_csv(csv_path, index=False)
    json_path = tmp_path / "measurements.json"
    json_path.write_text(json.dumps({
        "runs": [{
            "setter": "star_map", "backend": "npu",
            "config_path": "/fake/config.json",
        }],
    }))
    out_dir = tmp_path / "out"

    rc = analyze.main([
        "--measurements-csv", str(csv_path),
        "--measurements-json", str(json_path),
        "--output-dir", str(out_dir),
        "--baseline-setter", "star_map",
    ])
    assert rc == 0
    assert (out_dir / "report" / "mlp_experiment_report.md").is_file()
    assert (out_dir / "figures" / "fig1_kernel_edp.png").is_file()
    assert (out_dir / "figures" / "fig2_model_edp.png").is_file()
    assert (out_dir / "figures" / "fig3_level1_vs_level2.png").is_file()
    assert (out_dir / "tables" / "table1_edp_decomp.csv").is_file()
    assert (out_dir / "tables" / "per_setter.csv").is_file()
    assert (out_dir / "tables" / "gemm_time_fraction.csv").is_file()


def test_cli_empty_csv_returns_nonzero(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    csv_path.write_text(
        "measurement_type,setter,backend,layer,batch_idx,n_inner,time_us_min,"
        "time_us_mean,energy_uj_min_package,wall_s,idle_power_pre_mw,"
        "idle_power_post_mw,batch_energy_cv_pct,batch_time_cv_pct\n"
    )
    out_dir = tmp_path / "out"
    rc = analyze.main([
        "--measurements-csv", str(csv_path),
        "--output-dir", str(out_dir),
    ])
    assert rc == 2
