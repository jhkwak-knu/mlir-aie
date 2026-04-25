"""Smoke tests for experiments/scripts/publication_plots.py.

The aim is to confirm the CLI wires up against the real schemas and writes
the expected file set, not to validate every visual detail (which is best
checked by eye against the dissertation chapter).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import publication_plots as pp


# ----- Fixtures ------------------------------------------------------------


@pytest.fixture
def synthetic_measurements() -> pd.DataFrame:
    """Mirror analyze.test_aggregate_per_setter fixture's schema."""
    rows = []
    setters = {
        "star_map":   {"model": 100.0, "fc1": 40.0, "fc2": 30.0, "fc3": 5.0,
                       "model_e": 1000.0, "fc1_e": 400.0, "fc2_e": 300.0,
                       "fc3_e": 50.0},
        "max_p":      {"model": 120.0, "fc1": 50.0, "fc2": 35.0, "fc3": 6.0,
                       "model_e": 1200.0, "fc1_e": 500.0, "fc2_e": 350.0,
                       "fc3_e": 60.0},
        "charm_cdse": {"model": 90.0, "fc1": 35.0, "fc2": 28.0, "fc3": 4.0,
                       "model_e": 900.0, "fc1_e": 380.0, "fc2_e": 290.0,
                       "fc3_e": 40.0},
        "timeloop":   {"model": 110.0, "fc1": 45.0, "fc2": 32.0, "fc3": 5.5,
                       "model_e": 1100.0, "fc1_e": 450.0, "fc2_e": 320.0,
                       "fc3_e": 55.0},
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


@pytest.fixture
def synthetic_configurations(tmp_path: Path) -> Path:
    """Minimal configurations.json mirroring generate_configs.py output."""
    doc = {
        "configurations": [
            {"shape": {"layer": "fc1", "M": 32, "K": 784, "N": 512},
             "setter": "star_map",
             "config": {"P": 16, "SP_m": 1, "SP_n": 16,
                        "TP_m": 1, "TP_k": 2, "TP_n": 1, "d_inner": "M"}},
            {"shape": {"layer": "fc1", "M": 32, "K": 784, "N": 512},
             "setter": "max_p",
             "config": {"P": 32, "SP_m": 4, "SP_n": 8,
                        "TP_m": 1, "TP_k": 1, "TP_n": 1, "d_inner": "K"}},
        ],
    }
    path = tmp_path / "configurations.json"
    path.write_text(json.dumps(doc))
    return path


# ----- Table builders ------------------------------------------------------


def test_table1_main_columns_and_rows(synthetic_measurements):
    import analyze

    per_setter = analyze.aggregate_per_setter(synthetic_measurements)
    dilution = analyze.dilution_factors(per_setter, "max_p")
    gemm_frac = analyze.gemm_time_fraction(per_setter)
    t1 = pp.table1_main(synthetic_measurements, per_setter, dilution, gemm_frac)

    assert list(t1.columns) == [
        "setter", "setter_display", "time_us", "energy_uj", "edp",
        "level1_gain", "level2_gain", "dilution", "gemm_time_fraction",
    ]
    assert len(t1) == 4
    # Setters appear in canonical order (star_map first).
    assert t1.iloc[0]["setter"] == "star_map"
    # Display name is friendly.
    assert t1.iloc[0]["setter_display"] == "STAR-Map"


def test_table2_configs_columns(synthetic_configurations):
    t2 = pp.table2_configs(synthetic_configurations)
    expected = {"layer", "setter", "M", "K", "N",
                "P", "SP_m", "SP_n", "TP_m", "TP_k", "TP_n", "d_inner",
                "setter_display"}
    assert expected.issubset(set(t2.columns))
    assert len(t2) == 2


# ----- CLI end-to-end ------------------------------------------------------


def test_cli_writes_expected_artifacts(tmp_path, synthetic_measurements,
                                        synthetic_configurations):
    csv_path = tmp_path / "measurements.csv"
    synthetic_measurements.to_csv(csv_path, index=False)
    out_dir = tmp_path / "pub"

    rc = pp.main([
        "--measurements-csv", str(csv_path),
        "--configurations-json", str(synthetic_configurations),
        "--output-dir", str(out_dir),
        "--baseline-setter", "max_p",
    ])

    assert rc == 0
    assert (out_dir / "figures" / "fig1_setter_breakdown.pdf").is_file()
    assert (out_dir / "figures" / "fig1_setter_breakdown.png").is_file()
    assert (out_dir / "figures" / "fig2_te_decomposition.pdf").is_file()
    assert (out_dir / "figures" / "fig2_te_decomposition.png").is_file()
    assert (out_dir / "tables" / "table1_main.csv").is_file()
    assert (out_dir / "tables" / "table2_configs.csv").is_file()
