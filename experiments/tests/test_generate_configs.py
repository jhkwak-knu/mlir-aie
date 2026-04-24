"""Unit tests for experiments/scripts/generate_configs.py.

Covers:
  * Preflight across all 4 real setters + an unknown-setter guard.
  * Per-(shape, setter) generation on the canonical MLP shapes: all entries
    have a populated `config` block, none have `error`.
  * Error capture: a setter raising mid-search lands in `entry["error"]`
    and the outer loop continues.
  * tc_entry round-trip: the emitted dict is a valid tc.json-compatible
    entry (M, K, N, numCores, levels[0]{SPm, SPn, TPm, TPk, TPn, tpOrder}).
  * CLI smoke: full run against the repo's real config produces the
    expected output file and zero errors.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import generate_configs
from xdna_search.types import DEFAULT_COEFFS, SystemInfo


# ----- Fixtures ------------------------------------------------------------


@pytest.fixture
def sys_info() -> SystemInfo:
    """Minimal XDNA2 SystemInfo (matches xdna2_info.json)."""
    return SystemInfo(
        total_cores=32,
        comp_tiles_per_col=4,
        max_columns=8,
        spm_size_bytes=65536,
        mem_tile_mem_bytes=524288,
    )


@pytest.fixture
def mlp_shapes() -> list:
    """fc1/fc2/fc3 for 784-512-512-16 MLP at batch=32 (derived by Step 3).

    Final N is 16 (padded from MNIST 10 classes) to satisfy the XDNA2
    mmul<4,8,8> 2x2 expansion constraint TN % 16 == 0. The mlp_runner
    slices output[:10] before softmax for class probability.
    """
    return [
        {"layer": "fc1", "M": 32, "K": 784, "N": 512, "calls_per_inference": 1},
        {"layer": "fc2", "M": 32, "K": 512, "N": 512, "calls_per_inference": 1},
        {"layer": "fc3", "M": 32, "K": 512, "N": 16, "calls_per_inference": 1},
    ]


@pytest.fixture
def all_setters() -> list:
    return ["star_map", "max_p", "charm_cdse", "timeloop"]


# ----- Preflight ----------------------------------------------------------


def test_preflight_all_setters_ok(sys_info, all_setters):
    report = generate_configs.run_preflight(sys_info, DEFAULT_COEFFS, all_setters)
    assert report["dummy_shape"] == {"M": 32, "K": 784, "N": 512}
    assert set(report["setters"].keys()) == set(all_setters)
    for setter, info in report["setters"].items():
        assert info["status"] == "ok", (
            f"{setter} preflight failed: {info}"
        )
        assert info["valid_candidates"] > 0
        assert info["factory_key"] == generate_configs.SETTER_FACTORY_KEY[setter]


def test_preflight_unknown_setter(sys_info):
    report = generate_configs.run_preflight(
        sys_info, DEFAULT_COEFFS, ["bogus_setter"]
    )
    info = report["setters"]["bogus_setter"]
    assert info["status"] == "unknown_setter"
    assert "SETTER_FACTORY_KEY" in info["error"]


# ----- Per-(shape, setter) generation -------------------------------------


def test_generate_configurations_canonical(sys_info, mlp_shapes, all_setters):
    out = generate_configs.generate_configurations(
        mlp_shapes, all_setters, sys_info, DEFAULT_COEFFS
    )
    # 3 shapes x 4 setters = 12 entries.
    assert len(out) == 12

    for entry in out:
        assert "error" not in entry, f"entry has error: {entry}"
        assert entry["shape"]["layer"] in ("fc1", "fc2", "fc3")
        assert entry["setter"] in all_setters
        cfg = entry["config"]
        assert cfg["d_inner"] in ("M", "N", "K")
        # Notion-schema summary fields
        for k in ("P", "SP_m", "SP_n", "TP_m", "TP_n", "TP_k"):
            assert isinstance(cfg[k], int) and cfg[k] >= 1, f"{k}={cfg[k]!r}"
        # tc.json-compatible entry (for Step 5 compile_kernels consumption).
        tc = cfg["tc_entry"]
        assert tc["M"] == entry["shape"]["M"]
        assert tc["K"] == entry["shape"]["K"]
        assert tc["N"] == entry["shape"]["N"]
        assert tc["numCores"] == cfg["P"]
        levels = tc["levels"]
        assert isinstance(levels, list) and len(levels) == 1
        lvl = levels[0]
        for k in ("SPm", "SPn", "TPm", "TPk", "TPn", "TM", "TK", "TN", "tpOrder"):
            assert k in lvl, f"missing level key {k} in {lvl}"


def test_generate_configurations_sp_product_equals_p(sys_info, mlp_shapes, all_setters):
    """Sanity: SP_m * SP_n must equal P (num_cores) for every entry."""
    out = generate_configs.generate_configurations(
        mlp_shapes, all_setters, sys_info, DEFAULT_COEFFS
    )
    for entry in out:
        cfg = entry["config"]
        assert cfg["SP_m"] * cfg["SP_n"] == cfg["P"]


def test_max_p_picks_p_max_on_mlp_shapes(sys_info, mlp_shapes):
    """max_p must select P = P_max, and P_max >= sm-exh's P on all shapes."""
    out = generate_configs.generate_configurations(
        mlp_shapes, ["max_p", "star_map"], sys_info, DEFAULT_COEFFS
    )
    by_shape_setter = {(e["shape"]["layer"], e["setter"]): e for e in out}
    for layer in ("fc1", "fc2", "fc3"):
        p_max = by_shape_setter[(layer, "max_p")]["config"]["P"]
        p_star = by_shape_setter[(layer, "star_map")]["config"]["P"]
        assert p_star <= p_max, (
            f"{layer}: star_map P={p_star} should be <= max_p P={p_max}"
        )


# ----- Error capture ------------------------------------------------------


def test_error_does_not_abort_loop(monkeypatch, sys_info, mlp_shapes):
    """A setter failing for fc1 should still leave fc2/fc3 entries populated."""
    real_factory = generate_configs.get_searcher_factory

    def flaky_factory(name):
        if name == "star-map":
            def _f(*_a, **_k):
                raise RuntimeError("synthetic failure")
            return _f
        return real_factory(name)

    monkeypatch.setattr(generate_configs, "get_searcher_factory", flaky_factory)

    out = generate_configs.generate_configurations(
        mlp_shapes, ["star_map", "max_p"], sys_info, DEFAULT_COEFFS
    )
    star_entries = [e for e in out if e["setter"] == "star_map"]
    max_p_entries = [e for e in out if e["setter"] == "max_p"]

    # Every star_map row carries the error.
    assert len(star_entries) == 3
    for e in star_entries:
        assert e["error"].startswith("RuntimeError: synthetic failure")
        assert "config" not in e

    # max_p stays unaffected.
    assert len(max_p_entries) == 3
    for e in max_p_entries:
        assert "error" not in e
        assert "config" in e


def test_unknown_setter_captured_per_entry(sys_info, mlp_shapes):
    out = generate_configs.generate_configurations(
        mlp_shapes, ["bogus"], sys_info, DEFAULT_COEFFS
    )
    assert len(out) == 3
    for e in out:
        assert "config" not in e
        assert e["error"] == "unknown setter 'bogus'"


# ----- CLI smoke ----------------------------------------------------------


def test_cli_end_to_end_on_canonical_config(tmp_path):
    """Exercise the full CLI against the real repo config + real shapes."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "experiments" / "configs" / "mlp_512_512_bs32.json"
    assert cfg_path.is_file(), f"missing canonical config: {cfg_path}"

    # Isolate outputs so we don't touch the checked-in results dir.
    shapes_path = tmp_path / "shapes.json"
    out_path = tmp_path / "configurations.json"

    # Pre-run extract_shapes.py to produce the shapes file in tmp_path.
    extract = repo_root / "experiments" / "scripts" / "extract_shapes.py"
    subprocess.run(
        [sys.executable, str(extract),
         "--config", str(cfg_path), "--output", str(shapes_path)],
        check=True, capture_output=True, text=True,
    )

    script = repo_root / "experiments" / "scripts" / "generate_configs.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--config", str(cfg_path),
         "--shapes", str(shapes_path),
         "--output", str(out_path)],
        check=True, capture_output=True, text=True,
    )

    assert out_path.is_file()
    doc = json.loads(out_path.read_text())
    # 12 entries = 3 shapes x 4 setters.
    assert len(doc["configurations"]) == 12
    errors = [e for e in doc["configurations"] if "error" in e]
    assert not errors, f"errors: {errors}"
    # Preflight recorded for all requested setters.
    assert set(doc["preflight"]["setters"].keys()) == {
        "star_map", "max_p", "charm_cdse", "timeloop",
    }
    for info in doc["preflight"]["setters"].values():
        assert info["status"] == "ok"
    # CLI exit message includes "0 errors".
    assert "0 errors" in result.stdout


def test_cli_dry_run_skips_output(tmp_path):
    """--dry-run prints preflight but must not write configurations.json."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "experiments" / "configs" / "mlp_512_512_bs32.json"
    shapes_path = tmp_path / "shapes.json"
    out_path = tmp_path / "configurations.json"

    extract = repo_root / "experiments" / "scripts" / "extract_shapes.py"
    subprocess.run(
        [sys.executable, str(extract),
         "--config", str(cfg_path), "--output", str(shapes_path)],
        check=True, capture_output=True, text=True,
    )

    script = repo_root / "experiments" / "scripts" / "generate_configs.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--config", str(cfg_path),
         "--shapes", str(shapes_path),
         "--output", str(out_path),
         "--dry-run"],
        check=True, capture_output=True, text=True,
    )

    assert not out_path.exists()
    assert "preflight:" in result.stdout
