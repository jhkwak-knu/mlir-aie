"""Unit tests for experiments/scripts/compile_kernels.py.

The build primitive (gen_mlir + make) is expensive to run in unit tests,
so the production build path is exercised separately via a manual smoke
invocation. These tests cover:
  * Canonical-JSON tc_entry hashing (key-order / spacing invariance).
  * Dedup bucketing across configurations.
  * CLI --dry-run plan output.
  * Toolchain presence check error path.
  * compile_all wiring with a stubbed build function: artifacts appear
    in every consumer directory, dedup works, failures propagate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import compile_kernels


# ----- Hash semantics ------------------------------------------------------


def test_hash_insensitive_to_key_order_and_spacing():
    a = {"M": 32, "K": 512, "N": 16, "levels": [{"SPm": 4, "SPn": 1}]}
    b = {"N": 16, "M": 32, "levels": [{"SPn": 1, "SPm": 4}], "K": 512}
    assert compile_kernels.tc_entry_hash(a) == compile_kernels.tc_entry_hash(b)


def test_hash_differs_on_any_scalar_change():
    a = {"M": 32, "K": 512, "N": 16}
    b = {"M": 32, "K": 512, "N": 10}
    assert compile_kernels.tc_entry_hash(a) != compile_kernels.tc_entry_hash(b)


def test_hash_differs_on_nested_tp_order():
    base = {"M": 32, "K": 512, "N": 16, "levels": [{"tpOrder": [0, 1, 2]}]}
    other = {"M": 32, "K": 512, "N": 16, "levels": [{"tpOrder": [2, 0, 1]}]}
    assert compile_kernels.tc_entry_hash(base) != compile_kernels.tc_entry_hash(other)


def test_hash_ignores_cost_model_metadata():
    """t_total_pred / edp_pred / e_total_pred are informational only; identical
    build-relevant fields with different cost predictions must share a hash."""
    base = {
        "M": 32, "K": 512, "N": 16, "elemType": "bf16",
        "numCores": 4, "doubleBuffer": False, "t_total_pred": 1000.0,
        "levels": [{"SPm": 4, "SPn": 1, "TPm": 1, "TPk": 1, "TPn": 1,
                    "TM": 8, "TK": 512, "TN": 16, "tpOrder": [2, 1, 0]}],
    }
    other = dict(base)
    other["t_total_pred"] = 42.0
    other["edp_rank"] = 3
    other["edp_pred"] = 1e12
    other["e_total_pred"] = 5e8
    assert compile_kernels.tc_entry_hash(base) == compile_kernels.tc_entry_hash(other)


# ----- Toolchain guard -----------------------------------------------------


def test_assert_toolchain_available_raises_when_missing(monkeypatch):
    monkeypatch.setattr(compile_kernels.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="missing toolchain"):
        compile_kernels.assert_toolchain_available()


def test_assert_toolchain_available_passes_when_present(monkeypatch):
    monkeypatch.setattr(compile_kernels.shutil, "which", lambda _name: f"/fake/{_name}")
    # Should not raise.
    compile_kernels.assert_toolchain_available()


# ----- Fixtures & helpers --------------------------------------------------


def _fake_config_entries():
    """Four consumers mapping onto 3 unique tc_entries.

    fc1_star_map, fc1_max_p agree on hash_A; fc2_star_map is hash_B;
    fc3_star_map is hash_C. Tests dedup across (shape, setter) axis.
    """
    hash_A = {
        "M": 32, "K": 784, "N": 512, "elemType": "bf16",
        "numCores": 16, "doubleBuffer": False, "t_total_pred": 1.0,
        "levels": [{"SPm": 1, "SPn": 16, "TPm": 1, "TPk": 2, "TPn": 1,
                    "TM": 32, "TK": 392, "TN": 32, "tpOrder": [0, 2, 1]}],
    }
    hash_B = {
        "M": 32, "K": 512, "N": 512, "elemType": "bf16",
        "numCores": 8, "doubleBuffer": False, "t_total_pred": 1.0,
        "levels": [{"SPm": 1, "SPn": 8, "TPm": 1, "TPk": 2, "TPn": 1,
                    "TM": 32, "TK": 256, "TN": 64, "tpOrder": [2, 1, 0]}],
    }
    hash_C = {
        "M": 32, "K": 512, "N": 16, "elemType": "bf16",
        "numCores": 4, "doubleBuffer": False, "t_total_pred": 1.0,
        "levels": [{"SPm": 4, "SPn": 1, "TPm": 1, "TPk": 1, "TPn": 1,
                    "TM": 8, "TK": 512, "TN": 16, "tpOrder": [2, 1, 0]}],
    }
    return [
        {"shape": {"layer": "fc1", "M": 32, "K": 784, "N": 512},
         "setter": "star_map", "config": {"tc_entry": hash_A, "d_inner": "M"}},
        {"shape": {"layer": "fc1", "M": 32, "K": 784, "N": 512},
         "setter": "max_p", "config": {"tc_entry": hash_A, "d_inner": "M"}},
        {"shape": {"layer": "fc2", "M": 32, "K": 512, "N": 512},
         "setter": "star_map", "config": {"tc_entry": hash_B, "d_inner": "K"}},
        {"shape": {"layer": "fc3", "M": 32, "K": 512, "N": 16},
         "setter": "star_map", "config": {"tc_entry": hash_C, "d_inner": "K"}},
    ]


def _stub_build(tc_entry, log_path, make_jobs=1):
    """Pretend that `make` produced out/build/*.xclbin etc."""
    compile_kernels.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for name in compile_kernels.BINARY_FILENAMES:
        (compile_kernels.BUILD_DIR / name).write_bytes(
            f"stub-{name}-{tc_entry['numCores']}".encode()
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"stub build for {tc_entry['numCores']} cores\n")


# ----- compile_all wiring --------------------------------------------------


def test_compile_all_fans_out_binaries_to_every_consumer(
    tmp_path, monkeypatch
):
    """Two consumers sharing one tc_entry both receive identical binaries."""
    monkeypatch.setattr(compile_kernels, "assert_toolchain_available", lambda: None)
    monkeypatch.setattr(compile_kernels, "build_xclbin", _stub_build)

    kernel_binaries = tmp_path / "kernel_binaries"
    build_logs = tmp_path / "build_logs"
    entries = _fake_config_entries()
    status = compile_kernels.compile_all(
        entries, kernel_binaries, build_logs,
    )

    # 3 unique, all succeed.
    assert status["total_unique"] == 3
    assert status["built"] == 3
    assert all(e["status"] == "ok" for e in status["entries"].values())

    # fc1_star_map and fc1_max_p share content.
    def read_bin(layer_setter):
        return (kernel_binaries / layer_setter / "final.xclbin").read_bytes()

    assert read_bin("fc1_star_map") == read_bin("fc1_max_p")
    # fc1_* differs from fc2_* (different tc_entry -> different stub content).
    assert read_bin("fc1_star_map") != read_bin("fc2_star_map")
    # Each destination has tc.json saved alongside the binaries.
    for layer_setter in ("fc1_star_map", "fc1_max_p", "fc2_star_map", "fc3_star_map"):
        for name in (*compile_kernels.BINARY_FILENAMES, "tc.json"):
            assert (kernel_binaries / layer_setter / name).is_file()


def test_compile_all_captures_build_failures(tmp_path, monkeypatch):
    """A subprocess failure in one build does not stop other builds."""
    monkeypatch.setattr(compile_kernels, "assert_toolchain_available", lambda: None)

    call_idx = {"n": 0}

    def flaky_build(tc_entry, log_path, make_jobs=1):
        call_idx["n"] += 1
        if call_idx["n"] == 2:  # second unique build fails
            raise subprocess.CalledProcessError(1, ["make"])
        _stub_build(tc_entry, log_path, make_jobs=make_jobs)

    monkeypatch.setattr(compile_kernels, "build_xclbin", flaky_build)

    status = compile_kernels.compile_all(
        _fake_config_entries(),
        tmp_path / "kernel_binaries",
        tmp_path / "build_logs",
    )
    statuses = [e["status"] for e in status["entries"].values()]
    assert statuses.count("ok") == 2
    assert statuses.count("build_failed") == 1


def test_compile_all_limit_builds_subset(tmp_path, monkeypatch):
    """--limit N should build exactly N unique tc_entries."""
    monkeypatch.setattr(compile_kernels, "assert_toolchain_available", lambda: None)
    monkeypatch.setattr(compile_kernels, "build_xclbin", _stub_build)

    status = compile_kernels.compile_all(
        _fake_config_entries(),
        tmp_path / "kernel_binaries",
        tmp_path / "build_logs",
        limit=1,
    )
    assert status["built"] == 1
    assert status["skipped_limit"] == 2


def test_compile_all_skips_entries_with_error(tmp_path, monkeypatch):
    """Entries already flagged with 'error' in configurations.json are skipped."""
    monkeypatch.setattr(compile_kernels, "assert_toolchain_available", lambda: None)
    monkeypatch.setattr(compile_kernels, "build_xclbin", _stub_build)

    entries = _fake_config_entries() + [
        {"shape": {"layer": "fc9", "M": 1, "K": 1, "N": 1},
         "setter": "nope", "error": "synthetic"},
    ]
    status = compile_kernels.compile_all(
        entries,
        tmp_path / "kernel_binaries",
        tmp_path / "build_logs",
    )
    assert status["total_unique"] == 3  # pre-error entries unchanged


# ----- CLI dry-run --------------------------------------------------------


def test_cli_dry_run_lists_unique_entries(tmp_path, monkeypatch):
    """--dry-run prints the cache plan but touches no build tooling."""
    entries = _fake_config_entries()
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(
        {"output": {"results_dir": str(tmp_path / "results")}}
    ))
    cfgs_path = tmp_path / "results" / "configurations.json"
    cfgs_path.parent.mkdir(parents=True, exist_ok=True)
    cfgs_path.write_text(json.dumps({"configurations": entries}))

    # Ensure assert_toolchain_available is never called in dry-run.
    def _no_toolchain_check():
        raise RuntimeError("compile_all should not be entered in dry-run")
    monkeypatch.setattr(compile_kernels, "assert_toolchain_available", _no_toolchain_check)

    rc = compile_kernels.main([
        "--config", str(cfg_path),
        "--configurations", str(cfgs_path),
        "--dry-run",
    ])
    assert rc == 0
