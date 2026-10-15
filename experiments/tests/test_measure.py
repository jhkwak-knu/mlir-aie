"""Unit tests for experiments/scripts/measure.py.

mlp_runner itself is covered by C++ tests; these Python tests focus on:
  * environment preflight (RAPL, toolchain, NPU driver) and failure modes
  * dry-run exits without touching the runner
  * run_one_setter wires flags correctly (verified via a shim binary)
  * collect_cv_stats parses runner output into a reporting view
  * CV > 15% returns exit code 3 (gate enforcement)
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import measure


# ----- Environment preflight ----------------------------------------------


def test_check_environment_reports_missing_toolchain(monkeypatch):
    # Pretend RAPL is fine but no toolchain binary is resolvable.
    monkeypatch.setattr(measure.os.path, "isfile",
                        lambda p: p == measure.RAPL_PKG_PATH)
    monkeypatch.setattr(measure, "open", lambda *a, **k: _StringFile("12345"),
                        raising=False)
    monkeypatch.setattr(measure.shutil, "which", lambda _e: None)
    issues = measure.check_environment(require_npu_driver=False)
    assert any("not on PATH" in i for i in issues)


def test_check_environment_reports_low_npu_timeout(tmp_path, monkeypatch):
    # RAPL + toolchain OK; NPU driver present but timeout below 60s.
    def _isfile(p):
        return p in (measure.RAPL_PKG_PATH, measure.NPU_DRIVER_TIMEOUT_PATH)
    monkeypatch.setattr(measure.os.path, "isfile", _isfile)
    monkeypatch.setattr(measure.shutil, "which", lambda _e: "/fake/bin/" + _e)
    opens = {
        measure.RAPL_PKG_PATH: "12345",
        measure.NPU_DRIVER_TIMEOUT_PATH: "10",
    }
    monkeypatch.setattr(measure, "open",
                        lambda p, *a, **k: _StringFile(opens[p]),
                        raising=False)
    issues = measure.check_environment(require_npu_driver=True)
    assert any("timeout=" in i and "< 60" in i for i in issues)


class _StringFile:
    def __init__(self, s): self._s = s
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._s


# ----- Dry run ------------------------------------------------------------


def _canonical_config(tmp_path, results_dir=None):
    cfg = {
        "model": {
            "type": "mlp", "layer_sizes": [784, 512, 512, 16],
            "activations": ["relu", "relu"], "output_classes": 10,
            "weight_init": "random",
        },
        "dataset": {"name": "mnist", "batch_size": 32,
                     "num_inference_samples": 10},
        "measurement": {
            "warmup_iterations": 2, "outer_batches": 2,
            "inner_target_seconds": 0.1, "rapl_domains": ["package"],
            "bracket_idle_seconds": 0.05, "bracket_idle_count": 2,
        },
        "setters": ["star_map", "max_p"],
        "output": {"results_dir": str(results_dir or tmp_path / "results")},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path, cfg


def test_dry_run_does_not_invoke_runner(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "configurations.json").write_text('{"configurations": []}')
    cfg_path, _ = _canonical_config(tmp_path, results_dir)

    fake_runner = tmp_path / "fake_runner"
    fake_runner.write_text("#!/bin/sh\necho hi\n")
    fake_runner.chmod(0o755)

    monkeypatch.setattr(measure, "check_environment", lambda **_k: [])
    monkeypatch.setattr(
        measure, "run_one_setter",
        lambda *a, **k: pytest.fail("runner must not be invoked in dry-run"))

    rc = measure.main([
        "--config", str(cfg_path), "--dry-run",
        "--runner-binary", str(fake_runner),
        "--skip-env-check",
    ])
    assert rc == 0


# ----- run_one_setter shim --------------------------------------------------


def test_run_one_setter_invokes_runner_with_expected_flags(tmp_path):
    """Shim runner records the argv it was called with; check key flags."""
    argv_log = tmp_path / "argv.log"
    shim = tmp_path / "mlp_runner_shim.sh"
    shim.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > {argv_log}\n'
        'echo "done"\n'
    )
    shim.chmod(0o755)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    entry = measure.run_one_setter(
        runner_binary=shim,
        config_path=Path("/dummy/cfg.json"),
        setter="star_map",
        backend="npu",
        seed=123,
        configurations_path=Path("/dummy/cfg_list.json"),
        kernel_binaries_dir=Path("/dummy/kb"),
        output_dir=out_dir,
        log_dir=out_dir / "logs",
    )
    assert entry["returncode"] == 0
    args = argv_log.read_text().splitlines()
    assert "--config" in args and "/dummy/cfg.json" in args
    assert "--setter" in args and "star_map" in args
    assert "--backend" in args and "npu" in args
    assert "--mode" in args and "measure" in args
    assert "--seed" in args and "123" in args
    assert "--configurations" in args and "/dummy/cfg_list.json" in args
    assert "--kernel-binaries-dir" in args and "/dummy/kb" in args
    assert "--output-dir" in args and str(out_dir) in args
    assert (out_dir / "logs" / "measure_star_map.log").is_file()


# ----- CV summary parsing --------------------------------------------------


def test_collect_cv_stats_reads_runs_array(tmp_path):
    json_path = tmp_path / "measurements.json"
    json_path.write_text(json.dumps({
        "runs": [
            {"setter": "star_map", "backend": "cpu",
             "batch_energy_cv_pct": 3.1, "batch_time_cv_pct": 1.2,
             "batch_min_model_time_us": 1000.0,
             "batch_min_energy_per_inference_uj": 200.0,
             "layer_time_us_min_global": {"fc1": 100.0}},
            {"setter": "max_p", "backend": "cpu",
             "batch_energy_cv_pct": 16.0, "batch_time_cv_pct": 14.9,
             "batch_min_model_time_us": 1200.0,
             "batch_min_energy_per_inference_uj": 240.0,
             "layer_time_us_min_global": {"fc1": 150.0}},
        ],
    }))
    summary = measure.collect_cv_stats(json_path)
    assert [s["setter"] for s in summary] == ["star_map", "max_p"]
    assert summary[0]["batch_energy_cv_pct"] == 3.1
    assert summary[1]["batch_time_cv_pct"] == 14.9


def test_collect_cv_stats_missing_file_returns_empty(tmp_path):
    assert measure.collect_cv_stats(tmp_path / "does_not_exist.json") == []


# ----- CV gate enforcement + end-to-end orchestrator ----------------------


@pytest.fixture
def synthetic_runner(tmp_path):
    """Build a shim mlp_runner that produces deterministic measurements.

    When invoked it writes/append:
      * measurements.csv   (one header + one dummy 'model' row per setter)
      * measurements.json  (appends {setter, batch_*_cv_pct, ...} to runs[])
    """
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)

    shim = tmp_path / "runner_shim.py"
    shim.write_text(f"""#!/usr/bin/env python3
import json, os, pathlib, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
setter = args['--setter']
odir = pathlib.Path(args['--output-dir'])
odir.mkdir(parents=True, exist_ok=True)

# High CV for a specific setter, low otherwise -> exercises the 15% gate.
energy_cv = {{'bad': 20.0}}.get(setter, 5.0)
time_cv = {{'bad': 18.0}}.get(setter, 3.0)

csv_path = odir / 'measurements.csv'
is_new = not csv_path.is_file()
with csv_path.open('a') as f:
    if is_new:
        f.write('measurement_type,setter,backend,layer,batch_idx,n_inner,'
                'time_us_min,time_us_mean,energy_uj_min_package,wall_s,'
                'idle_power_pre_mw,idle_power_post_mw,batch_energy_cv_pct,batch_time_cv_pct\\n')
    f.write(f'model,{{setter}},cpu,_,0,1,100.0,100.0,1000,0.1,1000,1000,{{energy_cv}},{{time_cv}}\\n')

json_path = odir / 'measurements.json'
if json_path.is_file():
    doc = json.loads(json_path.read_text())
else:
    doc = {{'runs': []}}
doc['runs'].append({{
    'setter': setter, 'backend': args['--backend'],
    'batch_energy_cv_pct': energy_cv, 'batch_time_cv_pct': time_cv,
    'batch_min_model_time_us': 100.0,
    'batch_min_energy_per_inference_uj': 1000.0,
    'layer_time_us_min_global': {{'fc1': 50.0}},
}})
json_path.write_text(json.dumps(doc))
""")
    shim.chmod(shim.stat().st_mode | 0o111)
    return shim, out_dir


def test_end_to_end_cv_gate_fails_when_over_threshold(tmp_path, synthetic_runner, monkeypatch):
    shim, out_dir = synthetic_runner
    cfg_path, _ = _canonical_config(tmp_path, out_dir)

    # Create an empty configurations.json so the preflight path check succeeds.
    (out_dir / "configurations.json").write_text('{"configurations": []}')

    monkeypatch.setattr(measure, "check_environment", lambda **_k: [])

    rc = measure.main([
        "--config", str(cfg_path),
        "--runner-binary", str(shim),
        "--setters", "good", "bad",
        "--inter-setter-idle-seconds", "0",
        "--backend", "cpu",
        "--skip-env-check",
    ])
    assert rc == 3  # CV gate failure code
    summary = json.loads((out_dir / "measurement_run.json").read_text())
    assert {s["setter"] for s in summary["cv_summary"]} == {"good", "bad"}


def test_end_to_end_all_within_cv_returns_zero(tmp_path, synthetic_runner, monkeypatch):
    shim, out_dir = synthetic_runner
    cfg_path, _ = _canonical_config(tmp_path, out_dir)
    (out_dir / "configurations.json").write_text('{"configurations": []}')

    monkeypatch.setattr(measure, "check_environment", lambda **_k: [])
    rc = measure.main([
        "--config", str(cfg_path),
        "--runner-binary", str(shim),
        "--setters", "good1", "good2",
        "--inter-setter-idle-seconds", "0",
        "--backend", "cpu",
        "--skip-env-check",
    ])
    assert rc == 0
