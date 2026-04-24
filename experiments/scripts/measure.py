"""Orchestrate mlp_runner --mode measure across every requested setter.

Each setter is executed as an independent subprocess. Between setters the
orchestrator sleeps for `--inter-setter-idle-seconds` so RAPL counters can
drift back to baseline.  The Python side only:

  * validates environment pre-requisites (RAPL readable, setup_env.sh
    already sourced so aie-opt / aiecc.py / xchesscc are on PATH, NPU
    driver timeout set),
  * resolves input paths (config.json, configurations.json,
    kernel_binaries/, mlp_runner binary),
  * invokes the runner per setter and captures its stdout / stderr,
  * collates per-setter exit codes + the aggregated CV stats surfaced
    by the runner, emits `measurement_run.json` next to
    measurements.{csv,json}.

The runner itself writes measurements.csv (append) + measurements.json
(runs[] append) into <config.output.results_dir>, so after this script
finishes those two files contain every setter's data.

Typical usage (after Step 5 full build + Step 6-3..6-4 mlp_runner built):

  source test/onnx-mlir/scripts/setup_env.sh --measure
  python experiments/scripts/measure.py \\
      --config experiments/configs/mlp_512_512_bs32.json \\
      --backend npu
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER_BINARY = REPO_ROOT / "src" / "mlp_runner" / "build" / "mlp_runner"


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
SETUP_ENV_SCRIPT = REPO_ROOT / "test" / "onnx-mlir" / "scripts" / "setup_env.sh"
RAPL_PKG_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
NPU_DRIVER_TIMEOUT_PATH = "/sys/module/amdxdna/parameters/timeout_in_sec"


# ----- Environment preflight ------------------------------------------------


def check_environment(require_npu_driver: bool) -> List[str]:
    """Return a list of human-readable issues found with the current env."""
    issues: List[str] = []

    # RAPL readable (non-zero read).
    if not os.path.isfile(RAPL_PKG_PATH):
        issues.append(f"RAPL sysfs missing: {RAPL_PKG_PATH}")
    else:
        try:
            with open(RAPL_PKG_PATH) as f:
                int(f.read().strip())
        except (OSError, ValueError) as e:
            issues.append(f"RAPL readable failed ({RAPL_PKG_PATH}): {e}")

    # Toolchain on PATH (setup_env.sh was sourced).
    for exe in ("aie-opt", "aiecc.py", "xchesscc"):
        if shutil.which(exe) is None:
            issues.append(
                f"'{exe}' not on PATH; run `source {SETUP_ENV_SCRIPT} --measure` first"
            )
            break

    if require_npu_driver:
        if not os.path.isfile(NPU_DRIVER_TIMEOUT_PATH):
            issues.append(
                "amdxdna NPU driver not loaded "
                f"(missing {NPU_DRIVER_TIMEOUT_PATH})"
            )
        else:
            try:
                with open(NPU_DRIVER_TIMEOUT_PATH) as f:
                    timeout_s = int(f.read().strip())
                if timeout_s < 60:
                    issues.append(
                        f"amdxdna timeout={timeout_s}s < 60; "
                        "source setup_env.sh --measure to bump it"
                    )
            except (OSError, ValueError) as e:
                issues.append(f"reading amdxdna timeout failed: {e}")

    return issues


# ----- Runner invocation ----------------------------------------------------


def run_one_setter(
    runner_binary: Path,
    config_path: Path,
    setter: str,
    backend: str,
    seed: int,
    configurations_path: Optional[Path],
    kernel_binaries_dir: Optional[Path],
    output_dir: Path,
    log_dir: Path,
) -> Dict[str, Any]:
    """Run mlp_runner once for a single setter; capture stdout/stderr."""
    log_path = log_dir / f"measure_{setter}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(runner_binary),
        "--config", str(config_path),
        "--setter", setter,
        "--backend", backend,
        "--mode", "measure",
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    if configurations_path is not None:
        cmd += ["--configurations", str(configurations_path)]
    if kernel_binaries_dir is not None:
        cmd += ["--kernel-binaries-dir", str(kernel_binaries_dir)]

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# cmd: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    elapsed = time.time() - t0

    return {
        "setter": setter,
        "backend": backend,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 3),
        "log": _rel_or_abs(log_path),
    }


# ----- Cross-setter CV extraction ------------------------------------------


def collect_cv_stats(measurements_json: Path) -> List[Dict[str, Any]]:
    """Return a compact per-setter view of CV and minima for reporting."""
    if not measurements_json.is_file():
        return []
    try:
        doc = json.loads(measurements_json.read_text())
    except json.JSONDecodeError:
        return []
    out = []
    for run in doc.get("runs", []):
        out.append({
            "setter": run.get("setter"),
            "backend": run.get("backend"),
            "batch_energy_cv_pct": run.get("batch_energy_cv_pct"),
            "batch_time_cv_pct": run.get("batch_time_cv_pct"),
            "batch_min_model_time_us": run.get("batch_min_model_time_us"),
            "batch_min_energy_per_inference_uj":
                run.get("batch_min_energy_per_inference_uj"),
            "layer_time_us_min_global": run.get("layer_time_us_min_global"),
        })
    return out


# ----- CLI -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate measurement runs across setters."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--runner-binary", type=Path, default=DEFAULT_RUNNER_BINARY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configurations", type=Path, default=None)
    parser.add_argument("--kernel-binaries-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--inter-setter-idle-seconds", type=float, default=30.0,
                        help="sleep between setters so RAPL / thermals settle "
                             "(investigation-grade default 30; set 0 for smoke)")
    parser.add_argument("--setters", nargs="*", default=None,
                        help="override setter list from config")
    parser.add_argument("--skip-env-check", action="store_true",
                        help="bypass RAPL / toolchain / NPU driver checks "
                             "(use only for dry runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print plan and exit without invoking the runner")
    args = parser.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    setters = args.setters or config.get("setters") or []
    if not setters:
        parser.error("setters list is empty (CLI and config both missing it)")

    results_root = None
    results_dir_raw = config.get("output", {}).get("results_dir")
    if results_dir_raw:
        results_root = Path(results_dir_raw).expanduser()
    else:
        results_root = (REPO_ROOT / "experiments" / "results"
                        / args.config.stem)

    output_dir = args.output_dir or results_root
    configurations_path = args.configurations or results_root / "configurations.json"
    kernel_binaries_dir = args.kernel_binaries_dir or results_root / "kernel_binaries"
    log_dir = output_dir / "measure_logs"

    # Preflight
    issues = []
    if not args.skip_env_check:
        issues = check_environment(require_npu_driver=(args.backend == "npu"))
    if not args.runner_binary.is_file():
        issues.append(
            f"runner binary missing: {args.runner_binary}. "
            "Run `experiments/scripts/build_runner.sh` first."
        )
    if not configurations_path.is_file():
        issues.append(f"configurations.json missing: {configurations_path}")

    if issues:
        print("measure: preflight failed:", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        if not args.dry_run:
            return 2

    print(f"measure: runner={args.runner_binary}")
    print(f"measure: setters={setters}")
    print(f"measure: backend={args.backend}, seed={args.seed}")
    print(f"measure: output_dir={output_dir}")
    print(f"measure: inter_setter_idle={args.inter_setter_idle_seconds}s")
    if args.dry_run:
        print("measure: dry-run, nothing to execute")
        return 0

    # Clear stale measurements so re-running produces a clean CSV/JSON.
    measurements_csv = output_dir / "measurements.csv"
    measurements_json = output_dir / "measurements.json"
    for p in (measurements_csv, measurements_json):
        if p.exists():
            p.unlink()

    run_log: List[Dict[str, Any]] = []
    for i, setter in enumerate(setters):
        if i > 0 and args.inter_setter_idle_seconds > 0:
            print(f"measure: idle {args.inter_setter_idle_seconds:.1f}s before {setter}...")
            time.sleep(args.inter_setter_idle_seconds)

        print(f"[{i+1}/{len(setters)}] running setter={setter}")
        entry = run_one_setter(
            args.runner_binary, args.config, setter, args.backend,
            args.seed, configurations_path, kernel_binaries_dir,
            output_dir, log_dir,
        )
        run_log.append(entry)
        status = "ok" if entry["returncode"] == 0 else f"FAIL({entry['returncode']})"
        print(f"  -> {status} in {entry['elapsed_s']:.2f}s; log: {entry['log']}")

    # Assemble final run-level JSON with CV snapshot.
    cv_summary = collect_cv_stats(measurements_json)
    meta = {
        "config": str(args.config.resolve()),
        "backend": args.backend,
        "seed": args.seed,
        "runs": run_log,
        "cv_summary": cv_summary,
    }
    summary_path = output_dir / "measurement_run.json"
    summary_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"measure: wrote {summary_path}")

    # CV gate: fail run if any setter exceeds 15% on either axis.
    bad = [c for c in cv_summary
           if (c.get("batch_energy_cv_pct") or 0) > 15.0
           or (c.get("batch_time_cv_pct") or 0) > 15.0]
    if bad:
        print("measure: CV > 15% on these runs (check measurements.json):",
              file=sys.stderr)
        for b in bad:
            print(f"  - {b['setter']}: "
                  f"energy_cv={b['batch_energy_cv_pct']:.2f}% "
                  f"time_cv={b['batch_time_cv_pct']:.2f}%", file=sys.stderr)
        return 3

    failed = [r for r in run_log if r["returncode"] != 0]
    return 0 if not failed else 4


if __name__ == "__main__":
    sys.exit(main())
