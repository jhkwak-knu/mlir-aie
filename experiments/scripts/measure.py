"""Orchestrate <model>_runner --mode measure across every requested setter.

Each setter is executed as an independent subprocess. Between setters the
orchestrator sleeps for `--inter-setter-idle-seconds` so RAPL counters can
drift back to baseline.  The Python side only:

  * validates environment pre-requisites (RAPL readable, setup_env.sh
    already sourced so aie-opt / aiecc.py / xchesscc are on PATH, NPU
    driver timeout set),
  * resolves input paths (config.json, configurations.json,
    kernel_binaries/, runner binary, GGUF for DistilBERT),
  * picks the runner binary based on `config.model.type`
    (mlp -> mlp_runner, distilbert/bert -> distilbert_runner;
    --runner-binary overrides),
  * invokes the runner per setter and captures its stdout / stderr,
  * collates per-setter exit codes + the aggregated CV stats surfaced
    by the runner, emits `measurement_run.json` next to
    measurements.{csv,json}.

The runner itself writes measurements.csv (append) + measurements.json
(runs[] append) into <config.output.results_dir>, so after this script
finishes those two files contain every setter's data.

Typical usage:

  source test/onnx-mlir/scripts/setup_env.sh --measure
  python experiments/scripts/measure.py \\
      --config experiments/configs/distilbert_L128_bs1.json \\
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
DEFAULT_MLP_RUNNER = REPO_ROOT / "src" / "mlp_runner" / "build" / "mlp_runner"
DEFAULT_DISTILBERT_RUNNER = (
    REPO_ROOT / "src" / "distilbert_runner" / "build" / "distilbert_runner"
)
DEFAULT_DISTILBERT_GGUF = (
    REPO_ROOT / "external" / "bert.cpp" / "models" / "distilbert-sst2-f16.gguf"
)


_BERT_LIKE_TYPES = ("distilbert", "bert")


def _resolve_runner_binary(
    config: Dict[str, Any], override: Optional[Path]
) -> tuple[Path, str]:
    """Pick the runner binary based on `model.type` (override wins).

    Returns (binary_path, model_type). Defaults to mlp when model.type is
    absent so older mlp configs that pre-date the field keep working.
    distilbert_runner accepts both "distilbert" and "bert" model.type
    (same encoder graph), so route both to the same binary.
    """
    model_type = (config.get("model") or {}).get("type", "mlp")
    if override is not None:
        return override, model_type
    if model_type in _BERT_LIKE_TYPES:
        return DEFAULT_DISTILBERT_RUNNER, model_type
    return DEFAULT_MLP_RUNNER, model_type


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
    model_type: str = "mlp",
    gguf_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run <model>_runner once for a single setter; capture stdout/stderr.

    For distilbert we pass --gguf and skip mlp-only flags (--configurations
    / --kernel-binaries-dir) — distilbert_runner derives those paths from
    cfg.results_dir internally.
    """
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
    if model_type in _BERT_LIKE_TYPES:
        if gguf_path is None:
            raise RuntimeError(
                f"{model_type} measurement requires --gguf <path> or "
                "model.gguf in the config"
            )
        cmd += ["--gguf", str(gguf_path)]
    else:
        # mlp_runner-specific flags. distilbert_runner doesn't accept these.
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
    parser.add_argument("--runner-binary", type=Path, default=None,
                        help="override the runner binary; default is "
                             "auto-detected from config.model.type "
                             "(mlp -> mlp_runner, distilbert/bert -> "
                             "distilbert_runner)")
    parser.add_argument("--gguf", type=Path, default=None,
                        help="GGUF weights for DistilBERT/BERT runs; "
                             "resolution order: --gguf > config.model.gguf > "
                             "external/bert.cpp/models/"
                             "distilbert-sst2-f16.gguf (only when model.type "
                             "is distilbert); ignored for MLP runs")
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

    runner_binary, model_type = _resolve_runner_binary(
        config, args.runner_binary
    )
    gguf_path: Optional[Path] = None
    if model_type in _BERT_LIKE_TYPES:
        if args.gguf is not None:
            gguf_path = args.gguf
        else:
            cfg_gguf = (config.get("model") or {}).get("gguf")
            if cfg_gguf:
                gguf_path = Path(cfg_gguf).expanduser()
                if not gguf_path.is_absolute():
                    gguf_path = (REPO_ROOT / gguf_path).resolve()
            elif model_type == "distilbert":
                gguf_path = DEFAULT_DISTILBERT_GGUF

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
    if not runner_binary.is_file():
        issues.append(
            f"runner binary missing: {runner_binary}. "
            "Build the runner first (cmake --build src/<runner>/build)."
        )
    if not configurations_path.is_file():
        issues.append(f"configurations.json missing: {configurations_path}")
    if model_type in _BERT_LIKE_TYPES:
        if gguf_path is None or not gguf_path.is_file():
            issues.append(
                f"{model_type} GGUF missing: {gguf_path}. "
                "Provide --gguf, set model.gguf in the config, or place the "
                "file at the default DistilBERT path."
            )

    if issues:
        print("measure: preflight failed:", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        if not args.dry_run:
            return 2

    print(f"measure: runner={runner_binary} (model_type={model_type})")
    if model_type in _BERT_LIKE_TYPES:
        print(f"measure: gguf={gguf_path}")
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
            runner_binary, args.config, setter, args.backend,
            args.seed, configurations_path, kernel_binaries_dir,
            output_dir, log_dir,
            model_type=model_type,
            gguf_path=gguf_path,
        )
        run_log.append(entry)
        status = "ok" if entry["returncode"] == 0 else f"FAIL({entry['returncode']})"
        print(f"  -> {status} in {entry['elapsed_s']:.2f}s; log: {entry['log']}")

    # Assemble final run-level JSON with CV snapshot.
    cv_summary = collect_cv_stats(measurements_json)
    meta = {
        "config": str(args.config.resolve()),
        "model_type": model_type,
        "runner_binary": str(runner_binary),
        "gguf": str(gguf_path) if gguf_path is not None else None,
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
