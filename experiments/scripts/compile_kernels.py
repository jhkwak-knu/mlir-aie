"""Compile NPU kernels for every (layer, setter) configuration.

For each entry in configurations.json:
  1. Write its tc_entry to test/onnx-mlir/out/tc.json.
  2. Regenerate the matmul ONNX MLIR via gen_onnx_matmul_mlir.sh.
  3. Invoke `make all` in test/onnx-mlir/ to produce final.xclbin + insts.bin.
  4. Copy the binaries into experiments/results/<cfg>/kernel_binaries/<layer>_<setter>/.

Caching: identical tc_entry dicts (canonical-JSON SHA-256) share a single
build. Fan-out happens after the build via copy.

Build failures are captured per-configuration in build_status.json; the
outer loop continues. The CLI exits non-zero iff any build failed.

Prerequisites: the MLIR-AIE toolchain (aie-opt, xchesscc, aiecc.py) must be
on PATH. Source test/onnx-mlir/scripts/setup_env.sh before running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rel_or_abs(path: Path) -> str:
    """Return path relative to REPO_ROOT when possible, else absolute."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
ONNX_DIR = REPO_ROOT / "test" / "onnx-mlir"
OUT_DIR = ONNX_DIR / "out"
BUILD_DIR = OUT_DIR / "build"
TC_JSON_PATH = OUT_DIR / "tc.json"
GEN_MLIR_SCRIPT = ONNX_DIR / "scripts" / "generate" / "gen_onnx_matmul_mlir.sh"

BINARY_FILENAMES = ("final.xclbin", "insts.bin", "kernel.o")


# ----- Toolchain / environment checks --------------------------------------


def assert_toolchain_available() -> None:
    """Fail fast with a clear message if the toolchain is not on PATH."""
    missing = [exe for exe in ("aie-opt", "xchesscc", "aiecc.py")
               if shutil.which(exe) is None]
    if missing:
        raise RuntimeError(
            f"missing toolchain executable(s) on PATH: {missing}. "
            "Run `source test/onnx-mlir/scripts/setup_env.sh` first."
        )


# ----- Canonical hash for dedup --------------------------------------------


# Fields that actually affect xclbin build output. Cost-model predictions
# (t_total_pred, edp_pred, ...) are informational only, so they must not
# participate in the dedup hash or identical tile shapes from different
# cost models would rebuild redundantly.
_BUILD_FIELDS_TOP = ("M", "K", "N", "elemType", "numCores", "doubleBuffer")
_BUILD_FIELDS_LEVEL = (
    "SPm", "SPn", "TPm", "TPk", "TPn", "TM", "TK", "TN", "tpOrder",
)


def _canonical_build_payload(tc_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Project tc_entry down to the fields that drive the xclbin build."""
    payload: Dict[str, Any] = {k: tc_entry[k] for k in _BUILD_FIELDS_TOP if k in tc_entry}
    levels = tc_entry.get("levels") or []
    payload["levels"] = [
        {k: lvl[k] for k in _BUILD_FIELDS_LEVEL if k in lvl}
        for lvl in levels
    ]
    return payload


def tc_entry_hash(tc_entry: Dict[str, Any]) -> str:
    """SHA-256 of the build-relevant projection of a tc_entry dict."""
    payload = _canonical_build_payload(tc_entry)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ----- Build primitive (single tc_entry) ------------------------------------


def build_xclbin(
    tc_entry: Dict[str, Any],
    log_path: Path,
    make_jobs: int = 1,
) -> None:
    """Run gen_mlir + make to produce final.xclbin / insts.bin / kernel.o.

    Raises subprocess.CalledProcessError on failure; stdout + stderr are
    captured into log_path for diagnostics.
    """
    TC_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TC_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(tc_entry, f, indent=2)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# tc.json (hash prefix {tc_entry_hash(tc_entry)}):\n")
        log.write(json.dumps(tc_entry, indent=2) + "\n\n")
        log.flush()
        # 1) Regenerate the matmul ONNX MLIR for this shape.
        subprocess.run(
            ["bash", str(GEN_MLIR_SCRIPT), str(TC_JSON_PATH)],
            check=True, cwd=ONNX_DIR, stdout=log, stderr=subprocess.STDOUT, env=env,
        )
        # 2) Build final.xclbin (make all). `make clean` first so builds do
        #    not depend on stale artifacts from a previous tc_entry.
        subprocess.run(
            ["make", "-C", str(ONNX_DIR), "clean"],
            check=True, stdout=log, stderr=subprocess.STDOUT, env=env,
        )
        subprocess.run(
            ["make", "-C", str(ONNX_DIR), "all", f"-j{make_jobs}"],
            check=True, stdout=log, stderr=subprocess.STDOUT, env=env,
        )


def _copy_binaries(dest_dir: Path, tc_entry: Dict[str, Any]) -> None:
    """Copy build artifacts + tc.json to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in BINARY_FILENAMES:
        src = BUILD_DIR / name
        if not src.is_file():
            raise FileNotFoundError(f"expected build artifact missing: {src}")
        shutil.copy2(src, dest_dir / name)
    with (dest_dir / "tc.json").open("w", encoding="utf-8") as f:
        json.dump(tc_entry, f, indent=2)


# ----- Per-run orchestrator ------------------------------------------------


def compile_all(
    configurations: List[Dict[str, Any]],
    kernel_binaries_root: Path,
    build_log_dir: Path,
    make_jobs: int = 1,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build / cache binaries for each (layer, setter) configuration.

    Returns a build_status dict with per-entry results plus a cache summary.
    """
    assert_toolchain_available()
    build_log_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect unique tc_entries; record which (layer, setter)
    # consumers map to each hash.
    cache: Dict[str, Dict[str, Any]] = {}
    for entry in configurations:
        if "error" in entry:
            continue
        tc = entry["config"]["tc_entry"]
        h = tc_entry_hash(tc)
        bucket = cache.setdefault(h, {"tc_entry": tc, "consumers": []})
        bucket["consumers"].append((entry["shape"]["layer"], entry["setter"]))

    if limit is not None:
        # Deterministic subset for smoke testing.
        hashes_to_build = list(cache.keys())[:limit]
    else:
        hashes_to_build = list(cache.keys())

    results: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    for i, h in enumerate(hashes_to_build, start=1):
        bucket = cache[h]
        tc = bucket["tc_entry"]
        consumers = bucket["consumers"]
        primary_layer, primary_setter = consumers[0]
        log_path = build_log_dir / f"build_{h}.log"

        entry_status: Dict[str, Any] = {
            "hash": h,
            "consumers": [f"{lyr}_{st}" for lyr, st in consumers],
            "tc_entry_shape": {k: tc[k] for k in ("M", "K", "N")},
            "num_cores": tc["numCores"],
            "log": _rel_or_abs(log_path),
        }
        print(
            f"[{i}/{len(hashes_to_build)}] build {h} "
            f"({len(consumers)} consumer(s); M={tc['M']} K={tc['K']} N={tc['N']} P={tc['numCores']})"
        )
        build_t0 = time.time()
        try:
            build_xclbin(tc, log_path, make_jobs=make_jobs)
            for layer, setter in consumers:
                dest = kernel_binaries_root / f"{layer}_{setter}"
                _copy_binaries(dest, tc)
                entry_status.setdefault("artifacts", {})[f"{layer}_{setter}"] = _rel_or_abs(dest)
            entry_status["status"] = "ok"
        except FileNotFoundError as exc:
            entry_status["status"] = "copy_failed"
            entry_status["error"] = f"{type(exc).__name__}: {exc}"
        except subprocess.CalledProcessError as exc:
            entry_status["status"] = "build_failed"
            entry_status["error"] = f"returncode={exc.returncode} (see {log_path})"
        except Exception as exc:  # noqa: BLE001
            entry_status["status"] = "unexpected"
            entry_status["error"] = f"{type(exc).__name__}: {exc}"
        entry_status["elapsed_s"] = round(time.time() - build_t0, 2)
        results[h] = entry_status

    return {
        "total_unique": len(cache),
        "built": len(hashes_to_build),
        "skipped_limit": len(cache) - len(hashes_to_build),
        "elapsed_s": round(time.time() - t0, 2),
        "entries": results,
    }


# ----- CLI plumbing ---------------------------------------------------------


def load_configurations(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    cfgs = doc.get("configurations")
    if not isinstance(cfgs, list) or not cfgs:
        raise ValueError(f"{path}: 'configurations' array missing or empty")
    return cfgs


def _resolve_output_root(config_path: Path, config: Dict[str, Any]) -> Path:
    results_dir_raw = config.get("output", {}).get("results_dir")
    if results_dir_raw:
        return Path(results_dir_raw).expanduser()
    return Path(__file__).resolve().parents[1] / "results" / config_path.stem


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile NPU kernels for each (layer, setter) configuration.",
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Experiment config JSON (for results_dir lookup).")
    parser.add_argument("--configurations", type=Path, default=None,
                        help="Path to configurations.json. Defaults to "
                             "<results_dir>/configurations.json.")
    parser.add_argument("--kernel-binaries-dir", type=Path, default=None,
                        help="Destination root for per-(layer,setter) binaries. "
                             "Defaults to <results_dir>/kernel_binaries.")
    parser.add_argument("--build-log-dir", type=Path, default=None,
                        help="Where to stash build logs. Defaults to "
                             "<results_dir>/build_logs.")
    parser.add_argument("--status-json", type=Path, default=None,
                        help="Path to write build_status.json. Defaults to "
                             "<results_dir>/build_status.json.")
    parser.add_argument("--make-jobs", type=int, default=1,
                        help="Parallel make jobs per xclbin build (default 1).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Build only the first N unique tc_entries (smoke).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the cache plan without running any build.")
    args = parser.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    results_root = _resolve_output_root(args.config, config)
    configurations_path = args.configurations or results_root / "configurations.json"
    kernel_binaries_dir = args.kernel_binaries_dir or results_root / "kernel_binaries"
    build_log_dir = args.build_log_dir or results_root / "build_logs"
    status_path = args.status_json or results_root / "build_status.json"

    configurations = load_configurations(configurations_path)
    error_entries = [e for e in configurations if "error" in e]
    valid_entries = [e for e in configurations if "error" not in e]

    if not valid_entries:
        print("error: no valid configurations to build", file=sys.stderr)
        return 1

    if args.dry_run:
        from collections import defaultdict
        buckets = defaultdict(list)
        for e in valid_entries:
            buckets[tc_entry_hash(e["config"]["tc_entry"])].append(
                f"{e['shape']['layer']}_{e['setter']}"
            )
        print(f"unique tc_entries: {len(buckets)} / {len(valid_entries)} consumers")
        for i, (h, cons) in enumerate(buckets.items(), start=1):
            print(f"  {i:2d}. {h}  <- {', '.join(cons)}")
        if error_entries:
            print(f"(plus {len(error_entries)} entries with generate-time errors)")
        return 0

    status = compile_all(
        valid_entries, kernel_binaries_dir, build_log_dir,
        make_jobs=args.make_jobs, limit=args.limit,
    )
    status["source_configurations"] = str(configurations_path.resolve())
    status["skipped_with_error"] = len(error_entries)

    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
        f.write("\n")

    n_ok = sum(1 for e in status["entries"].values() if e["status"] == "ok")
    n_fail = len(status["entries"]) - n_ok
    print(
        f"\nbuild summary: {n_ok}/{len(status['entries'])} unique entries ok "
        f"({n_fail} failed), elapsed {status['elapsed_s']:.1f}s; "
        f"wrote {status_path}"
    )
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
