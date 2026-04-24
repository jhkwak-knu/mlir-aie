"""JSON / filesystem I/O for the search/cost library.

Loads op_list.json, xdna2_info.json, calibration.json; writes tc_list.json
atomically; provides metadata stamping for traceability. Moved from
scripts/generate/tiling_common.py without any behavioral change.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from xdna_search.hw_constants import DEFAULT_CALIB_PATH, REPO_ROOT
from xdna_search.types import CalibCoeffs, DEFAULT_COEFFS, OpCase, SystemInfo


def load_op_list(path: Path) -> List[OpCase]:
    """Read op_list.json and return OpCase list from the 'cases' array."""
    if not path.is_file():
        raise FileNotFoundError(f"op_list.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    cases = doc.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Invalid op_list.json: missing 'cases' array")
    return [
        OpCase(
            M=int(c["M"]), K=int(c["K"]), N=int(c["N"]),
            elem_type=str(c.get("elemType", "bf16")),
        )
        for c in cases
    ]


def load_system_info(path: Path) -> SystemInfo:
    """Read xdna2_info.json and return SystemInfo."""
    if not path.is_file():
        raise FileNotFoundError(f"system info not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    sys_obj = doc["system"]
    device = sys_obj.get("device", {})
    spm_levels = sys_obj.get("spm_levels", [])
    if not spm_levels:
        raise ValueError("empty spm_levels")
    return SystemInfo(
        total_cores=int(sys_obj["total_cores"]),
        comp_tiles_per_col=int(device.get("comp_tiles_per_col", 4)),
        max_columns=int(device.get("max_columns", 8)),
        spm_size_bytes=int(spm_levels[0]["spm_size_bytes"]),
        mem_tile_mem_bytes=int(device.get("mem_tile_mem_bytes", 524288)),
    )


def load_calibration(path: Path = DEFAULT_CALIB_PATH) -> CalibCoeffs:
    """Load calibration.json. Returns DEFAULT_COEFFS if file is missing.

    Supports v2 (perf only) and v3 (perf + energy).
    """
    if not path.is_file():
        return DEFAULT_COEFFS
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    # Energy calibration (v3)
    energy_section = doc.get("energy", {})
    energy_model = energy_section.get("model", "")
    energy_params = energy_section.get("params", {})
    energy_calibrated = bool(energy_model)

    return CalibCoeffs(
        eff_macs=float(doc["eff_macs"]),
        bw_eff_bpc=float(doc["bw_eff_bpc"]),
        l_sync_cy=float(doc["l_sync_cy"]),
        l_pe_cy=float(doc.get("l_pe_cy", doc.get("l_core_cy", doc.get("l_sync2_cy", 0.0)))),
        l_startup_cy=float(doc.get("l_startup_cy", 0)),
        calibrated=True,
        perf_alpha=float(doc.get("perf_alpha", 1.0)),
        perf_beta=float(doc.get("perf_beta", 1.0)),
        l_dma_cy=float(doc.get("l_dma_cy", 0.0)),
        l_setup_cy=float(doc.get("l_setup_cy", 0.0)),
        perf_model=str(doc.get("model", "Core-Sync")),
        energy_model=energy_model,
        energy_params=energy_params,
        energy_calibrated=energy_calibrated,
    )


class CommentFilterFile:
    """Wraps a file object to skip lines starting with '#'.

    Use with csv.DictReader to transparently handle metadata comments:
        with open(path) as f:
            reader = csv.DictReader(CommentFilterFile(f))
    """

    def __init__(self, f):
        self._f = f
        self.metadata: List[str] = []

    def __iter__(self):
        for line in self._f:
            if line.startswith("#"):
                self.metadata.append(line.rstrip())
            else:
                yield line

    def __next__(self):
        for line in self._f:
            if line.startswith("#"):
                self.metadata.append(line.rstrip())
                continue
            return line
        raise StopIteration

    def readline(self):
        """Support csv.reader which calls readline()."""
        return next(self, "")


def atomic_write_json(obj: Any, out_path: Path) -> None:
    """Write JSON atomically to prevent partial writes on crash."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".tmp", delete=False,
            dir=str(out_path.parent), encoding="utf-8",
        ) as tmp:
            json.dump(obj, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(text: str, out_path: Path) -> None:
    """Write text atomically to prevent partial writes on crash."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".logtmp", delete=False,
            dir=str(out_path.parent), encoding="utf-8",
        ) as tmp:
            tmp.write(text)
            tmp.flush()
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _git_commit_short() -> str:
    """Get current git commit hash (short), or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_metadata(
    calib_path: Optional[Path] = None,
    coeffs: Optional["CalibCoeffs"] = None,
    searcher_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build metadata dict for tc_list.json and result CSV traceability.

    searcher_name (optional) records which searcher produced the cases, so
    downstream consumers can tell STAR-Map / SM-exh / Max-P / Timeloop apart.
    """
    meta: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit_short(),
    }
    if calib_path:
        meta["calibration_file"] = str(calib_path.name)
    if coeffs:
        meta["perf_model"] = coeffs.__class__.__name__
        meta["perf_version"] = 9
        meta["bw_eff_bpc"] = coeffs.bw_eff_bpc
        meta["energy_model"] = coeffs.energy_model or "none"
    if searcher_name:
        meta["searcher_name"] = searcher_name
    return meta


def write_tc_list(
    tc_cases: List[Dict[str, Any]], out_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write tc_list.json atomically with optional metadata."""
    doc: Dict[str, Any] = {}
    if metadata:
        doc["metadata"] = metadata
    doc["cases"] = tc_cases
    atomic_write_json(doc, out_path)
