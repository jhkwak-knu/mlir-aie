"""Unit tests for experiments/scripts/extract_shapes.py.

Covers schema derivation, validation, the default canonical MLP
(784-512-512-10 bs=32), and CLI-level file I/O.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import extract_shapes


# ----- Fixtures --------------------------------------------------------------


@pytest.fixture
def canonical_config() -> dict:
    """Mirror of experiments/configs/mlp_512_512_bs32.json."""
    return {
        "model": {
            "type": "mlp",
            "layer_sizes": [784, 512, 512, 10],
            "activations": ["relu", "relu"],
            "weight_init": "random",
        },
        "dataset": {"name": "mnist", "batch_size": 32, "num_inference_samples": 100},
        "measurement": {
            "warmup_iterations": 10,
            "outer_batches": 5,
            "inner_target_seconds": 0.5,
            "rapl_domains": ["package", "core"],
            "bracket_idle_seconds": 0.3,
            "bracket_idle_count": 5,
        },
        "setters": ["star_map", "max_p", "charm_cdse", "timeloop"],
        "output": {"results_dir": "./experiments/results/mlp_512_512_bs32/"},
    }


# ----- derive_shapes: happy path --------------------------------------------


def test_derive_shapes_canonical_mlp(canonical_config):
    """784-512-512-10 bs=32 → fc1, fc2, fc3 with expected dims."""
    out = extract_shapes.derive_shapes(canonical_config)

    assert out["unique_shapes"] == 3
    assert out["total_calls_per_inference"] == 3

    assert out["shapes"] == [
        {"layer": "fc1", "M": 32, "K": 784, "N": 512, "calls_per_inference": 1},
        {"layer": "fc2", "M": 32, "K": 512, "N": 512, "calls_per_inference": 1},
        {"layer": "fc3", "M": 32, "K": 512, "N": 10, "calls_per_inference": 1},
    ]


def test_derive_shapes_two_layer_mlp():
    """Two-layer MLP has one GEMM and no activations."""
    cfg = {
        "model": {"type": "mlp", "layer_sizes": [10, 5], "activations": []},
        "dataset": {"batch_size": 4},
    }
    out = extract_shapes.derive_shapes(cfg)
    assert out["shapes"] == [
        {"layer": "fc1", "M": 4, "K": 10, "N": 5, "calls_per_inference": 1}
    ]


def test_derive_shapes_five_layer_mlp():
    """Layer names stay in forward order across a deeper stack."""
    cfg = {
        "model": {
            "type": "mlp",
            "layer_sizes": [16, 32, 32, 32, 4],
            "activations": ["relu", "relu", "relu"],
        },
        "dataset": {"batch_size": 8},
    }
    out = extract_shapes.derive_shapes(cfg)
    assert [s["layer"] for s in out["shapes"]] == ["fc1", "fc2", "fc3", "fc4"]
    assert out["unique_shapes"] == 4


# ----- derive_shapes: validation errors -------------------------------------


@pytest.mark.parametrize("bad_type", ["lstm", "cnn", "transformer", ""])
def test_derive_shapes_rejects_non_mlp_type(bad_type, canonical_config):
    canonical_config["model"]["type"] = bad_type
    with pytest.raises(ValueError, match="unsupported model.type"):
        extract_shapes.derive_shapes(canonical_config)


def test_derive_shapes_rejects_short_layer_sizes(canonical_config):
    canonical_config["model"]["layer_sizes"] = [10]
    with pytest.raises(ValueError, match="at least 2 ints"):
        extract_shapes.derive_shapes(canonical_config)


def test_derive_shapes_rejects_non_positive_layer_sizes(canonical_config):
    canonical_config["model"]["layer_sizes"] = [10, 0, 5]
    with pytest.raises(ValueError, match="positive ints"):
        extract_shapes.derive_shapes(canonical_config)


def test_derive_shapes_rejects_activation_length_mismatch(canonical_config):
    canonical_config["model"]["activations"] = ["relu"]  # expected 2
    with pytest.raises(ValueError, match="activations length must be 2"):
        extract_shapes.derive_shapes(canonical_config)


def test_derive_shapes_rejects_non_positive_batch_size(canonical_config):
    canonical_config["dataset"]["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size must be a positive int"):
        extract_shapes.derive_shapes(canonical_config)


def test_derive_shapes_rejects_missing_model(canonical_config):
    del canonical_config["model"]
    with pytest.raises(ValueError, match="must contain 'model' and 'dataset'"):
        extract_shapes.derive_shapes(canonical_config)


# ----- CLI / file I/O -------------------------------------------------------


def test_cli_writes_shapes_to_output(tmp_path, canonical_config):
    """Invoking extract_shapes.py writes shapes.json to --output."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(canonical_config))
    out_path = tmp_path / "shapes.json"

    rc = extract_shapes.main(["--config", str(cfg_path), "--output", str(out_path)])
    assert rc == 0
    assert out_path.is_file()

    out = json.loads(out_path.read_text())
    assert out["unique_shapes"] == 3
    assert [s["layer"] for s in out["shapes"]] == ["fc1", "fc2", "fc3"]


def test_cli_defaults_output_to_results_dir(tmp_path, canonical_config):
    """Without --output, CLI writes to config.output.results_dir/shapes.json."""
    results_dir = tmp_path / "results" / "case_a"
    canonical_config["output"] = {"results_dir": str(results_dir)}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(canonical_config))

    rc = extract_shapes.main(["--config", str(cfg_path)])
    assert rc == 0
    default_out = results_dir / "shapes.json"
    assert default_out.is_file()
    assert json.loads(default_out.read_text())["unique_shapes"] == 3


def test_cli_stdout_mode(tmp_path, canonical_config, capsys):
    """--stdout prints JSON to stdout and does not write a file."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(canonical_config))

    rc = extract_shapes.main(["--config", str(cfg_path), "--stdout"])
    assert rc == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert out["unique_shapes"] == 3
    # No file written implicitly — confirm no shapes.json under config dir.
    assert not (cfg_path.parent / "shapes.json").exists()


def test_cli_script_runnable_from_shebang(tmp_path, canonical_config):
    """The script is invokable via `python extract_shapes.py` (smoke)."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(canonical_config))
    out_path = tmp_path / "shapes.json"

    script = Path(__file__).resolve().parent.parent / "scripts" / "extract_shapes.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--config", str(cfg_path),
         "--output", str(out_path)],
        capture_output=True, text=True, check=True,
    )
    assert out_path.is_file()
    assert "wrote" in result.stdout
