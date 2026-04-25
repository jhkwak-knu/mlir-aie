"""Unit tests for the DistilBERT branch of experiments/scripts/extract_shapes.py.

Covers static schema derivation only (no HF model load). The optional
`--verify-with-hf` path that runs forward hooks on the real Hugging Face
checkpoint is exercised separately in an integration test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import extract_shapes


# ----- Fixtures --------------------------------------------------------------


@pytest.fixture
def canonical_distilbert_config() -> dict:
    """Mirror of experiments/configs/distilbert_L128_bs1.json."""
    return {
        "model": {
            "type": "distilbert",
            "pretrained_source": "distilbert-base-uncased",
            "num_layers": 6,
            "hidden_dim": 768,
            "ffn_dim": 3072,
            "num_heads": 12,
            "sequence_length": 128,
            "multi_head_strategy": "batched",
        },
        "dataset": {
            "name": "sst2", "batch_size": 1, "num_inference_samples": 100,
        },
        "measurement": {
            "warmup_iterations": 10,
            "outer_batches": 5,
            "inner_target_seconds": 1.0,
            "rapl_domains": ["package", "core"],
            "bracket_idle_seconds": 0.3,
            "bracket_idle_count": 5,
        },
        "setters": ["star_map", "max_p", "charm_cdse", "timeloop"],
        "output": {"results_dir": "./experiments/results/distilbert_L128_bs1/"},
    }


# ----- Shape derivation: happy path -----------------------------------------


def test_derive_shapes_canonical_distilbert(canonical_distilbert_config):
    """L=128, hidden=768, ffn=3072, heads=12, layers=6 -> Notion §1-3 table."""
    out = extract_shapes.derive_shapes(canonical_distilbert_config)

    assert out["unique_shapes"] == 5
    assert out["total_calls_per_inference"] == 48
    # 6 GEMM-type entries exposed (qkv / output share a shape, but stay split
    # so per-type EDP analysis in Step 6 is straightforward).
    assert len(out["shapes"]) == 6

    by_type = {s["gemm_type"]: s for s in out["shapes"]}
    assert set(by_type) == {
        "attention_qkv", "attention_output",
        "attention_score", "attention_context",
        "ffn_expand", "ffn_compress",
    }

    qkv = by_type["attention_qkv"]
    assert (qkv["M"], qkv["K"], qkv["N"]) == (128, 768, 768)
    assert qkv["calls_per_inference"] == 18  # 3 (Q/K/V) * 6 layers
    assert qkv["sub_type"] == ["Q", "K", "V"]
    assert qkv["layers"] == "0-5"

    out_proj = by_type["attention_output"]
    assert (out_proj["M"], out_proj["K"], out_proj["N"]) == (128, 768, 768)
    assert out_proj["calls_per_inference"] == 6

    score = by_type["attention_score"]
    assert (score["M"], score["K"], score["N"]) == (128, 64, 128)
    assert score["calls_per_inference"] == 6
    assert score["head_batched"] is True

    context = by_type["attention_context"]
    assert (context["M"], context["K"], context["N"]) == (128, 128, 64)
    assert context["calls_per_inference"] == 6
    assert context["head_batched"] is True

    ffn_e = by_type["ffn_expand"]
    assert (ffn_e["M"], ffn_e["K"], ffn_e["N"]) == (128, 768, 3072)
    assert ffn_e["calls_per_inference"] == 6

    ffn_c = by_type["ffn_compress"]
    assert (ffn_c["M"], ffn_c["K"], ffn_c["N"]) == (128, 3072, 768)
    assert ffn_c["calls_per_inference"] == 6


def test_derive_shapes_includes_layer_alias(canonical_distilbert_config):
    """Each entry exposes a `layer` field (= gemm_type) so generate_configs.py
    can index by `shape['layer']` without an MLP/DistilBERT branch."""
    out = extract_shapes.derive_shapes(canonical_distilbert_config)
    for s in out["shapes"]:
        assert s["layer"] == s["gemm_type"]


def test_derive_shapes_records_metadata(canonical_distilbert_config):
    out = extract_shapes.derive_shapes(canonical_distilbert_config)
    assert out["model"] == "distilbert-base-uncased"
    assert out["sequence_length"] == 128
    assert out["batch_size"] == 1
    assert out["multi_head_strategy"] == "batched"


def test_derive_shapes_c5_violations_empty_for_canonical(
    canonical_distilbert_config,
):
    """All 5 canonical DistilBERT shapes satisfy XDNA2 C5 (M%8, K%8, N%16)."""
    out = extract_shapes.derive_shapes(canonical_distilbert_config)
    assert out["c5_violations"] == []


def test_derive_shapes_scales_with_layers(canonical_distilbert_config):
    """num_layers=4 cuts every per-layer call count proportionally; unique
    shapes stay fixed."""
    canonical_distilbert_config["model"]["num_layers"] = 4
    out = extract_shapes.derive_shapes(canonical_distilbert_config)
    by_type = {s["gemm_type"]: s for s in out["shapes"]}
    assert by_type["attention_qkv"]["calls_per_inference"] == 12  # 3 * 4
    assert by_type["attention_output"]["calls_per_inference"] == 4
    assert by_type["ffn_expand"]["calls_per_inference"] == 4
    assert out["total_calls_per_inference"] == 32  # 12 + 4 + 4 + 4 + 4 + 4


def test_derive_shapes_scales_with_sequence_length(canonical_distilbert_config):
    canonical_distilbert_config["model"]["sequence_length"] = 64
    out = extract_shapes.derive_shapes(canonical_distilbert_config)
    by_type = {s["gemm_type"]: s for s in out["shapes"]}
    assert by_type["attention_qkv"]["M"] == 64
    assert by_type["attention_score"]["M"] == 64
    assert by_type["attention_score"]["N"] == 64  # = sequence_length


# ----- Validation errors -----------------------------------------------------


def test_distilbert_requires_known_fields(canonical_distilbert_config):
    """Missing num_layers must surface as ValueError, not KeyError."""
    del canonical_distilbert_config["model"]["num_layers"]
    with pytest.raises(ValueError, match="num_layers"):
        extract_shapes.derive_shapes(canonical_distilbert_config)


def test_distilbert_rejects_non_divisible_heads(canonical_distilbert_config):
    """hidden_dim must be divisible by num_heads."""
    canonical_distilbert_config["model"]["num_heads"] = 11
    with pytest.raises(ValueError, match="num_heads"):
        extract_shapes.derive_shapes(canonical_distilbert_config)


def test_distilbert_rejects_zero_layers(canonical_distilbert_config):
    canonical_distilbert_config["model"]["num_layers"] = 0
    with pytest.raises(ValueError, match="num_layers"):
        extract_shapes.derive_shapes(canonical_distilbert_config)


# ----- C5 violation surfacing -----------------------------------------------


def test_derive_shapes_flags_c5_violation_for_odd_seq_length():
    """sequence_length=33 makes attention_score N=33 fail N%16==0; the entry
    should appear in c5_violations rather than abort the call."""
    cfg = {
        "model": {
            "type": "distilbert",
            "pretrained_source": "distilbert-base-uncased",
            "num_layers": 1,
            "hidden_dim": 64,
            "ffn_dim": 128,
            "num_heads": 8,
            "sequence_length": 33,
            "multi_head_strategy": "batched",
        },
        "dataset": {"batch_size": 1},
    }
    out = extract_shapes.derive_shapes(cfg)
    assert out["c5_violations"], "33-long seq must trigger at least one C5 hit"
    flagged_types = {v["gemm_type"] for v in out["c5_violations"]}
    assert "attention_score" in flagged_types or "attention_qkv" in flagged_types


# ----- CLI -------------------------------------------------------------------


def test_cli_writes_distilbert_shapes(tmp_path, canonical_distilbert_config):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(canonical_distilbert_config))
    out_path = tmp_path / "shapes.json"

    rc = extract_shapes.main(
        ["--config", str(cfg_path), "--output", str(out_path)]
    )
    assert rc == 0
    out = json.loads(out_path.read_text())
    assert out["unique_shapes"] == 5
    assert out["total_calls_per_inference"] == 48
