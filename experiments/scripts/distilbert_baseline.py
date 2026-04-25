"""Baseline accuracy + numeric parity check for DistilBERT (Step 4-2-C).

Two questions this script answers:

1. **Reference accuracy** — what does Hugging Face's
   `DistilBertForSequenceClassification` ("distilbert-base-uncased-
   finetuned-sst-2-english") score on the first 100 samples of the
   SST-2 validation split? That number is the target the C++
   distilbert_runner has to land within ±2 %p in Step 4-3.

2. **Numeric parity** — does our vendored `bert.cpp` produce the same
   embedding as PyTorch's `DistilBertModel`? We compare the [CLS]
   token's hidden state on a handful of probe sentences via cosine
   similarity. >= 0.99 means the GGUF conversion + bert.cpp forward
   are numerically faithful (we do not chase strict equality because
   bert.cpp uses f16 weights and PyTorch runs in f32).

Both targets are written to a JSON report under
`experiments/results/distilbert_L128_bs1/baseline.json` so Step 4-3
can read them.

Run:
    python experiments/scripts/distilbert_baseline.py \\
        --reference-repo distilbert-base-uncased-finetuned-sst-2-english \\
        --probe-repo distilbert-base-uncased \\
        --num-samples 100
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BERT_CPP_DIR = REPO_ROOT / "external" / "bert.cpp"
BERT_CPP_BIN = BERT_CPP_DIR / "build" / "bin" / "main"
BERT_CPP_MODELS = BERT_CPP_DIR / "models"
DEFAULT_PROBE_GGUF = BERT_CPP_MODELS / "distilbert-base-uncased-f16.gguf"
PROBE_SENTENCES = [
    "Hello world",
    "The movie was absolutely brilliant.",
    "I did not enjoy this film at all.",
]


# ----- Numeric parity (PyTorch vs bert.cpp on probe sentences) -------------


def _bert_cpp_cls_embedding(gguf_path: Path, prompt: str) -> List[float]:
    """Run `bin/main -r -c -m <gguf> -p <prompt>` and parse the embedding.

    The flag `-r` requests the raw (non-normalized) embedding, matching
    what PyTorch's last_hidden_state[0,0] returns.
    """
    if not BERT_CPP_BIN.is_file():
        raise FileNotFoundError(
            f"bert.cpp main binary not built: {BERT_CPP_BIN}. "
            "Run `cmake -B build . && make -C build -j` from external/bert.cpp first."
        )
    cmd = [str(BERT_CPP_BIN), "-r", "-c", "-m", str(gguf_path), "-p", prompt]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout
    # main writes the embedding as `[ f1, f2, ..., fN ]` with newlines; pull
    # the first balanced bracket pair out and split.
    m = re.search(r"\[\s*(?:[-\d.eE,\s]+)\]", out)
    if not m:
        raise RuntimeError(
            f"could not locate embedding in bert.cpp main stdout. raw stdout:\n{out}"
        )
    body = m.group(0).strip("[]")
    vals = [float(x) for x in body.split(",") if x.strip()]
    return vals


def _torch_distilbert_mean_embedding(repo_id: str, prompt: str) -> List[float]:
    """Run HuggingFace DistilBertModel and return the mean over the
    non-pad tokens of last_hidden_state.

    bert.cpp's `bin/main` emits a mean-pooled embedding (src/bert.cpp
    line 918 builds `sum` with weights 1/cur_len for valid tokens and
    0 for pads, then dots it with the transposed last hidden state).
    To compare like for like we apply the same mean here.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    mdl = AutoModel.from_pretrained(repo_id)
    mdl.eval()
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
    mask = enc["attention_mask"][0].to(torch.float32)
    # last_hidden_state is [1, L, E]; collapse L by mean over valid tokens.
    h = out.last_hidden_state[0]                  # [L, E]
    weighted = (h * mask.unsqueeze(-1)).sum(dim=0) / mask.sum()
    return weighted.tolist()


def _cosine(a: List[float], b: List[float]) -> float:
    import numpy as np
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    if av.shape != bv.shape:
        raise ValueError(f"shape mismatch: {av.shape} vs {bv.shape}")
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def parity_check(probe_repo: str, gguf_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sent in PROBE_SENTENCES:
        torch_vec = _torch_distilbert_mean_embedding(probe_repo, sent)
        bert_vec = _bert_cpp_cls_embedding(gguf_path, sent)
        cos = _cosine(torch_vec, bert_vec)
        rows.append({
            "sentence": sent,
            "torch_dim": len(torch_vec),
            "bertcpp_dim": len(bert_vec),
            "cosine_similarity": cos,
        })
        print(f"  parity '{sent[:40]:40s}'  cos={cos:.4f}  "
              f"(torch={len(torch_vec)}, bertcpp={len(bert_vec)})")
    return rows


# ----- Reference SST-2 accuracy (PyTorch) -----------------------------------


def reference_accuracy(reference_repo: str, num_samples: int) -> Dict[str, Any]:
    """Score the HF reference classifier on `num_samples` SST-2 validation rows."""
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
    )

    ds = load_dataset("glue", "sst2", split=f"validation[:{num_samples}]")
    tok = AutoTokenizer.from_pretrained(reference_repo)
    mdl = AutoModelForSequenceClassification.from_pretrained(reference_repo)
    mdl.eval()

    correct = 0
    n = 0
    pred_labels: List[int] = []
    gold_labels: List[int] = []
    for row in ds:
        enc = tok(row["sentence"], return_tensors="pt", truncation=True,
                  max_length=128, padding="max_length")
        with torch.no_grad():
            logits = mdl(**enc).logits
        pred = int(logits.argmax(dim=-1).item())
        pred_labels.append(pred)
        gold = int(row["label"])
        gold_labels.append(gold)
        if pred == gold:
            correct += 1
        n += 1

    return {
        "reference_repo": reference_repo,
        "num_samples": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "first_10_pred": pred_labels[:10],
        "first_10_gold": gold_labels[:10],
    }


# ----- CLI -----------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reference-repo",
                   default="distilbert-base-uncased-finetuned-sst-2-english")
    p.add_argument("--probe-repo", default="distilbert-base-uncased")
    p.add_argument("--probe-gguf", type=Path, default=DEFAULT_PROBE_GGUF)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path; defaults to "
                        "experiments/results/distilbert_L128_bs1/baseline.json")
    p.add_argument("--skip-parity", action="store_true",
                   help="skip the bert.cpp/PyTorch cosine-similarity check")
    p.add_argument("--skip-reference", action="store_true",
                   help="skip the SST-2 reference accuracy run")
    args = p.parse_args(argv)

    out_path = args.output or (
        REPO_ROOT / "experiments" / "results" / "distilbert_L128_bs1"
        / "baseline.json"
    )

    report: Dict[str, Any] = {
        "reference_repo": args.reference_repo,
        "probe_repo": args.probe_repo,
        "probe_gguf": str(args.probe_gguf),
        "num_samples": args.num_samples,
    }

    if not args.skip_parity:
        print(f"== numeric parity (PyTorch <-> bert.cpp, "
              f"probe={args.probe_repo}) ==")
        report["parity"] = parity_check(args.probe_repo, args.probe_gguf)
    if not args.skip_reference:
        print(f"== reference accuracy "
              f"(SST-2 validation[:{args.num_samples}], "
              f"{args.reference_repo}) ==")
        report["reference"] = reference_accuracy(
            args.reference_repo, args.num_samples,
        )
        print(f"  acc = {report['reference']['accuracy']:.4f} "
              f"({report['reference']['correct']}/{report['reference']['num_samples']})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
