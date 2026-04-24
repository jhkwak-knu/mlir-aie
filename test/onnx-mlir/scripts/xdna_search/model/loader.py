"""Loader for op_list.json producing an OpSequence.

Accepts both the legacy schema (just {cases: [...]}) and the extended form
with an optional top-level `model_id` field. Existing consumers that call
io.load_op_list() still receive a plain List[OpCase].
"""

from __future__ import annotations

import json
from pathlib import Path

from xdna_search.io import load_op_list
from xdna_search.model.op_sequence import OpSequence


def load_model_config(path: Path) -> OpSequence:
    """Read op_list.json and return an OpSequence.

    If the document contains a top-level string `model_id`, it is preserved
    on the returned OpSequence; otherwise model_id is None.
    """
    ops = load_op_list(path)
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    model_id = doc.get("model_id") if isinstance(doc, dict) else None
    return OpSequence(ops=ops, model_id=model_id if isinstance(model_id, str) else None)
