"""OpSequence — minimal wrapper for a list of OpCase items.

Adds an optional `model_id` label so the same code path can consume both
ad-hoc op lists and model-level configs. Searchers iterate over an
OpSequence the same way they iterate over a plain List[OpCase].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from xdna_search.types import OpCase


@dataclass
class OpSequence:
    ops: List[OpCase] = field(default_factory=list)
    model_id: Optional[str] = None

    def __iter__(self) -> Iterator[OpCase]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __getitem__(self, idx: int) -> OpCase:
        return self.ops[idx]
