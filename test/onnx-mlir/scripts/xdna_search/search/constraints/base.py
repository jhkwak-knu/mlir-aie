"""Constraint Protocol for the FilterSet.

A Constraint takes a (Candidate, OpCase, SystemInfo) triple and returns
True if the candidate passes. Constraints are composed inside a FilterSet
and applied in order.
"""

from __future__ import annotations

from typing import Protocol

from xdna_search.types import Candidate, OpCase, SystemInfo


class Constraint(Protocol):
    """Single-candidate legality or pruning rule."""

    name: str

    def check(self, c: Candidate, op: OpCase, sys_info: SystemInfo) -> bool: ...
