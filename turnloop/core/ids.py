"""Identifier generation.

Indirected through a module-level counter so golden-transcript tests can make
ids deterministic by calling `use_deterministic_ids()`.
"""

from __future__ import annotations

import itertools
from uuid import uuid4

_counter: itertools.count | None = None


def new_id(prefix: str) -> str:
    if _counter is not None:
        return f"{prefix}_{next(_counter):08d}"
    return f"{prefix}_{uuid4().hex[:16]}"


def use_deterministic_ids(start: int = 1) -> None:
    """Make new_id() emit prefix_00000001, prefix_00000002, ... (tests only)."""
    global _counter
    _counter = itertools.count(start)


def use_random_ids() -> None:
    global _counter
    _counter = None
