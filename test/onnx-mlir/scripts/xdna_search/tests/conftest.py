"""Test configuration for xdna_search.

Ensures the `scripts/` directory is on sys.path so that `import xdna_search`
resolves when pytest is invoked from anywhere in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/xdna_search/tests/conftest.py -> scripts/
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
