"""Test configuration for the experiments/ scripts.

Places experiments/scripts on sys.path so that `import extract_shapes`
etc. resolve when pytest is invoked from anywhere in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# experiments/tests/conftest.py -> experiments/scripts
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
