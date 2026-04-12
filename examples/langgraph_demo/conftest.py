"""Make the demo's sibling modules importable under ``--import-mode=importlib``.

The repo runs pytest with ``--import-mode=importlib``, which does not add a
test file's directory to ``sys.path``. The demo's ``test_demo.py`` imports
``tools`` / ``orchestrator`` / ``agent`` as top-level modules; this conftest
puts the demo directory on ``sys.path`` so those imports resolve whether
the test file is invoked from the demo directory or the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = str(Path(__file__).parent)
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)
