from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from author_name_disambiguation._modal_app import APP_NAME, FUNCTION_NAME, app, remote_disambiguate

__all__ = ["APP_NAME", "FUNCTION_NAME", "app", "remote_disambiguate"]
