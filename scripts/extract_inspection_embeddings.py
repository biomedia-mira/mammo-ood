#!/usr/bin/env python
"""Extract native image features for model-inspection UMAPs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mammo_benchmark.inspection.extract_embeddings import main


if __name__ == "__main__":
    main()
