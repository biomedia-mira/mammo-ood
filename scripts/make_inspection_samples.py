#!/usr/bin/env python
"""Validate fixed inspection sample CSVs.

The original sample construction notebook is kept at
``analysis/resample_inspection.ipynb`` because it depends on raw dataset
metadata. This script provides a lightweight release check for the fixed sample
CSVs used by the UMAP notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mammo_benchmark.inspection.sample_selection import validate_sample_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vindr", default=str(REPO_ROOT / "data/inspection_samples/vindr_inspection_sample.csv"))
    parser.add_argument("--embed", default=str(REPO_ROOT / "data/inspection_samples/embed_inspection_sample.csv"))
    args = parser.parse_args()

    for name, path in {"VinDr": args.vindr, "EMBED": args.embed}.items():
        sample = validate_sample_csv(path)
        print(f"{name}: {len(sample)} unique inspection images in {path}")


if __name__ == "__main__":
    main()
