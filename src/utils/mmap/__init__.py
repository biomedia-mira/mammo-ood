"""Shared utilities for building mmap-backed mammography datasets.

Public API:
  - preprocess_image: float32 resize + breast-mask, matches EMBED logic
  - MMapBuilder: writes (images.npy, metadata.parquet) with multiprocessing
  - METADATA_COLUMNS, ALLOWED_*: schema definitions
  - empty_metadata_dataframe, rows_to_dataframe, validate_metadata: schema helpers
"""

from .mmap_writer import BuildResult, MMapBuilder
from .preprocessing import preprocess_image
from .splits import DEFAULT_RATIOS, DEFAULT_SEED, make_patient_split
from .schema import (
    ALLOWED_BIRADS,
    ALLOWED_CANCER,
    ALLOWED_DENSITY,
    ALLOWED_SIDES,
    ALLOWED_SPLITS,
    ALLOWED_VIEWS,
    METADATA_COLUMNS,
    empty_metadata_dataframe,
    rows_to_dataframe,
    validate_metadata,
)

__all__ = [
    "BuildResult",
    "MMapBuilder",
    "preprocess_image",
    "make_patient_split",
    "DEFAULT_RATIOS",
    "DEFAULT_SEED",
    "METADATA_COLUMNS",
    "ALLOWED_BIRADS",
    "ALLOWED_CANCER",
    "ALLOWED_DENSITY",
    "ALLOWED_SIDES",
    "ALLOWED_SPLITS",
    "ALLOWED_VIEWS",
    "empty_metadata_dataframe",
    "rows_to_dataframe",
    "validate_metadata",
]
