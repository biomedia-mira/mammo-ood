"""Per-dataset mmap creation entry scripts.

Each module in this package builds a (images.npy, metadata.parquet) pair for
one dataset using the shared utilities in `utils/mmap/`.

Run a single dataset with::

    python -m utils.create_mmap.<dataset_name>

Each module exposes a `main()` function and a module-level constant block
describing the source paths, output path, and image size.
"""
