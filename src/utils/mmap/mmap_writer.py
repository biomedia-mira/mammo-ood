"""MMapBuilder: turn a list of (image_path, metadata_row) into a paired
(images.npy mmap, metadata.parquet) on disk.

Uses multiprocessing for image preprocessing. Supports resume: if a previous
run wrote some indices and crashed, calling build() again skips the already
written indices (tracked in a sidecar `progress.json`). Use
`force_rebuild=True` to wipe and start from scratch.

The mmap stores float32 of shape (N, 1, H, W) — matching
`pre-training/util/offline_mmap_creator.py` so 16-bit precision from
DICOM-converted PNGs is preserved. The parquet has the canonical schema from
`schema.py`.

Does NOT touch any existing `.pt` cache. Output paths are user-specified and
should live under a fresh directory (e.g. `<dataset>/mmap_<H>x<W>/`).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocessing import preprocess_image
from .schema import rows_to_dataframe, validate_metadata


# Globals set in worker processes by `_init_worker`.
_WORKER_MMAP: Optional[np.ndarray] = None
_WORKER_OUTPUT_SIZE: Optional[Tuple[int, int]] = None
_WORKER_MMAP_PATH: Optional[str] = None
_WORKER_MMAP_SHAPE: Optional[Tuple[int, ...]] = None


def _init_worker(mmap_path: str, mmap_shape: Tuple[int, ...], output_size: Tuple[int, int]) -> None:
    global _WORKER_MMAP, _WORKER_OUTPUT_SIZE, _WORKER_MMAP_PATH, _WORKER_MMAP_SHAPE
    _WORKER_OUTPUT_SIZE = output_size
    _WORKER_MMAP_PATH = mmap_path
    _WORKER_MMAP_SHAPE = mmap_shape
    _WORKER_MMAP = np.lib.format.open_memmap(mmap_path, mode="r+", dtype="float32", shape=mmap_shape)


def _process_one(args: Tuple[int, str]) -> Tuple[int, bool, Optional[str]]:
    """Worker entrypoint. Returns (idx, success, error_message_or_None)."""
    idx, image_path = args
    try:
        arr = preprocess_image(image_path, _WORKER_OUTPUT_SIZE)
        if arr.dtype != np.float32:
            return idx, False, f"unexpected dtype {arr.dtype}"
        if arr.shape != (1, _WORKER_OUTPUT_SIZE[0], _WORKER_OUTPUT_SIZE[1]):
            return idx, False, f"unexpected shape {arr.shape}"
        _WORKER_MMAP[idx] = arr
        return idx, True, None
    except Exception as exc:  # noqa: BLE001 - want to report any failure
        return idx, False, f"{type(exc).__name__}: {exc}"


@dataclass
class BuildResult:
    n_total: int
    n_written: int
    n_skipped: int
    n_failed: int
    failed: List[Tuple[int, str, str]]  # (idx, image_path, error)


class MMapBuilder:
    """Builds an (images.npy mmap, metadata.parquet) pair on disk."""

    def __init__(
        self,
        mmap_path: Path,
        metadata_path: Path,
        img_size: Tuple[int, int],
        progress_path: Optional[Path] = None,
    ) -> None:
        self.mmap_path = Path(mmap_path)
        self.metadata_path = Path(metadata_path)
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.progress_path = Path(progress_path) if progress_path else self.mmap_path.with_suffix(".progress.json")

    # ------------------------------------------------------------------ #
    # Progress / resume
    # ------------------------------------------------------------------ #
    def _load_progress(self) -> set:
        if not self.progress_path.exists():
            return set()
        try:
            with open(self.progress_path) as f:
                data = json.load(f)
            return set(int(i) for i in data.get("done", []))
        except Exception:
            return set()

    def _save_progress(self, done: set) -> None:
        tmp = self.progress_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"done": sorted(int(i) for i in done)}, f)
        os.replace(tmp, self.progress_path)

    def _wipe(self) -> None:
        for p in [self.mmap_path, self.metadata_path, self.progress_path]:
            if p.exists():
                p.unlink()

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(
        self,
        image_paths: List[str],
        metadata_rows: List[Dict[str, Any]],
        num_workers: Optional[int] = None,
        force_rebuild: bool = False,
        flush_every: int = 5000,
        progress_save_every: int = 25000,
    ) -> BuildResult:
        """Build the mmap + metadata.

        `image_paths` and `metadata_rows` must be the same length and aligned
        by position. The mmap_idx in each row is overwritten with its position.

        Resume: if `mmap_path` exists with the correct shape and the progress
        file lists previously done indices, those indices are skipped. Pass
        `force_rebuild=True` to start from scratch.
        """
        if len(image_paths) != len(metadata_rows):
            raise ValueError(
                f"image_paths length {len(image_paths)} != metadata_rows length {len(metadata_rows)}"
            )

        n_total = len(image_paths)
        if n_total == 0:
            raise ValueError("No images to write")

        self.mmap_path.parent.mkdir(parents=True, exist_ok=True)

        if force_rebuild:
            self._wipe()

        shape = (n_total, 1, self.img_size[0], self.img_size[1])

        # Allocate or validate existing mmap
        if self.mmap_path.exists():
            existing = np.load(self.mmap_path, mmap_mode="r")
            if existing.shape != shape or existing.dtype != np.dtype("float32"):
                raise RuntimeError(
                    f"Existing mmap at {self.mmap_path} has shape={existing.shape} "
                    f"dtype={existing.dtype}; expected shape={shape} dtype=float32. "
                    f"Use force_rebuild=True to overwrite."
                )
            del existing
            done = self._load_progress()
        else:
            print(f"Allocating mmap at {self.mmap_path} with shape {shape} (float32)...")
            mmap_main = np.lib.format.open_memmap(self.mmap_path, mode="w+", dtype="float32", shape=shape)
            mmap_main.flush()
            del mmap_main
            done = set()

        todo = [(i, str(image_paths[i])) for i in range(n_total) if i not in done]
        if not todo:
            print(f"All {n_total} indices already processed; nothing to do.")
            self._finalize_metadata(metadata_rows, n_total)
            return BuildResult(n_total=n_total, n_written=0, n_skipped=n_total, n_failed=0, failed=[])

        print(f"Total: {n_total}; already done: {len(done)}; to process: {len(todo)}")

        if num_workers is None:
            try:
                num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                num_workers = mp.cpu_count()

        print(f"Starting {num_workers} workers...")

        failed: List[Tuple[int, str, str]] = []
        n_written = 0
        last_progress_save = len(done)

        # Re-open the main mmap for periodic flush from the parent process.
        mmap_main = np.lib.format.open_memmap(self.mmap_path, mode="r+", dtype="float32", shape=shape)

        try:
            with mp.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(str(self.mmap_path), shape, self.img_size),
            ) as pool:
                for idx, ok, err in tqdm(
                    pool.imap_unordered(_process_one, todo, chunksize=64),
                    total=len(todo),
                    desc=f"Writing {self.mmap_path.name}",
                ):
                    if ok:
                        done.add(idx)
                        n_written += 1
                    else:
                        failed.append((idx, str(image_paths[idx]), err or "unknown"))

                    if n_written and n_written % flush_every == 0:
                        mmap_main.flush()
                    if (len(done) - last_progress_save) >= progress_save_every:
                        self._save_progress(done)
                        last_progress_save = len(done)
        finally:
            mmap_main.flush()
            del mmap_main
            self._save_progress(done)

        if failed:
            print(f"WARNING: {len(failed)} images failed to process.")
            for idx, path, err in failed[:10]:
                print(f"  idx={idx} path={path} error={err}")

        self._finalize_metadata(metadata_rows, n_total)

        return BuildResult(
            n_total=n_total,
            n_written=n_written,
            n_skipped=len(done) - n_written,
            n_failed=len(failed),
            failed=failed,
        )

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def _finalize_metadata(self, metadata_rows: List[Dict[str, Any]], n_total: int) -> None:
        """Write metadata.parquet, overwriting mmap_idx with positional index."""
        rows = [dict(row) for row in metadata_rows]
        for i, row in enumerate(rows):
            row["mmap_idx"] = i

        df = rows_to_dataframe(rows)
        validate_metadata(df, n_total)

        tmp = self.metadata_path.with_suffix(".tmp.parquet")
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp, index=False)
        os.replace(tmp, self.metadata_path)
        print(f"Wrote metadata: {self.metadata_path} ({len(df)} rows)")

    # ------------------------------------------------------------------ #
    # Verify
    # ------------------------------------------------------------------ #
    def verify(self, sample_size: int = 32, rng_seed: int = 0) -> None:
        """Spot-check: load N random rows from the mmap and assert non-empty."""
        if not self.mmap_path.exists():
            raise FileNotFoundError(self.mmap_path)
        arr = np.load(self.mmap_path, mmap_mode="r")
        n = arr.shape[0]
        rng = np.random.RandomState(rng_seed)
        idxs = rng.choice(n, size=min(sample_size, n), replace=False)
        for i in idxs:
            img = arr[int(i)]
            if img.shape != (1, self.img_size[0], self.img_size[1]):
                raise ValueError(f"idx {i} has shape {img.shape}")
            if int(img.sum()) == 0:
                raise ValueError(f"idx {i} is all zeros (preprocessing likely failed)")
        print(f"Verified {len(idxs)} random samples from {self.mmap_path.name}: all non-empty.")
