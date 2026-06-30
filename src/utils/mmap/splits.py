"""Deterministic Train/Eval/Test split helpers.

`make_patient_split` produces a stable patient-level split given a fixed seed
and ratios. Used by per-dataset creator scripts when the source data does not
ship with a Train/Eval/Test column.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

DEFAULT_RATIOS: Tuple[float, float, float] = (0.7, 0.1, 0.2)
DEFAULT_SEED: int = 42


def make_patient_split(
    patient_ids: Iterable[str],
    seed: int = DEFAULT_SEED,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
) -> Dict[str, str]:
    """Return {patient_id -> 'Train'/'Eval'/'Test'}, deterministic per seed.

    The patient list is de-duplicated, sorted, and shuffled with the given
    seed. Sizes are computed by `int(round(n * ratio))` for the first two
    splits with the remainder going to Test.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")

    pids = sorted({str(p) for p in patient_ids if p is not None and str(p) != "nan"})
    if not pids:
        raise ValueError("No patient IDs provided")

    rng = np.random.RandomState(seed)
    arr = np.array(pids, dtype=object)
    rng.shuffle(arr)

    n = len(arr)
    n_train = int(round(n * ratios[0]))
    n_eval = int(round(n * ratios[1]))
    if n_train + n_eval > n:
        raise ValueError(f"Computed split sizes exceed total: {n_train}+{n_eval} > {n}")

    out: Dict[str, str] = {}
    for i, pid in enumerate(arr.tolist()):
        if i < n_train:
            out[pid] = "Train"
        elif i < n_train + n_eval:
            out[pid] = "Eval"
        else:
            out[pid] = "Test"
    return out
