"""Image-level mammography data module backed by per-dataset mmap/parquet.

Each sample is exactly one metadata row and one mmap image. Labels are read
from that same metadata row. If a source dataset only provides study-, exam-,
patient-, or breast-level labels, utils/create_mmap broadcasts them into image
rows before this loader sees them.
"""
from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from mammo_benchmark.config import DATASET_CONFIG
from mammo_benchmark.data.image_utils import resize_image
from mammo_benchmark.data.sampler import SamplerFactory


# mmap stores raw float32 from 16-bit PNGs; divide by uint16 max to bring images
# into [0, 1] before any per-backbone pretrain_norm is applied.
IMAGE_NORMALIZATION = float(np.iinfo(np.uint16).max)


class GammaCorrectionTransform:
    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, 'gammacorrection')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        gamma_factor = None if self.gamma is None else float(torch.empty(1).uniform_(self.gamma[0], self.gamma[1]))
        if gamma_factor is not None:
            img = TF.adjust_gamma(img, gamma_factor, gain=1)
        return img


def normalize_label(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    return str(value).strip().lower()


def build_label_lookup(label_mapping: Dict[Any, int]) -> Dict[str, int]:
    return {normalize_label(raw_label): class_index for raw_label, class_index in label_mapping.items()}


# All datasets currently have mmaps built at this resolution; smaller targets
# (e.g. 1008x756) are derived on-the-fly via aspect-preserving resize.
SOURCE_MMAP_SIZE: Tuple[int, int] = (1024, 768)


def get_mmap_dir(info_root: Path, dataset_name: str) -> Path:
    image_dataset = DATASET_CONFIG[dataset_name].get("image_dataset", dataset_name)
    h, w = SOURCE_MMAP_SIZE
    return info_root / image_dataset / f"mmap_{h}x{w}"


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def iter_label_values(value: Any) -> List[Any]:
    if is_missing_value(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def label_row_value(raw_value: Any, lookup: Dict[str, int]) -> Optional[int]:
    for item in iter_label_values(raw_value):
        norm = normalize_label(item)
        if norm in lookup:
            return lookup[norm]
    return None


def collatable_metadata_value(value: Any) -> str:
    """Return a string sentinel for optional metadata fields used only for logging."""
    if is_missing_value(value):
        return ""
    return str(value)


class ImageMammoDataset(Dataset):
    """One sample per image row in metadata.parquet.

    Labels are taken from the same row as the image. If a source dataset only
    provides exam-, study-, patient-, or breast-level labels, those labels are
    already broadcast into metadata by utils/create_mmap and are used as-is.
    """

    def __init__(
        self,
        mmap_path: Path,
        metadata: pd.DataFrame,
        label_mappings: Dict[str, Dict[Any, int]],
        augmentation: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        if not mmap_path.exists():
            raise FileNotFoundError(f"images mmap not found: {mmap_path}")
        self.mmap_path = mmap_path
        self.images = np.load(str(mmap_path), mmap_mode="r")
        if self.images.ndim != 4 or self.images.shape[1] != 1:
            raise ValueError(f"Expected mmap shape [N,1,H,W], got {self.images.shape}")
        _, _, src_h, src_w = self.images.shape
        self.target_size: Tuple[int, int] = target_size if target_size is not None else (src_h, src_w)
        self.h, self.w = self.target_size
        self._needs_resize = (src_h, src_w) != self.target_size

        self.label_lookups = {task: build_label_lookup(mapping) for task, mapping in label_mappings.items()}
        self.augmentation = augmentation

        if augmentation:
            self.photometric_augment = T.Compose([
                T.RandomApply(transforms=[GammaCorrectionTransform(gamma=0.3)], p=0.5),
                T.RandomApply(transforms=[T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
                T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
                T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            ])
            self.geometric_augment = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(transforms=[T.RandomAffine(degrees=(-10, 10), scale=(0.85, 1.15))], p=0.5),
            ])

        meta = metadata.reset_index(drop=True).sort_values("mmap_idx", kind="stable")
        mmap_indices: List[int] = []
        sample_names: List[str] = []
        exam_ids: List[str] = []
        sides: List[str] = []
        views: List[str] = []
        per_task_label: Dict[str, List[int]] = {task: [] for task in self.label_lookups}

        for row in meta.to_dict("records"):
            labels: Dict[str, int] = {}
            ok_for_all_tasks = True
            for task, lookup in self.label_lookups.items():
                cls = label_row_value(row.get(task_to_column(task)), lookup)
                if cls is None:
                    ok_for_all_tasks = False
                    break
                labels[task] = cls
            if not ok_for_all_tasks:
                continue

            fallback_name = len(mmap_indices)
            mmap_indices.append(int(row["mmap_idx"]))
            sample_names.append(collatable_metadata_value(row.get("sample_name", fallback_name)))
            exam_ids.append(collatable_metadata_value(row.get("exam_id", sample_names[-1])))
            sides.append(collatable_metadata_value(row.get("side")))
            views.append(collatable_metadata_value(row.get("view")))
            for task, cls in labels.items():
                per_task_label[task].append(cls)

        self.mmap_indices = np.asarray(mmap_indices, dtype=np.int64)
        self.sample_names = sample_names
        self.exam_ids = exam_ids
        self.sides = sides
        self.views = views
        self.cached_labels = {task: np.asarray(per_task_label[task], dtype=np.int64) for task in self.label_lookups}

    def __len__(self) -> int:
        return len(self.mmap_indices)

    def _load_image(self, mmap_idx: int) -> torch.Tensor:
        arr = np.asarray(self.images[mmap_idx, 0], dtype=np.float32)
        if self._needs_resize:
            arr = resize_image(arr, center_fill=False, target_size=self.target_size)
            arr = np.asarray(arr, dtype=np.float32)
        return torch.from_numpy(arr.copy()).unsqueeze(0) / IMAGE_NORMALIZATION

    def __getitem__(self, index: int) -> Dict[str, Any]:
        mmap_idx = int(self.mmap_indices[index])
        image = self._load_image(mmap_idx)
        if self.augmentation:
            image = self.photometric_augment(image)
            image = self.geometric_augment(image)

        labels = {
            task: torch.tensor(int(self.cached_labels[task][index]), dtype=torch.long)
            for task in self.label_lookups
        }

        return {
            "imidx": mmap_idx,
            "sample_name": self.sample_names[index],
            "exam_id": self.exam_ids[index],
            "images": image.expand(3, self.h, self.w).contiguous(),  # [3,H,W]
            "labels": labels,
            "side": self.sides[index],
            "view": self.views[index],
        }

    def get_labels(self, task: Optional[str] = None) -> np.ndarray:
        if task is None:
            task = next(iter(self.cached_labels))
        return self.cached_labels[task]

    def get_balance_labels(self) -> np.ndarray:
        if len(self.cached_labels) == 1:
            return self.get_labels()
        label_matrix = np.stack([self.cached_labels[task] for task in self.cached_labels], axis=1)
        _, labels = np.unique(label_matrix, axis=0, return_inverse=True)
        return labels


def task_to_column(task: str) -> str:
    """Map a task name from DATASET_CONFIG to the parquet column."""
    return {
        "Bi-Rads": "bi_rads",
        "Composition": "density",
        "CancerStatus": "cancer_status",
    }[task]


def apply_metadata_filters(metadata: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    filters = DATASET_CONFIG[dataset_name].get("metadata_filters", {})
    if not filters:
        return metadata
    out = metadata
    for column, allowed_values in filters.items():
        if column not in out.columns:
            raise ValueError(f"Dataset {dataset_name} requested metadata filter on missing column {column!r}")
        allowed = {str(value) for value in allowed_values}
        before = len(out)
        out = out[out[column].astype(str).isin(allowed)]
        print(f"[Filter][{dataset_name}] {column} in {sorted(allowed)}: {len(out)}/{before} rows")
    return out.reset_index(drop=True)


def load_metadata_for_split(parquet_path: Path, split: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    sub = df[df["split"] == split]
    sub = apply_metadata_filters(sub, dataset_name)
    if len(sub) == 0:
        raise ValueError(f"No rows for split={split!r} in {parquet_path} after filters for dataset={dataset_name}")
    return sub.reset_index(drop=True)


class UniversalMammoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        info_root: str,
        cache_image_size: Tuple[int, int],
        batch_size: int,
        num_workers: int,
        ood_dataset_names: Optional[List[str]] = None,
        ood_test_only_datasets: Optional[List[str]] = None,
        batch_alpha: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.task_name = task_name
        self.info_root = Path(info_root)
        self.output_size = (int(cache_image_size[0]), int(cache_image_size[1]))
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ood_test_only_datasets: set[str] = set(ood_test_only_datasets or [])
        self.batch_alpha = batch_alpha
        self.seed = int(seed)
        self.train_sampler = None

        dataset_cfg = DATASET_CONFIG[dataset_name]
        all_tasks = dataset_cfg["label_mappings"]
        if task_name not in all_tasks:
            raise ValueError(f"Task {task_name} not available for dataset {dataset_name}. Available: {list(all_tasks.keys())}")
        self.label_mappings = {task_name: all_tasks[task_name]}

        self.mmap_dir = get_mmap_dir(self.info_root, dataset_name)
        self.parquet_path = self.mmap_dir / "metadata.parquet"
        self.images_path = self.mmap_dir / f"images_{SOURCE_MMAP_SIZE[0]}x{SOURCE_MMAP_SIZE[1]}.npy"

        self.ood_label_mappings: Dict[str, Dict[str, Dict[Any, int]]] = {}
        self.ood_mmap_dirs: Dict[str, Path] = {}
        for ood_name in ood_dataset_names or []:
            if ood_name == dataset_name or ood_name not in DATASET_CONFIG:
                continue
            ood_tasks = DATASET_CONFIG[ood_name]["label_mappings"]
            compatible = {t: self.label_mappings[t] for t in self.label_mappings if t in ood_tasks}
            if not compatible:
                continue
            ood_dir = get_mmap_dir(self.info_root, ood_name)
            if not (ood_dir / "metadata.parquet").exists():
                print(f"[OOD] skipping {ood_name}: missing parquet at {ood_dir}")
                continue
            self.ood_label_mappings[ood_name] = compatible
            self.ood_mmap_dirs[ood_name] = ood_dir

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.ood_test_datasets: Dict[str, Dataset] = {}
        self.test_dataset_names: List[str] = []

    def prepare_data(self) -> None:
        # Mmap is built offline (utils.create_mmap.*); just verify presence.
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Missing metadata parquet: {self.parquet_path}. Build it with utils.create_mmap.<dataset>.")
        if not self.images_path.exists():
            raise FileNotFoundError(f"Missing image mmap: {self.images_path}.")

    def _make_split_dataset(
        self,
        mmap_dir: Path,
        label_mappings: Dict[str, Dict[Any, int]],
        split: str,
        augmentation: bool,
        dataset_name: str,
    ) -> ImageMammoDataset:
        meta = load_metadata_for_split(mmap_dir / "metadata.parquet", split, dataset_name)
        return ImageMammoDataset(
            mmap_path=mmap_dir / f"images_{SOURCE_MMAP_SIZE[0]}x{SOURCE_MMAP_SIZE[1]}.npy",
            metadata=meta,
            label_mappings=label_mappings,
            augmentation=augmentation,
            target_size=self.output_size,
        )

    def _ood_splits(self, ood_name: str, available: set[str]) -> List[str]:
        # Large datasets (ood_test_only_datasets) → Test split only (fallback Eval).
        # Small datasets → full (Train+Eval+Test) for more reliable evaluation.
        if ood_name in self.ood_test_only_datasets:
            if "Test" in available:
                return ["Test"]
            return ["Eval"] if "Eval" in available else []
        return [s for s in ("Train", "Eval", "Test") if s in available]

    def _log_class_dist(self, dataset, split: str) -> None:
        if dataset is None or len(dataset) == 0:
            return
        task = next(iter(self.label_mappings))
        if isinstance(dataset, ConcatDataset):
            parts_list = [d.get_labels(task) for d in dataset.datasets if hasattr(d, "get_labels")]
            if not parts_list:
                return
            labels = np.concatenate(parts_list)
        elif hasattr(dataset, "get_labels"):
            labels = dataset.get_labels(task)
        else:
            return
        cls_ids, cls_counts = np.unique(labels, return_counts=True)
        total = int(cls_counts.sum())
        parts = " ".join(f"{int(c)}:{int(n)}({100 * n / total:.1f}%)" for c, n in zip(cls_ids, cls_counts))
        print(f"[ClassDist][{split}] task={task} N={total} {parts}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_sampler = None
            full_train = self._make_split_dataset(self.mmap_dir, self.label_mappings, "Train", augmentation=True, dataset_name=self.dataset_name)
            self.train_dataset = full_train
            self.val_dataset = self._make_split_dataset(self.mmap_dir, self.label_mappings, "Eval", augmentation=False, dataset_name=self.dataset_name)

            if self.batch_alpha > 0 and len(self.train_dataset) > 0:
                train_labels = self.train_dataset.get_balance_labels()
                train_class_idx = [np.where(train_labels == t)[0] for t in np.unique(train_labels)]
                train_batches = len(self.train_dataset) // self.batch_size
                self.train_sampler = SamplerFactory().get(
                    train_class_idx,
                    self.batch_size,
                    train_batches,
                    alpha=self.batch_alpha,
                    kind='fixed',
                )

            self._log_class_dist(self.train_dataset, "train")
            self._log_class_dist(self.val_dataset, "val")
            if self.batch_alpha > 0:
                print(f"[ClassDist][train_sampler] batch_alpha={self.batch_alpha} (0=natural, 1=uniform per batch)")

        if stage in (None, "test"):
            self.test_dataset = self._make_split_dataset(self.mmap_dir, self.label_mappings, "Test", augmentation=False, dataset_name=self.dataset_name)

            self.ood_test_datasets = {}
            for ood_name, ood_dir in self.ood_mmap_dirs.items():
                meta = pd.read_parquet(ood_dir / "metadata.parquet")
                meta = apply_metadata_filters(meta, ood_name)
                avail = set(meta["split"].dropna().unique())
                splits = self._ood_splits(ood_name, avail)
                if not splits:
                    continue
                datasets = []
                for split in splits:
                    sub = meta[meta["split"] == split].reset_index(drop=True)
                    if len(sub) == 0:
                        continue
                    ds = ImageMammoDataset(
                        mmap_path=ood_dir / f"images_{SOURCE_MMAP_SIZE[0]}x{SOURCE_MMAP_SIZE[1]}.npy",
                        metadata=sub,
                        label_mappings=self.ood_label_mappings[ood_name],
                        augmentation=False,
                        target_size=self.output_size,
                    )
                    if len(ds) > 0:
                        datasets.append(ds)
                if not datasets:
                    continue
                self.ood_test_datasets[ood_name] = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

            self.test_dataset_names = []
            if self.test_dataset is not None and len(self.test_dataset) > 0:
                self.test_dataset_names.append(self.dataset_name)
            self.test_dataset_names.extend(self.ood_test_datasets.keys())

            self._log_class_dist(self.test_dataset, f"test/{self.dataset_name}")
            for ood_name, ood_ds in self.ood_test_datasets.items():
                self._log_class_dist(ood_ds, f"test/{ood_name}")

    def _worker_kwargs(self) -> dict:
        if self.num_workers > 0:
            return {"persistent_workers": True, "prefetch_factor": 4}
        return {}

    def train_dataloader(self) -> DataLoader:
        if self.train_sampler is not None:
            return DataLoader(self.train_dataset, batch_sampler=self.train_sampler, num_workers=self.num_workers, pin_memory=True, **self._worker_kwargs())
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            **self._worker_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False, **self._worker_kwargs())

    def test_dataloader(self):
        dataloaders = []
        names = []
        if self.test_dataset is not None and len(self.test_dataset) > 0:
            dataloaders.append(DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False))
            names.append(self.dataset_name)
        for ood_name, ds in self.ood_test_datasets.items():
            dataloaders.append(DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False))
            names.append(ood_name)
        if not dataloaders:
            raise RuntimeError("No test dataset available")
        self.test_dataset_names = names
        return dataloaders if len(dataloaders) > 1 else dataloaders[0]

    def get_test_dataset_names(self) -> List[str]:
        return list(self.test_dataset_names)
