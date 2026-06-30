"""Extract image-level foundation embeddings from mmap/parquet datasets."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mammo_benchmark.config import DATASET_CONFIG, MODEL_SPECS, resolve_model_specs
from mammo_benchmark.data.datamodule import IMAGE_NORMALIZATION, SOURCE_MMAP_SIZE, get_mmap_dir, load_metadata_for_split
from mammo_benchmark.data.image_utils import resize_image
from mammo_benchmark.models.backbones import load_backbone_bundle


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "inspection_embeddings"

DATASET_ALIASES = {
    "vindr": "VinDr-Mammo-breast",
    "vindr-mammo": "VinDr-Mammo-breast",
    "embed": "EMBED",
}


def parse_dataset_list(raw: str) -> list[str]:
    datasets = []
    for item in raw.split(","):
        dataset = item.strip()
        if not dataset:
            continue
        datasets.append(DATASET_ALIASES.get(dataset.lower(), dataset))
    return datasets


def path_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value).strip("-")


def resolve_input_size(input_height: int | None, input_width: int | None, default_size: Tuple[int, int]) -> Tuple[int, int]:
    if input_height is None and input_width is None:
        return default_size
    if input_height is None or input_width is None:
        raise ValueError("Both --input_height and --input_width must be provided together")
    return (int(input_height), int(input_width))


def normalize_for_backbone(images: torch.Tensor, model_spec: Dict) -> torch.Tensor:
    pretrain_norm = model_spec.get("pretrain_norm")
    if pretrain_norm is None:
        return images
    mean = torch.tensor(pretrain_norm["mean"], dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(pretrain_norm["std"], dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return (images - mean) / std


class InspectionMammoDataset(Dataset):
    """Image-only mmap dataset for inspection embedding extraction."""

    def __init__(
        self,
        mmap_path: Path,
        metadata: pd.DataFrame,
        target_size: Tuple[int, int],
    ):
        if not mmap_path.exists():
            raise FileNotFoundError(f"images mmap not found: {mmap_path}")
        self.images = np.load(str(mmap_path), mmap_mode="r")
        if self.images.ndim != 4 or self.images.shape[1] != 1:
            raise ValueError(f"Expected mmap shape [N,1,H,W], got {self.images.shape}")

        _, _, src_h, src_w = self.images.shape
        self.target_size = tuple(int(v) for v in target_size)
        self.h, self.w = self.target_size
        self._needs_resize = (src_h, src_w) != self.target_size

        self.metadata = metadata.reset_index(drop=True).sort_values("mmap_idx", kind="stable").reset_index(drop=True)
        self.mmap_indices = self.metadata["mmap_idx"].astype(np.int64).to_numpy()

    def __len__(self) -> int:
        return len(self.mmap_indices)

    def _load_image(self, mmap_idx: int) -> torch.Tensor:
        arr = np.asarray(self.images[mmap_idx, 0], dtype=np.float32)
        if self._needs_resize:
            arr = resize_image(arr, center_fill=False, target_size=self.target_size)
            arr = np.asarray(arr, dtype=np.float32)
        return torch.from_numpy(arr.copy()).unsqueeze(0) / IMAGE_NORMALIZATION

    def __getitem__(self, index: int) -> Dict:
        mmap_idx = int(self.mmap_indices[index])
        image = self._load_image(mmap_idx)
        return {
            "imidx": mmap_idx,
            "images": image.expand(3, self.h, self.w).contiguous(),
        }


@torch.no_grad()
def extract_embeddings(
    *,
    model_key: str,
    dataset_name: str,
    splits: list[str],
    info_root: Path,
    input_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    feature_mode: str,
    output_root: Path,
    model_specs: Dict[str, Dict],
) -> None:
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset={dataset_name}")
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model={model_key}")

    model_spec = model_specs[model_key]
    mmap_dir = get_mmap_dir(info_root, dataset_name)
    h, w = SOURCE_MMAP_SIZE
    bundle = load_backbone_bundle(model_key=model_key, model_spec=model_spec, input_size=input_size)
    backbone = bundle.backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device).eval()

    if feature_mode == "native" and not hasattr(backbone, "forward_native_feature"):
        raise ValueError(f"Model {model_key} does not support native features")

    autocast_context = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    embeddings = []
    metadata_parts = []

    for split in splits:
        metadata = load_metadata_for_split(mmap_dir / "metadata.parquet", split, dataset_name)
        dataset = InspectionMammoDataset(
            mmap_path=mmap_dir / f"images_{h}x{w}.npy",
            metadata=metadata,
            target_size=input_size,
        )
        metadata_parts.append(dataset.metadata.copy())
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        for batch in tqdm(dataloader, desc=f"{model_key}/{dataset_name}/{split}"):
            images = batch["images"].to(device, non_blocking=True).float()
            images = normalize_for_backbone(images, model_spec)

            with autocast_context:
                emb = backbone.forward_native_feature(images)

            embeddings.append(emb.float().cpu())

    output_dir = output_root / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    if not embeddings:
        raise RuntimeError(f"No embeddings extracted for {dataset_name}/{splits}")

    dataset_tag = path_tag(dataset_name)
    metadata_out = pd.concat(metadata_parts, ignore_index=True)
    embeddings_out = torch.cat(embeddings, dim=0).numpy()
    if len(metadata_out) != embeddings_out.shape[0]:
        raise RuntimeError(
            f"Embedding/metadata row mismatch for {dataset_name}: "
            f"{embeddings_out.shape[0]} embeddings vs {len(metadata_out)} metadata rows"
        )

    np.save(output_dir / f"embeddings_{dataset_tag}_{feature_mode}.npy", embeddings_out)
    metadata_out.to_parquet(output_dir / f"metadata_{dataset_tag}.parquet", index=False)
    print(f"[Embeddings] saved {len(metadata_out)} rows to {output_dir} for {dataset_name} splits={splits}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract image-level foundation embeddings from mmap/parquet data.")
    parser.add_argument("--dataset", default=None, help="Single dataset key. Kept for backward compatibility.")
    parser.add_argument("--datasets", default=None, help="Comma-separated DATASET_CONFIG keys or aliases: vindr,embed.")
    parser.add_argument("--models", default="dinov3_vitb", help="Comma-separated MODEL_SPECS keys.")
    parser.add_argument("--splits", default="Train,Eval,Test", help="Comma-separated split names: Train,Eval,Test.")
    parser.add_argument("--info_root", default=str(REPO_ROOT / "data" / "classification_data"))
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--adapted_checkpoint_root", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--feature_mode", choices=("native",), default="native")
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    info_root = Path(args.info_root)
    output_root = Path(args.output_root)
    model_specs = resolve_model_specs(args.checkpoint_root, args.adapted_checkpoint_root)
    raw_datasets = args.datasets if args.datasets is not None else args.dataset
    if raw_datasets is None:
        raise ValueError("Provide --dataset or --datasets")
    splits = parse_dataset_list(args.splits)

    for model_key in parse_dataset_list(args.models):
        if model_key not in model_specs:
            raise ValueError(f"Unknown model={model_key}")
        input_size = resolve_input_size(args.input_height, args.input_width, model_specs[model_key]["default_input_size"])
        for dataset_name in parse_dataset_list(raw_datasets):
            if dataset_name not in DATASET_CONFIG:
                raise ValueError(f"Unknown dataset={dataset_name}")
            extract_embeddings(
                model_key=model_key,
                dataset_name=dataset_name,
                splits=splits,
                info_root=info_root,
                input_size=input_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_mode=args.feature_mode,
                output_root=output_root,
                model_specs=model_specs,
            )


if __name__ == "__main__":
    main()
