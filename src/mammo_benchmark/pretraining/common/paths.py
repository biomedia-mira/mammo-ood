from __future__ import annotations

import os
from pathlib import Path

PRETRAIN_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PRETRAIN_DIR.parents[2]
OUTPUT_SSL_DIR = PRETRAIN_DIR / "output_ssl"
DATA_DIR = REPO_DIR / "data"
EMBED_CSV_PATH = DATA_DIR / "embed.csv"

EMBED_PNG_1024X768_DIR = Path(os.environ.get("EMBED_PNG_1024X768_DIR", DATA_DIR / "EMBED" / "pngs" / "1024x768"))
EMBED_MEMMAP_1024X768 = Path(os.environ.get("EMBED_MEMMAP_1024X768", DATA_DIR / "embed_processed_1024x768.npy"))
EMBED_MEMMAP_512X384 = Path(os.environ.get("EMBED_MEMMAP_512X384", DATA_DIR / "embed_processed_512x384.npy"))

MAE_VIT_BASE_CHECKPOINT = Path(os.environ.get("MAE_INIT_CKPT", REPO_DIR / "checkpoints" / "mae" / "mae_pretrain_vit_base.pth"))
