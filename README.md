# Benchmarking Foundation Models for Mammography under Domain Shift

This repository contains code for the paper:

> _Benchmarking the Robustness of Foundation Models for Mammography under Domain Shift_. Deep-Brea3th 2026: 3rd Deep Breast Workshop on AI and Imaging for Diagnostic and Treatment Challenges in Breast Care.

## Abstract

Foundation models are increasingly used as image feature extractors for mammography, but their robustness under external domain shift remains unclear. We benchmark 15 foundation-model backbones across breast density, BI-RADS severity, and cancer status using a unified frozen-backbone linear-probe protocol, training on 3 source datasets and evaluating on 12 task-compatible out-of-distribution (OOD) datasets after label harmonization. Mammography-specific vision-language models (`Mammo-FM` and `MaMA`) provide the strongest mean OOD performance, but robustness is not explained by mammography exposure alone. `DINOv3` remains a competitive vision-only baseline, and mammography-adapted pretraining does not consistently improve generalization. Dataset-level analysis further shows that even leading models show heterogeneous performance across datasets. Feature-space inspection reveals that useful representations can preserve clinical signal while retaining dataset and acquisition structure. These findings highlight dataset-level OOD evaluation as a central criterion for assessing mammography representations.

## Overview

<b>Configs</b> - `configs/` defines the model list, benchmark tasks, and EMBED adaptation pretraining configs used in the paper.

<b>Preprocessing</b> - `tools/preprocess_images/` converts raw mammography datasets to aligned PNG images. `tools/build_mmap/` converts processed images and harmonized metadata to mmap-backed downstream datasets.

<b>Pretraining</b> - `src/mammo_benchmark/pretraining/` contains the EMBED adaptation code for DINOv2, DINOv3, and MAE. The EMBED pretraining mmap builder is in `tools/build_pretrain_mmap/`.

<b>Linear probing</b> - `scripts/run_linear_probe.py` trains frozen-backbone linear heads and evaluates source and OOD datasets.

<b>Evaluation</b> - `src/mammo_benchmark/evaluation/` contains metric computation and exam-level aggregation. `scripts/bootstrap_eval.py` computes bootstrap confidence intervals from prediction CSVs.

<b>Model inspection</b> - `analysis/` contains notebooks for selecting inspection samples, rendering the 5x5 image grid, and visualizing backbone feature spaces with PCA/UMAP.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

Raw datasets, processed mmap arrays, model checkpoints, and experiment outputs are not included.

Downstream datasets are expected under an `INFO_ROOT` directory:

```text
<INFO_ROOT>/<dataset>/mmap_1024x768/
  metadata.parquet
  images_1024x768.npy
```

The metadata table should contain the harmonized labels used by the benchmark:

```text
split, mmap_idx, sample_name, exam_id, side, view,
bi_rads, density, cancer_status
```

Dataset conversion scripts use placeholder paths such as `/path/to/Mammo`; update these paths or pass script arguments for a local setup. Additional notes are in `docs/data_preparation.md`.

## Usage

Build the EMBED pretraining mmap:

```bash
python tools/build_pretrain_mmap/embed.py \
  --csv-file data/embed_pretrain.csv \
  --image-root /path/to/Mammo/EMBED/pngs/1024x768 \
  --output-mmap data/embed_processed_512x384_pretrain.npy \
  --image-size 512 384
```

Run EMBED adaptation pretraining:

```bash
PYTHONPATH=$PWD/src:$PWD/src/mammo_benchmark/pretraining \
python src/mammo_benchmark/pretraining/main.py \
  experiment=dinov3_embed \
  data.csv_file=$PWD/data/embed_pretrain.csv \
  data.mmap_path=$PWD/data/embed_processed_512x384_pretrain.npy
```

Run a frozen-backbone linear probe:

```bash
python scripts/run_linear_probe.py \
  --experiment all \
  --model dinov3_embed_vitb_in \
  --info_root /path/to/classification_data \
  --checkpoint_root /path/to/checkpoints \
  --adapted_checkpoint_root /path/to/adapted_checkpoints
```

Extract inspection embeddings:

```bash
python scripts/extract_inspection_embeddings.py \
  --datasets vindr,embed \
  --models dinov3_vitb,dinov3_embed_vitb_in \
  --info_root /path/to/classification_data \
  --checkpoint_root /path/to/checkpoints \
  --adapted_checkpoint_root /path/to/adapted_checkpoints
```

## Tasks and Models

The benchmark evaluates breast density, BI-RADS severity, and cancer status. Task definitions and OOD dataset lists are in `configs/experiments/`.

The 15 evaluated backbones are listed in `configs/models.yaml`; the executable registry is `src/mammo_benchmark/config.py`.

## Citation

```bibtex
@inproceedings{anonymous2026mammoood,
  title     = {Benchmarking the Robustness of Foundation Models for Mammography under Domain Shift},
  author    = {Anonymous Authors},
  booktitle = {Deep-Brea3th 2026: 3rd Deep Breast Workshop on AI and Imaging for Diagnostic and Treatment Challenges in Breast Care},
  year      = {2026}
}
```
