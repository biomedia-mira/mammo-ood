# Data Preparation

The benchmark uses aligned PNG images and mmap-backed arrays.

## Raw Images

`tools/preprocess_images/` contains one converter per dataset. The converters handle DICOM decoding, mammography photometric conventions, left alignment, and resizing/padding to `1024x768`.

Most scripts contain placeholder paths under `/path/to/Mammo`; update these paths for the local dataset location.

## EMBED Pretraining Data

`analysis/study_sample_embed_pretrain.ipynb` creates `data/embed_pretrain.csv` from EMBED metadata and clinical tables.

Build the pretraining mmap with:

```bash
python tools/build_pretrain_mmap/embed.py \
  --csv-file data/embed_pretrain.csv \
  --image-root /path/to/Mammo/EMBED/pngs/1024x768 \
  --output-mmap data/embed_processed_512x384_pretrain.npy \
  --image-size 512 384
```

The builder preserves filtered CSV row order so `global_index` in the pretraining dataloader matches the mmap rows.

## Downstream Data

Downstream linear probing expects:

```text
<INFO_ROOT>/<dataset>/mmap_1024x768/
  metadata.parquet
  images_1024x768.npy
```

Required metadata columns:

```text
split, mmap_idx, sample_name, exam_id, side, view,
density, bi_rads, cancer_status
```
