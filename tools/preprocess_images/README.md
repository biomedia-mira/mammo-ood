# Raw Image Preprocessing

This folder contains the dataset-specific preprocessing scripts used before mmap
creation. The scripts convert raw mammography files into aligned PNG images at
the canonical resolution used by the benchmark.

The intended order is:

1. Convert raw DICOM/JPEG/PNG files to `1024x768` grayscale PNGs with the
   dataset-specific `convert_*.py` script.
2. Create the EMBED pretraining CSV with
   `analysis/study_sample_embed_pretrain.ipynb`.
3. Build the EMBED SSL pretraining mmap with `tools/build_pretrain_mmap/embed.py`.
4. Build downstream mmap arrays with `tools/build_mmap/`.

Most datasets require local raw-data paths to be edited at the top of the
corresponding script before running on a new machine.

## Dataset Scripts

| Dataset | Script |
| --- | --- |
| VinDr-Mammo | `convert_vindr.py` |
| EMBED | `convert_embed.py` |
| RSNA-site2 source data | `convert_rsna.py` |
| OPTIMAM | `export_optimam_csvs.py`, then `convert_optimam.py` |
| CBIS-DDSM | `convert_cbis_ddsm.py` |
| CDD-CESM | `convert_cdd_cesm.py` |
| INbreast | `convert_inbreast.py` |
| DMID | `convert_dmid.py` |
| BMCD | `convert_bcmd.py` |
| KAU-BCMD | `convert_kau.py` or `convert_kau_dicom.py` depending on raw format |
| CMMD | `convert_cmmd.py` |
| DBT | `convert_dbt.py` |
| MIAS | `convert_mias.py` |
| Mammogram Mastery | `convert_mm.py` |
| NLBS | `convert_nlbs.py` |

## OPTIMAM CSV Export

OPTIMAM uses an additional export step before image conversion:

```bash
python tools/preprocess_images/export_optimam_csvs.py
python tools/preprocess_images/convert_optimam.py --num-workers 8
```

Set `OPTIMAM_ROOT` when running on a machine where OPTIMAM is stored outside
the default placeholder location.
