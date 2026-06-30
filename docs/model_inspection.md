# Model Inspection

The inspection branch has three stages:

1. `analysis/resample_inspection.ipynb` creates fixed plotting samples for VinDr and EMBED.
2. `scripts/extract_inspection_embeddings.py` extracts native image features for selected frozen backbones.
3. `analysis/backbone_inspection_overlay_umap.ipynb` fits PCA/UMAP on the fixed cross-dataset sample and renders overlay panels.

The released fixed samples are:

- `data/inspection_samples/vindr_inspection_sample.csv`
- `data/inspection_samples/embed_inspection_sample.csv`

The UMAP notebook uses L2-normalized features, PCA to at most 50 components, and UMAP with `n_neighbors=80`, `min_dist=0.35`, `metric='cosine'`, `random_state=42`.

