# Reproduction

The main paper pipeline is:

1. Convert raw mammograms to aligned PNGs with `tools/preprocess_images/`.
2. Build mmap-backed downstream datasets with `tools/build_mmap/`.
3. Create `data/embed_pretrain.csv` with `analysis/study_sample_embed_pretrain.ipynb`.
4. Build the EMBED pretraining mmap with `tools/build_pretrain_mmap/embed.py`.
5. Run EMBED adaptation pretraining with `src/mammo_benchmark/pretraining/main.py`.
6. Run frozen-backbone linear probing with `scripts/run_linear_probe.py`.
7. Reproduce inspection figures with the notebooks in `analysis/`.

See the root `README.md` for minimal commands.
