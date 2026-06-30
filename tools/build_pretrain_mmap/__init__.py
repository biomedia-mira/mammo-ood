"""Pretraining mmap builders.

These scripts create the single `.npy` image mmap consumed by the SSL
pretraining dataloaders. This format is separate from downstream benchmark
mmaps, which also include `metadata.parquet`.
"""
