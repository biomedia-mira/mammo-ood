# Model Checkpoints

Model weights are not included in this repository.

Set the checkpoint roots before running linear probing or inspection:

```bash
export CHECKPOINT_ROOT=/path/to/public_or_third_party_checkpoints
export ADAPTED_CHECKPOINT_ROOT=/path/to/embed_adapted_checkpoints
```

`CHECKPOINT_ROOT` is used for public and third-party backbones. `ADAPTED_CHECKPOINT_ROOT` is used for the EMBED-adapted DINOv2, DINOv3, and MAE backbones produced by the paper pretraining branch.

The model registry in `src/mammo_benchmark/config.py` defines the expected model keys, feature dimensions, input sizes, checkpoint filenames, and loading modes.
