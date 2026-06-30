import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from common.checkpointing import load_best_or_current_prefixed_state_dict, load_last_or_current_prefixed_state_dict
from common.cli import parse_bool, parse_limit_batches
from common.embed import load_embed_dataframe, split_embed_dataframe
from common.mmap_dataset import BaseMammoMMapDataset
from common.online_probe import OnlineProbe, extract_density_labels
from common.paths import (
    EMBED_CSV_PATH,
    EMBED_MEMMAP_512X384,
    EMBED_PNG_1024X768_DIR,
    MAE_VIT_BASE_CHECKPOINT,
    OUTPUT_SSL_DIR,
)
from common.trainer_utils import (
    build_logger,
    build_output_dir,
    build_strategy,
    build_trainer,
)



from transformers import get_cosine_schedule_with_warmup

# Import Facebook patched MAE
from architectures.vit import mae_vit_base_patch16
from methods.mae_checkpointing import load_mae_ssl_checkpoint

# Default Paths (Adjust as needed)
embed_data_dir = str(EMBED_PNG_1024X768_DIR)
mmap_file_path = str(EMBED_MEMMAP_512X384)
default_mae_ssl_checkpoint = str(MAE_VIT_BASE_CHECKPOINT)

# MAE default models are 224x224 patch-based
image_size = (512, 384) # (Height, Width) - 4:3 aspect ratio
mae_patch_size = 16

# Use ImageNet stats to match the Facebook MAE pretrain distribution — important when
# continue-pretraining from ImageNet MAE checkpoints to avoid distribution shock on
# the patch_embed conv.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _mae_param_groups_with_no_decay(model: torch.nn.Module, weight_decay: float, lr: float):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
    ]


class NoEmptyCrop:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xt = self.base_transform(x)
        if x.min() == x.max():
            return xt
        while xt.min() == xt.max():
            xt = self.base_transform(x)
        return xt


class TissueAwareCrop:
    """Crop that requires a minimum fraction of tissue (non-zero) pixels.

    Since the mmap is built with background zeroed out, tissue pixels are
    those > 0. Replaces NoEmptyCrop when tissue_aware_crop is enabled so
    crops land on meaningful breast tissue rather than mostly-empty
    background regions.
    """

    def __init__(self, base_transform, min_tissue_fraction: float = 0.3, max_retries: int = 20):
        self.base_transform = base_transform
        self.min_tissue_fraction = min_tissue_fraction
        self.max_retries = max_retries

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() == x.max():
            return self.base_transform(x)

        best_crop = None
        best_fraction = -1.0
        for _ in range(self.max_retries):
            crop = self.base_transform(x)
            fraction = (crop > 0).float().mean().item()
            if fraction >= self.min_tissue_fraction:
                return crop
            if fraction > best_fraction:
                best_fraction = fraction
                best_crop = crop
        return best_crop


class MammoDatasetMAE(BaseMammoMMapDataset):
    def __init__(
        self,
        data,
        image_size=(512, 384),
        image_normalization=65535.0,
        mmap_path=mmap_file_path,
        is_train=True,
        crop_scale=(0.4, 1.0),
        horizontal_flip=True,
        color_jitter=0.1,
        color_jitter_prob=0.3,
        tissue_aware_crop=False,
        min_tissue_fraction=0.3,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
    ):
        self.image_size = image_size
        self.labels = extract_density_labels(data)
        self.image_normalization = image_normalization
        self.is_train = is_train
        image_mean = tuple(float(v) for v in image_mean)
        image_std = tuple(float(v) for v in image_std)
        aspect_ratio = float(image_size[1]) / float(image_size[0])
        crop_ratio = (aspect_ratio * 0.95, aspect_ratio * 1.05)
        crop_scale = tuple(float(value) for value in crop_scale)
        if len(crop_scale) != 2:
            raise ValueError(f"Expected crop_scale with two values, got {crop_scale}.")
        flip_prob = 0.5 if horizontal_flip else 0.0


        if self.is_train:
            base_spatial = torchvision.transforms.Compose([
                # Keep scale relatively large and preserve breast geometry during crop-resize.
                torchvision.transforms.RandomResizedCrop(
                    image_size,
                    scale=crop_scale,
                    ratio=crop_ratio,
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),

                # Horizontal flip is anatomically valid. Never use vertical flip (gravity matters in CC/MLO views)
                torchvision.transforms.RandomHorizontalFlip(p=flip_prob),
            ])
            if tissue_aware_crop:
                self.spatial_transform = TissueAwareCrop(base_spatial, min_tissue_fraction=min_tissue_fraction)
            else:
                self.spatial_transform = NoEmptyCrop(base_spatial)
            self.photometric_transform = torchvision.transforms.Compose([
                # Keep intensity jitter very light so MAE still focuses on reconstructing breast morphology and fine details.
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter)
                ], p=color_jitter_prob),
                
                # Omit GaussianBlur and ElasticTransform: MAE should preserve microcalcifications, margins, and lesion shape.
                
                torchvision.transforms.Normalize(mean=list(image_mean), std=list(image_std))
            ])
        else:
            self.spatial_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
            ])
            self.photometric_transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean=list(image_mean), std=list(image_std))
            ])
            
        # MAE relies primarily on crop + flip, with only very light intensity jitter for scanner-style robustness.

        super().__init__(
            data,
            image_normalization=image_normalization,
            mmap_path=mmap_path,
            repeat_channels=3,
        )

    def __len__(self):
        return self.study_ids.shape[0]

    def __getitem__(self, index):
        image_tensor = self.load_image_tensor(index)
        spatial_image = self.spatial_transform(image_tensor)
        informative_patch_mask = (
            F.max_pool2d(spatial_image[:1], kernel_size=mae_patch_size, stride=mae_patch_size) > 0
        ).flatten()
        pixel_values = self.photometric_transform(spatial_image)
        return {
            'pixel_values': pixel_values,
            'informative_patch_mask': informative_patch_mask,
            'y': torch.tensor(self.labels[index], dtype=torch.long),
            'study_id': self.study_ids[index],
            'image_id': self.image_ids[index],
        }

class EMBEDMammoDataModuleMAE(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        csv_file,
        image_size,
        test_percent,
        val_percent,
        batch_size,
        num_workers,
        mmap_path,
        crop_scale,
        horizontal_flip,
        color_jitter,
        color_jitter_prob,
        tissue_aware_crop,
        min_tissue_fraction,
        image_mean,
        image_std,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.test_percent = test_percent
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mmap_path = mmap_path
        self.crop_scale = crop_scale
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter
        self.color_jitter_prob = color_jitter_prob
        self.tissue_aware_crop = bool(tissue_aware_crop)
        self.min_tissue_fraction = float(min_tissue_fraction)
        self.image_mean = tuple(float(v) for v in image_mean)
        self.image_std = tuple(float(v) for v in image_std)

        self.data = load_embed_dataframe(csv_file, self.data_dir)
        split_data = split_embed_dataframe(
            self.data,
            test_percent=self.test_percent,
            val_percent=self.val_percent,
            seed=42,
        )
        self.train_data = split_data['train']
        self.val_data = split_data['val']
        self.test_data = split_data['test']

    def setup(self, stage=None):
        self.train_set = MammoDatasetMAE(
            self.train_data,
            self.image_size,
            image_normalization=65535.0,
            mmap_path=self.mmap_path,
            is_train=True,
            crop_scale=self.crop_scale,
            horizontal_flip=self.horizontal_flip,
            color_jitter=self.color_jitter,
            color_jitter_prob=self.color_jitter_prob,
            tissue_aware_crop=self.tissue_aware_crop,
            min_tissue_fraction=self.min_tissue_fraction,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        self.val_set = MammoDatasetMAE(
            self.val_data, self.image_size, image_normalization=65535.0, mmap_path=self.mmap_path,
            is_train=False, image_mean=self.image_mean, image_std=self.image_std,
        )
        self.test_set = MammoDatasetMAE(
            self.test_data, self.image_size, image_normalization=65535.0, mmap_path=self.mmap_path,
            is_train=False, image_mean=self.image_mean, image_std=self.image_std,
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print('samples (train): ', len(self.train_set))
            print('samples (val):   ', len(self.val_set))
            print('samples (test):  ', len(self.test_set))
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)


class ViTMAELightning(pl.LightningModule):
    def __init__(
        self,
        model_name="facebook/vit-mae-base",
        learning_rate=1.5e-4,
        weight_decay=0.05,
        max_epochs=200,
        image_size=(512, 384),
        mask_ratio=0.75,
        masking_mode="full_image",
        init_mode="random",
        init_ckpt=None,
        init_ckpt_format="official",
        num_classes=4,
        online_probe_enabled=False,
        online_probe_learning_rate=1e-4,
        online_probe_weight_decay=1e-4,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.image_size = image_size
        self.mask_ratio = float(mask_ratio)
        if masking_mode not in {"full_image", "informative_first"}:
            raise ValueError(
                f"Unsupported masking_mode '{masking_mode}'. Expected 'full_image' or 'informative_first'."
            )
        self.masking_mode = masking_mode
        self.online_probe_enabled = bool(online_probe_enabled)
        self.online_probe_learning_rate = float(online_probe_learning_rate)
        self.online_probe_weight_decay = float(online_probe_weight_decay)
        self.register_buffer(
            "display_image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "display_image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )

        if init_mode not in {"random", "ssl"}:
            raise ValueError(f"Unsupported init_mode '{init_mode}'. Expected 'random' or 'ssl'.")

        # Instantiate patched Meta MAE expecting image_size=(H,W) to build 16x16 patch grids.
        self.model = mae_vit_base_patch16(img_size=self.image_size, norm_pix_loss=True)

        if init_mode == "ssl":
            if not init_ckpt:
                raise ValueError("init_ckpt is required when init_mode='ssl' for MAE.")
            load_mae_ssl_checkpoint(self.model, init_ckpt, init_ckpt_format)

        self.online_probe = OnlineProbe(self.model.embed_dim, num_classes) if self.online_probe_enabled else None
        
    def forward(self, pixel_values, informative_patch_mask=None):
        # The Custom MAE returns: loss, pred, mask
        if self.masking_mode == "full_image":
            informative_patch_mask = None
        return self.model(pixel_values, mask_ratio=self.mask_ratio, informative_patch_mask=informative_patch_mask)

    def _to_display_image(self, pixel_values):
        images = pixel_values.detach().float().cpu()
        mean = self.display_image_mean.detach().cpu()
        std = self.display_image_std.detach().cpu()
        return (images * std + mean).clamp(0.0, 1.0)

    def _log_mae_visualizations(self, pixel_values, pred, mask, batch_idx):
        if batch_idx != 0 or self.current_epoch % 10 != 0:
            return
        if not self.trainer.is_global_zero or self.logger is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        if not hasattr(experiment, "add_image"):
            return

        pixel_values = pixel_values[:8]
        pred = pred[:8]
        mask = mask[:8].unsqueeze(-1)

        with torch.no_grad():
            target = self.model.patchify(pixel_values)
            if self.model.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                std = target.var(dim=-1, keepdim=True).add(1.0e-6).sqrt()
                pred = pred * std + mean

            masked_patches = target * (1 - mask)
            overlay_patches = target * (1 - mask) + pred * mask

            masked_images = self.model.unpatchify(masked_patches)
            reconstruction_images = self.model.unpatchify(pred)
            overlay_images = self.model.unpatchify(overlay_patches)

        image_grids = {
            "mae/augmented_inputs": self._to_display_image(pixel_values),
            "mae/masked_inputs": self._to_display_image(masked_images),
            "mae/reconstructions": self._to_display_image(reconstruction_images),
            "mae/reconstruction_overlay": self._to_display_image(overlay_images),
        }
        for tag, images in image_grids.items():
            grid = torchvision.utils.make_grid(images, nrow=4, padding=2)
            experiment.add_image(tag, grid, global_step=self.global_step)

    def _log_masking_stats(self, mask, informative_patch_mask):
        if self.masking_mode != "informative_first" or informative_patch_mask is None:
            return
        if self.global_step % 200 != 0:
            return

        informative_patch_mask = informative_patch_mask.to(device=mask.device, dtype=mask.dtype)
        mask = mask.to(dtype=torch.float32)
        masked_per_sample = mask.sum(dim=1).clamp_min(1.0)
        informative_fraction = informative_patch_mask.mean(dim=1)
        masked_in_informative = (mask * informative_patch_mask).sum(dim=1)
        overflow_fraction = 1.0 - (masked_in_informative / masked_per_sample)

        self.log('train_informative_fraction_mean', informative_fraction.mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_overflow_fraction_mean', overflow_fraction.mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_overflow_fraction_max', overflow_fraction.max(), on_step=True, on_epoch=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        informative_patch_mask = batch.get('informative_patch_mask')
        mae_loss, pred, mask = self(pixel_values=pixel_values, informative_patch_mask=informative_patch_mask)
        self._log_mae_visualizations(pixel_values, pred, mask, batch_idx)
        self._log_masking_stats(mask, informative_patch_mask)
        self.log('train_loss', mae_loss, sync_dist=True)
        if self.online_probe is None:
            return mae_loss
        with torch.no_grad():
            latent, _, _ = self.model.forward_encoder(pixel_values, mask_ratio=0.0)
            features = latent[:, 0]
        probe_loss, probas = self.online_probe.step(features, batch['y'])
        self.online_probe.update('train', probas, batch['y'])
        self.log('train_probe_loss', probe_loss, sync_dist=True)
        return mae_loss + probe_loss
        
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        informative_patch_mask = batch.get('informative_patch_mask')
        mae_loss, _, _ = self(pixel_values=pixel_values, informative_patch_mask=informative_patch_mask)
        self.log('val_loss', mae_loss, sync_dist=True)
        if self.online_probe is None:
            return mae_loss
        with torch.no_grad():
            latent, _, _ = self.model.forward_encoder(pixel_values, mask_ratio=0.0)
            features = latent[:, 0]
        probe_loss, probas = self.online_probe.step(features, batch['y'])
        self.online_probe.update('val', probas, batch['y'])
        self.log('val_probe_loss', probe_loss, sync_dist=True)
        return mae_loss + probe_loss

    def on_train_epoch_start(self):
        if self.online_probe is not None:
            self.online_probe.reset('train')

    def on_validation_epoch_start(self):
        if self.online_probe is not None:
            self.online_probe.reset('val')

    def _log_probe_metrics(self, split: str):
        if self.online_probe is None:
            return
        metrics = self.online_probe.compute_metrics(split)
        if metrics is None:
            return
        self.log(f'{split}_probe_acc', metrics.acc, sync_dist=True)
        self.log(f'{split}_probe_balacc', metrics.balacc, sync_dist=True)
        if metrics.auroc is not None:
            self.log(f'{split}_probe_auroc', metrics.auroc, sync_dist=True)

    def on_train_epoch_end(self):
        self._log_probe_metrics('train')

    def on_validation_epoch_end(self):
        self._log_probe_metrics('val')

    def configure_optimizers(self):
        # Match the original MAE/timm recipe: do not decay bias or LayerNorm-like 1D params.
        params = _mae_param_groups_with_no_decay(self.model, self.hparams.weight_decay, self.lr)
        if self.online_probe is not None:
            params.append({
                'params': self.online_probe.parameters(),
                'lr': self.online_probe_learning_rate,
                'weight_decay': self.online_probe_weight_decay,
            })
        optim = torch.optim.AdamW(params, betas=(0.9, 0.95))
        
        # 5% warmup steps like the original MAE Table 8 config
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.05)
        
        scheduler = get_cosine_schedule_with_warmup(
            optim, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        return [optim], [{"scheduler": scheduler, "interval": "step"}]


def main(hparams):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(hparams.seed, workers=True)

    data = EMBEDMammoDataModuleMAE(
        data_dir=embed_data_dir,
        csv_file=hparams.csv_file,
        image_size=image_size,
        test_percent=hparams.test_percent,
        val_percent=hparams.val_percent,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        mmap_path=hparams.mmap_path,
        crop_scale=hparams.mae_crop_scale,
        horizontal_flip=hparams.mae_horizontal_flip,
        color_jitter=hparams.mae_color_jitter,
        color_jitter_prob=hparams.mae_color_jitter_prob,
        tissue_aware_crop=hparams.tissue_aware_crop,
        min_tissue_fraction=hparams.min_tissue_fraction,
        image_mean=hparams.image_mean,
        image_std=hparams.image_std,
    )

    model = ViTMAELightning(
        model_name=hparams.model,
        learning_rate=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
        max_epochs=hparams.epochs,
        image_size=image_size,
        mask_ratio=hparams.mask_ratio,
        masking_mode=hparams.masking_mode,
        init_mode=hparams.init_mode,
        init_ckpt=hparams.init_ckpt,
        init_ckpt_format=hparams.init_ckpt_format,
        num_classes=4,
        online_probe_enabled=hparams.online_probe_enabled,
        online_probe_learning_rate=hparams.online_probe_learning_rate,
        online_probe_weight_decay=hparams.online_probe_weight_decay,
        image_mean=hparams.image_mean,
        image_std=hparams.image_std,
    )

    output_dir = build_output_dir(hparams.output_root, hparams.output_name)

    print('\n=============================================================')
    print('TRAINING MAE (Masked Autoencoder)...')
    print('=============================================================\n')

    strategy = build_strategy(hparams.num_devices, find_unused_parameters=False)
    checkpoint_callback = ModelCheckpoint(
        monitor=hparams.checkpoint_metric,
        mode=hparams.checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )
    logger = build_logger(hparams.output_root, hparams.output_name)

    trainer = build_trainer(
        max_epochs=hparams.epochs,
        num_devices=hparams.num_devices,
        strategy=strategy,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
        logger=logger,
        precision=hparams.precision,
        sync_batchnorm=hparams.sync_batchnorm,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        num_sanity_val_steps=0,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
    )
    
    trainer.fit(model=model, datamodule=data)
    
    print('\n=============================================================')
    print('TRAINING COMPLETE. Saving BEST MAE wrapper...')
    print('=============================================================\n')

    if trainer.is_global_zero:
        best_model_state_dict = load_best_or_current_prefixed_state_dict(
            checkpoint_callback,
            'model.',
            model.model.state_dict(),
        )

        # Save the standalone MAE model weights instead of the full Lightning checkpoint.
        torch.save(best_model_state_dict, os.path.join(output_dir, 'mae_custom_pretrained_model.pth'))
        torch.save(best_model_state_dict, os.path.join(output_dir, 'mae_custom_pretrained_model_best.pth'))
        del best_model_state_dict

        last_model_state_dict = load_last_or_current_prefixed_state_dict(
            checkpoint_callback,
            'model.',
            model.model.state_dict(),
        )
        torch.save(last_model_state_dict, os.path.join(output_dir, 'mae_custom_pretrained_model_last.pth'))


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=3e-4) # Scaling LR for multi-gpu config
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--model', type=str, default='facebook/vit-mae-base')
    parser.add_argument('--init_mode', type=str, default='random', choices=['random', 'ssl'])
    parser.add_argument('--init_ckpt', type=str, default=default_mae_ssl_checkpoint)
    parser.add_argument('--init_ckpt_format', type=str, default='official', choices=['official', 'mammo_model'])
    parser.add_argument('--csv_file', type=str, default=str(EMBED_CSV_PATH))
    parser.add_argument('--mmap_path', type=str, default=mmap_file_path)
    parser.add_argument('--output_root', type=str, default=str(OUTPUT_SSL_DIR))
    parser.add_argument('--output_name', type=str, default='mae_run')
    parser.add_argument('--test_percent', type=float, default=0.2)
    parser.add_argument('--val_percent', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--sync_batchnorm', type=parse_bool, default=False)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--checkpoint_metric', type=str, default='val_loss')
    parser.add_argument('--checkpoint_mode', type=str, default='min')
    parser.add_argument('--online_probe_enabled', type=parse_bool, default=False)
    parser.add_argument('--online_probe_learning_rate', type=float, default=1e-4)
    parser.add_argument('--online_probe_weight_decay', type=float, default=1e-4)
    parser.add_argument('--limit_train_batches', type=parse_limit_batches, default=None)
    parser.add_argument('--limit_val_batches', type=parse_limit_batches, default=None)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--masking_mode', type=str, default='full_image', choices=['full_image', 'informative_first'])
    parser.add_argument('--mae_crop_scale', nargs=2, type=float, default=(0.4, 1.0))
    parser.add_argument('--mae_horizontal_flip', type=parse_bool, default=True)
    parser.add_argument('--mae_color_jitter', type=float, default=0.1)
    parser.add_argument('--mae_color_jitter_prob', type=float, default=0.3)
    parser.add_argument('--tissue_aware_crop', type=parse_bool, default=False)
    parser.add_argument('--min_tissue_fraction', type=float, default=0.3)
    parser.add_argument('--image_mean', nargs=3, type=float, default=list(IMAGENET_MEAN))
    parser.add_argument('--image_std', nargs=3, type=float, default=list(IMAGENET_STD))
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    main(args)
