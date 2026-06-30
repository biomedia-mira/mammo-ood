from __future__ import annotations

import gc
import math
import os
from argparse import ArgumentParser
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from architectures.dinov3.configs import get_default_config
from architectures.dinov3.data.collate import collate_data_and_cast
from architectures.dinov3.data.masking import MaskingGenerator
from architectures.dinov3.train.cosine_lr_scheduler import CosineScheduler
from architectures.dinov3.train.ssl_meta_arch import SSLMetaArch
from common.cli import parse_bool, parse_limit_batches
from common.checkpointing import load_best_or_current_prefixed_state_dict, load_last_or_current_prefixed_state_dict
from common.embed import load_embed_dataframe, split_embed_dataframe
from common.mmap_dataset import BaseMammoMMapDataset
from common.online_probe import MammoDensityProbeDataset, OnlineProbe, extract_density_labels
from common.paths import EMBED_CSV_PATH, EMBED_MEMMAP_512X384, EMBED_PNG_1024X768_DIR, OUTPUT_SSL_DIR
from common.token_heatmap_callback import TokenHeatmapCallback
from common.trainer_utils import build_logger, build_output_dir, build_strategy, build_trainer
from methods.dino_common import PairedLoader, log_dino_crops
from methods.dinov3_augmentations import MammoDataAugmentationDINOv3, MammoValAugmentationDINOv3


embed_data_dir = str(EMBED_PNG_1024X768_DIR)
mmap_file_path = str(EMBED_MEMMAP_512X384)
image_size = (512, 384)
local_image_size = (192, 144)


def _strip_prefixed_backbone(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    extracted = {
        key.replace(prefix, "", 1): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not extracted:
        raise ValueError(f"Could not find any weights for prefix '{prefix}'")
    return extracted


def _save_backbone_exports(state_dict: dict[str, torch.Tensor], output_dir: str, *, suffix: str = "") -> None:
    student_backbone = _strip_prefixed_backbone(state_dict, "student.backbone.")
    teacher_backbone = _strip_prefixed_backbone(state_dict, "teacher.backbone.")
    torch.save(student_backbone, os.path.join(output_dir, f"dinov3_student_backbone{suffix}.pth"))
    torch.save(teacher_backbone, os.path.join(output_dir, f"dinov3_teacher_backbone{suffix}.pth"))


class MammoDatasetDINOv3(BaseMammoMMapDataset):
    def __init__(self, data, transform, mmap_path=mmap_file_path, input_channels=3, image_normalization=65535.0):
        self.transform = transform
        self.input_channels = input_channels
        self.labels = extract_density_labels(data)
        super().__init__(
            data,
            image_normalization=image_normalization,
            mmap_path=mmap_path,
            repeat_channels=1 if input_channels == 1 else input_channels,
        )

    def __getitem__(self, index):
        img = self.load_image_tensor(index)
        if self.input_channels == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        transformed = self.transform(img.float())
        transformed['y'] = torch.tensor(self.labels[index], dtype=torch.long)
        return transformed, ()


class EMBEDMammoDataModuleDINOv3(pl.LightningDataModule):
    def __init__(
        self,
        *,
        data_dir: str,
        csv_file: str,
        image_size: Sequence[int],
        test_percent: float,
        val_percent: float,
        batch_size: int,
        num_workers: int,
        mmap_path: str,
        input_channels: int,
        global_crops_scale: Sequence[float],
        local_crops_scale: Sequence[float],
        local_crops_number: int,
        local_crops_size: Sequence[int],
        patch_size: int,
        ibot_mask_ratio_min_max: Sequence[float],
        mask_probability: float,
        mask_random_circular_shift: bool,
        precision_dtype: str,
        horizontal_flip: bool,
        share_color_jitter: bool,
        random_color_jitter: float,
        color_jitter_prob: float,
        mean: Sequence[float],
        std: Sequence[float],
        online_probe_enabled: bool = False,
        tissue_aware_crop: bool = False,
        tissue_aware_mask: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_size = tuple(image_size)
        self.test_percent = test_percent
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mmap_path = mmap_path
        self.input_channels = input_channels
        self.global_crops_scale = tuple(global_crops_scale)
        self.local_crops_scale = tuple(local_crops_scale)
        self.local_crops_number = local_crops_number
        self.local_crops_size = tuple(local_crops_size)
        self.patch_size = patch_size
        self.ibot_mask_ratio_min_max = tuple(ibot_mask_ratio_min_max)
        self.mask_probability = mask_probability
        self.mask_random_circular_shift = bool(mask_random_circular_shift)
        self.precision_dtype = precision_dtype
        self.horizontal_flip = horizontal_flip
        self.share_color_jitter = share_color_jitter
        self.random_color_jitter = random_color_jitter
        self.color_jitter_prob = float(color_jitter_prob)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.online_probe_enabled = bool(online_probe_enabled)
        self.tissue_aware_crop = bool(tissue_aware_crop)
        self.tissue_aware_mask = bool(tissue_aware_mask)
        self.probe_train_set = None
        self.probe_val_set = None
        self._train_samplers = []

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
        train_transform = MammoDataAugmentationDINOv3(
            global_crops_scale=self.global_crops_scale,
            local_crops_scale=self.local_crops_scale,
            local_crops_number=self.local_crops_number,
            global_crops_size=self.image_size,
            local_crops_size=self.local_crops_size,
            input_channels=self.input_channels,
            horizontal_flips=self.horizontal_flip,
            share_color_jitter=self.share_color_jitter,
            random_color_jitter=self.random_color_jitter,
            color_jitter_prob=self.color_jitter_prob,
            mean=self.mean,
            std=self.std,
            tissue_aware_crop=self.tissue_aware_crop,
            tissue_aware_mask=self.tissue_aware_mask,
            patch_size=self.patch_size,
        )
        val_transform = MammoValAugmentationDINOv3(
            global_crops_size=self.image_size,
            local_crops_size=self.local_crops_size,
            local_crops_number=self.local_crops_number,
            input_channels=self.input_channels,
            mean=self.mean,
            std=self.std,
            tissue_aware_mask=self.tissue_aware_mask,
            patch_size=self.patch_size,
        )
        self.dataset_train = MammoDatasetDINOv3(
            self.train_data,
            transform=train_transform,
            mmap_path=self.mmap_path,
            input_channels=self.input_channels,
        )
        self.dataset_val = MammoDatasetDINOv3(
            self.val_data,
            transform=val_transform,
            mmap_path=self.mmap_path,
            input_channels=self.input_channels,
        )

        if self.online_probe_enabled:
            repeat_channels = 1 if self.input_channels == 1 else self.input_channels
            self.probe_train_set = MammoDensityProbeDataset(
                self.train_data,
                image_size=self.image_size,
                mmap_path=self.mmap_path,
                repeat_channels=repeat_channels,
                mean=self.mean,
                std=self.std,
            )
            self.probe_val_set = MammoDensityProbeDataset(
                self.val_data,
                image_size=self.image_size,
                mmap_path=self.mmap_path,
                repeat_channels=repeat_channels,
                mean=self.mean,
                std=self.std,
            )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print('samples (train): ', len(self.dataset_train))
            print('samples (val):   ', len(self.val_data))
            print('samples (test):  ', len(self.test_data))

    def get_collate_fn(self):
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        mask_generator = MaskingGenerator(
            input_size=(self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size),
            max_num_patches=int(0.5 * n_tokens),
        )
        dtype = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }[self.precision_dtype]
        tissue_aware_mask = self.tissue_aware_mask
        return lambda samples: collate_data_and_cast(
            samples,
            mask_ratio_tuple=self.ibot_mask_ratio_min_max,
            mask_probability=self.mask_probability,
            dtype=dtype,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            random_circular_shift=self.mask_random_circular_shift,
            local_batch_size=None,
            tissue_aware_mask=tissue_aware_mask,
        )

    def _make_distributed_sampler(self, dataset, *, shuffle: bool, drop_last: bool):
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def set_epoch(self, epoch: int) -> None:
        for sampler in self._train_samplers:
            sampler.set_epoch(epoch)

    def train_dataloader(self):
        ssl_sampler = self._make_distributed_sampler(self.dataset_train, shuffle=True, drop_last=True)
        ssl_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=ssl_sampler is None,
            sampler=ssl_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            collate_fn=self.get_collate_fn(),
        )
        if not self.online_probe_enabled or self.probe_train_set is None:
            self._train_samplers = [sampler for sampler in (ssl_sampler,) if sampler is not None]
            return ssl_loader
        probe_sampler = self._make_distributed_sampler(self.probe_train_set, shuffle=True, drop_last=True)
        probe_loader = DataLoader(
            self.probe_train_set,
            batch_size=self.batch_size,
            shuffle=probe_sampler is None,
            sampler=probe_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
        self._train_samplers = [sampler for sampler in (ssl_sampler, probe_sampler) if sampler is not None]
        return PairedLoader(ssl_loader, probe_loader)

    def val_dataloader(self):
        ssl_sampler = self._make_distributed_sampler(self.dataset_val, shuffle=False, drop_last=False)
        ssl_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=ssl_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            collate_fn=self.get_collate_fn(),
        )
        if self.probe_val_set is None:
            return ssl_loader
        probe_sampler = self._make_distributed_sampler(self.probe_val_set, shuffle=False, drop_last=False)
        probe_loader = DataLoader(
            self.probe_val_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=probe_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )
        return PairedLoader(ssl_loader, probe_loader)

    def test_dataloader(self):
        return None


def _precision_to_dino_dtype(precision: str) -> str:
    precision = str(precision).lower()
    if 'bf16' in precision:
        return 'bf16'
    if precision.startswith('16') or 'fp16' in precision or 'half' in precision:
        return 'fp16'
    return 'fp32'


def _default_channel_stats(input_channels: int, normalization: str = 'imagenet') -> tuple[list[float], list[float]]:
    normalization = str(normalization).lower()
    if normalization in {'half', '0.5', 'zero_center'}:
        return [0.5] * int(input_channels), [0.5] * int(input_channels)
    if normalization != 'imagenet':
        raise ValueError(f"Unsupported normalization '{normalization}'. Expected 'imagenet' or 'half'.")

    # ImageNet stats match the original DINOv3 pretrain distribution.
    if int(input_channels) == 1:
        mean = sum((0.485, 0.456, 0.406)) / 3.0
        std = sum((0.229, 0.224, 0.225)) / 3.0
        return [mean], [std]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _resolve_official_epoch_length(datamodule: pl.LightningDataModule, *, world_size: int = 1) -> int:
    world_size = max(int(world_size), 1)
    if world_size > 1:
        dataset_lengths = [len(datamodule.dataset_train)]
        probe_train_set = getattr(datamodule, 'probe_train_set', None)
        if probe_train_set is not None:
            dataset_lengths.append(len(probe_train_set))
        batches_per_rank = [
            (dataset_length // world_size) // datamodule.batch_size
            for dataset_length in dataset_lengths
        ]
        return max(min(batches_per_rank), 1)

    train_loader = datamodule.train_dataloader()
    try:
        return max(int(len(train_loader)), 1)
    except TypeError as exc:  # pragma: no cover - defensive path
        raise RuntimeError(
            "Could not determine DINOv3 OFFICIAL_EPOCH_LENGTH from the train dataloader. "
            "The train loader must implement __len__ so the optimizer schedule can be built correctly."
        ) from exc


def _build_dinov3_cfg(hparams, output_dir: str):
    cfg = get_default_config()
    OmegaConf.set_struct(cfg, False)

    mean, std = _default_channel_stats(hparams.input_channels, hparams.normalization)
    precision_dtype = _precision_to_dino_dtype(hparams.precision)

    cfg.train.output_dir = output_dir
    cfg.train.batch_size_per_gpu = hparams.batch_size
    cfg.train.num_workers = hparams.num_workers
    cfg.train.seed = hparams.seed
    cfg.train.OFFICIAL_EPOCH_LENGTH = 1
    cfg.train.compile = hparams.compile
    cfg.train.cudagraphs = hparams.cudagraphs
    cfg.train.checkpointing = hparams.checkpointing
    cfg.train.checkpointing_full = False
    cfg.train.cache_dataset = False

    cfg.compute_precision.param_dtype = precision_dtype
    cfg.compute_precision.reduce_dtype = 'fp32'
    cfg.compute_precision.sharding_strategy = 'SHARD_GRAD_OP'

    cfg.student.arch = hparams.model
    cfg.student.patch_size = hparams.patch_size
    cfg.student.in_chans = hparams.input_channels
    cfg.student.n_storage_tokens = hparams.num_storage_tokens
    cfg.student.mask_k_bias = True
    cfg.student.pos_embed_rope_rescale_coords = 2
    cfg.student.pos_embed_rope_dtype = "fp32"
    cfg.student.norm_layer = "layernormbf16"
    cfg.student.drop_path_rate = hparams.drop_path_rate
    cfg.student.pretrained_weights = ''
    cfg.student.resume_from_teacher_chkpt = ''
    cfg.teacher.in_chans = hparams.input_channels

    cfg.crops.global_crops_scale = list(hparams.global_crops_scale)
    cfg.crops.local_crops_scale = list(hparams.local_crops_scale)
    cfg.crops.local_crops_number = hparams.local_crops_number
    cfg.crops.global_crops_size = list(hparams.image_size)
    cfg.crops.local_crops_size = list(hparams.local_crops_size)
    cfg.crops.gram_teacher_crops_size = None
    cfg.crops.localcrops_subset_of_globalcrops = False
    cfg.crops.share_color_jitter = hparams.share_color_jitter
    cfg.crops.horizontal_flips = hparams.horizontal_flip
    cfg.crops.gram_teacher_no_distortions = False
    cfg.crops.rgb_mean = mean
    cfg.crops.rgb_std = std

    cfg.dino.head_n_prototypes = hparams.head_n_prototypes
    cfg.dino.head_bottleneck_dim = hparams.head_bottleneck_dim
    cfg.dino.head_hidden_dim = hparams.head_hidden_dim
    cfg.dino.head_nlayers = hparams.head_nlayers
    cfg.dino.koleo_loss_weight = hparams.koleo_loss_weight

    cfg.ibot.loss_weight = hparams.ibot_loss_weight
    cfg.ibot.mask_ratio_min_max = list(hparams.ibot_mask_ratio_min_max)
    cfg.ibot.mask_sample_probability = hparams.mask_probability
    cfg.ibot.mask_random_circular_shift = hparams.mask_random_circular_shift

    cfg.gram.use_loss = False
    cfg.gram.compute_stats = False
    cfg.gram.ema_teacher = False
    cfg.gram.ckpt = None
    cfg.gram.it_load_ema_teacher = -1

    cfg.teacher.momentum_teacher = hparams.momentum_teacher
    cfg.teacher.final_momentum_teacher = hparams.final_momentum_teacher
    cfg.teacher.warmup_teacher_temp = hparams.warmup_teacher_temp
    cfg.teacher.teacher_temp = hparams.teacher_temp
    cfg.teacher.warmup_teacher_temp_epochs = hparams.warmup_teacher_temp_epochs

    cfg.optim.epochs = hparams.epochs
    cfg.optim.lr = hparams.learning_rate
    cfg.optim.weight_decay = hparams.weight_decay
    cfg.optim.weight_decay_end = hparams.weight_decay_end
    cfg.optim.warmup_epochs = hparams.warmup_epochs
    cfg.optim.min_lr = hparams.min_learning_rate
    cfg.optim.clip_grad = hparams.clip_grad
    cfg.optim.freeze_last_layer_epochs = hparams.freeze_last_layer_epochs
    cfg.optim.layerwise_decay = hparams.layerwise_decay
    cfg.optim.patch_embed_lr_mult = hparams.patch_embed_lr_mult
    cfg.optim.dino_head_wd_multiplier = hparams.dino_head_wd_multiplier
    cfg.optim.adamw_beta1 = hparams.adamw_beta1
    cfg.optim.adamw_beta2 = hparams.adamw_beta2

    return cfg


class DINOv3Lightning(pl.LightningModule):
    def __init__(self, *, cfg, init_mode: str = 'random', init_ckpt: str | None = None, num_classes: int = 4, online_probe_enabled: bool = False, online_probe_learning_rate: float = 1e-4, online_probe_weight_decay: float = 1e-4):
        super().__init__()
        self.cfg = cfg
        self.init_mode = init_mode
        self.init_ckpt = init_ckpt
        self.lr_schedule = None
        self.wd_schedule = None
        self.momentum_schedule = None
        self.teacher_temp_schedule = None
        self.last_layer_lr_schedule = None
        self.current_iteration = 0
        self._fit_ready = False
        self._model_on_cpu = False
        self.save_hyperparameters({'init_mode': init_mode, 'init_ckpt': init_ckpt})
        self.automatic_optimization = False
        self.online_probe_enabled = bool(online_probe_enabled)
        self.online_probe_learning_rate = float(online_probe_learning_rate)
        self.online_probe_weight_decay = float(online_probe_weight_decay)
        self._build_model()
        self.online_probe = OnlineProbe(self.ssl_model.embed_dim, num_classes) if self.online_probe_enabled else None

    def _build_model(self):
        with torch.device('meta'):
            self.ssl_model = SSLMetaArch(self.cfg)
        expected_in_chans = int(self.cfg.student.in_chans)
        student_in_chans = int(self.ssl_model.student.backbone.patch_embed.proj.weight.shape[1])
        teacher_in_chans = int(self.ssl_model.teacher.backbone.patch_embed.proj.weight.shape[1])
        if student_in_chans != expected_in_chans or teacher_in_chans != expected_in_chans:
            raise RuntimeError(
                "DINOv3 model/data channel mismatch during build: "
                f"expected in_chans={expected_in_chans}, "
                f"student patch_embed={student_in_chans}, teacher patch_embed={teacher_in_chans}."
            )
        self.ssl_model._apply(
            lambda t: torch.full_like(
                t,
                fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
                device='cpu',
            ),
            recurse=True,
        )
        self._model_on_cpu = True

    def setup(self, stage: str):
        if stage != 'fit' or self._fit_ready:
            return
        if getattr(self, '_model_on_cpu', False):
            self.ssl_model = self.ssl_model.to(self.device)
            self._model_on_cpu = False
        self.ssl_model.init_weights()
        if self.init_mode == 'ssl' and self.init_ckpt:
            self._load_pretrained_backbone(self.init_ckpt)
        self._fit_ready = True

    def _load_pretrained_backbone(self, ckpt_path: str) -> None:
        """Load pretrained DINOv3 backbone weights into student + teacher."""
        self.print(f'[DINOv3] Loading pretrained backbone from {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']

        student_backbone = self.ssl_model.student.backbone
        expected_in_chans = int(self.cfg.student.in_chans)
        pe_key = 'patch_embed.proj.weight'
        if pe_key in state_dict and state_dict[pe_key].shape[1] != expected_in_chans:
            raise RuntimeError(
                "[DINOv3] Checkpoint/data channel mismatch: "
                f"checkpoint patch_embed expects {state_dict[pe_key].shape[1]} channels, "
                f"but model is configured for {expected_in_chans}. "
                "Use matching input_channels or add an explicit adaptation path."
            )

        # Load into student backbone
        missing, unexpected = student_backbone.load_state_dict(state_dict, strict=False)
        self.print(f'[DINOv3] Student backbone load — missing: {missing}, unexpected: {unexpected}')

        # Copy student backbone to teacher backbone (EMA target)
        teacher_backbone = self.ssl_model.teacher.backbone
        teacher_backbone.load_state_dict(student_backbone.state_dict(), strict=True)

        # Also sync the full model_ema (includes heads from init_weights)
        self.ssl_model.model_ema.load_state_dict(self.ssl_model.student.state_dict())
        self.print('[DINOv3] Pretrained backbone loaded into student, teacher, and EMA.')

    def on_fit_start(self):
        official_epoch_length = max(int(self.cfg.train.OFFICIAL_EPOCH_LENGTH), 1)
        if official_epoch_length <= 1:
            num_training_batches = self.trainer.num_training_batches
            if isinstance(num_training_batches, float) and not math.isfinite(num_training_batches):
                official_epoch_length = 1
            else:
                official_epoch_length = max(int(num_training_batches), 1)
            self.cfg.train.OFFICIAL_EPOCH_LENGTH = official_epoch_length
            self.print(
                f"[DINOv3][Debug] OFFICIAL_EPOCH_LENGTH fallback from trainer.num_training_batches -> {official_epoch_length}"
            )
        self._build_schedules()
        self._debug_print_schedule_setup()

    def _debug_print_schedule_setup(self) -> None:
        if not self.trainer.is_global_zero:
            return

        total_iters = max(int(self.cfg.train.OFFICIAL_EPOCH_LENGTH) * max(int(self.cfg.optim.epochs), 1), 1)
        warmup_iters = min(int(self.cfg.optim.warmup_epochs), int(self.cfg.optim.epochs)) * int(
            self.cfg.train.OFFICIAL_EPOCH_LENGTH
        )

        if self.lr_schedule is None:
            self.print("[DINOv3][Debug] lr_schedules not initialized")
            return

        warmup_end_it = min(max(warmup_iters - 1, 0), total_iters - 1)
        lr_start = self._schedule_value(self.lr_schedule, 0, self.cfg.optim.lr)
        wd_start = self._schedule_value(self.wd_schedule, 0, self.cfg.optim.weight_decay)
        temp_start = self._schedule_value(self.teacher_temp_schedule, 0, self.cfg.teacher.teacher_temp)
        lr_warmup_end = self._schedule_value(self.lr_schedule, warmup_end_it, self.cfg.optim.lr)
        lr_final = self._schedule_value(self.lr_schedule, total_iters - 1, self.cfg.optim.min_lr)
        wd_final = self._schedule_value(self.wd_schedule, total_iters - 1, self.cfg.optim.weight_decay_end)
        temp_final = self._schedule_value(self.teacher_temp_schedule, total_iters - 1, self.cfg.teacher.teacher_temp)

        ssl_optimizer = self.trainer.optimizers[0]
        lr_multipliers = [float(pg.get('lr_multiplier', 1.0)) for pg in ssl_optimizer.param_groups]
        wd_multipliers = [float(pg.get('wd_multiplier', 1.0)) for pg in ssl_optimizer.param_groups]
        probe_lr = None
        if len(self.trainer.optimizers) > 1:
            probe_lr = float(self.trainer.optimizers[1].param_groups[0]['lr'])

        self.print(
            "[DINOv3][Debug] schedule "
            f"epoch_len={int(self.cfg.train.OFFICIAL_EPOCH_LENGTH)} total_iters={total_iters} warmup_iters={warmup_iters} "
            f"lr(start={lr_start:.3e}, warmup_end={lr_warmup_end:.3e}, final={lr_final:.3e}) "
            f"wd(start={wd_start:.3e}, final={wd_final:.3e}) "
            f"teacher_temp(start={temp_start:.3e}, final={temp_final:.3e})"
        )
        self.print(
            "[DINOv3][Debug] optimizer "
            f"ssl_groups={len(ssl_optimizer.param_groups)} "
            f"lr_mult[min,max]=({min(lr_multipliers):.3e}, {max(lr_multipliers):.3e}) "
            f"wd_mult[min,max]=({min(wd_multipliers):.3e}, {max(wd_multipliers):.3e}) "
            f"probe_lr={probe_lr if probe_lr is not None else 'disabled'}"
        )

    def _build_schedules(self):
        official_epoch_length = max(int(self.cfg.train.OFFICIAL_EPOCH_LENGTH), 1)
        total_epochs = max(int(self.cfg.optim.epochs), 1)
        warmup_epochs = min(int(self.cfg.optim.warmup_epochs), total_epochs)
        freeze_last_layer_epochs = min(int(self.cfg.optim.freeze_last_layer_epochs), total_epochs)
        warmup_teacher_temp_epochs = min(int(self.cfg.teacher.warmup_teacher_temp_epochs), total_epochs)
        total_iters = max(total_epochs * official_epoch_length, 1)

        lr_schedule = CosineScheduler(
            base_value=self.cfg.optim.lr,
            final_value=self.cfg.optim.min_lr,
            total_iters=total_iters,
            warmup_iters=warmup_epochs * official_epoch_length,
            start_warmup_value=0,
            trunc_extra=self.cfg.optim.get('schedule_trunc_extra', False),
        )
        wd_schedule = CosineScheduler(
            base_value=self.cfg.optim.weight_decay,
            final_value=self.cfg.optim.weight_decay_end,
            total_iters=total_iters,
            trunc_extra=self.cfg.optim.get('schedule_trunc_extra', False),
        )
        momentum_schedule = CosineScheduler(
            base_value=self.cfg.teacher.momentum_teacher,
            final_value=self.cfg.teacher.final_momentum_teacher,
            total_iters=total_iters,
            trunc_extra=self.cfg.optim.get('schedule_trunc_extra', False),
        )
        teacher_temp_schedule = CosineScheduler(
            base_value=self.cfg.teacher.teacher_temp,
            final_value=self.cfg.teacher.teacher_temp,
            total_iters=max(warmup_teacher_temp_epochs * official_epoch_length, 1),
            warmup_iters=max(warmup_teacher_temp_epochs * official_epoch_length, 1),
            start_warmup_value=self.cfg.teacher.warmup_teacher_temp,
        )
        last_layer_lr_schedule = CosineScheduler(
            base_value=self.cfg.optim.lr,
            final_value=self.cfg.optim.min_lr,
            total_iters=total_iters,
            warmup_iters=warmup_epochs * official_epoch_length,
            start_warmup_value=0,
            trunc_extra=self.cfg.optim.get('schedule_trunc_extra', False),
        )
        freeze_iters = freeze_last_layer_epochs * official_epoch_length
        last_layer_lr_schedule.schedule[:freeze_iters] = 0
        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule
        self.momentum_schedule = momentum_schedule
        self.teacher_temp_schedule = teacher_temp_schedule
        self.last_layer_lr_schedule = last_layer_lr_schedule

    @staticmethod
    def _schedule_value(schedule, iteration: int, default: float) -> float:
        if schedule is None:
            return float(default)
        return float(schedule[iteration])

    def configure_optimizers(self):
        params_groups = self.ssl_model.get_params_groups()
        ssl_optimizer = AdamW(
            params_groups,
            betas=(self.cfg.optim.adamw_beta1, self.cfg.optim.adamw_beta2),
        )
        if self.online_probe is None:
            return ssl_optimizer
        probe_optimizer = AdamW(
            self.online_probe.parameters(),
            lr=self.online_probe_learning_rate,
            weight_decay=self.online_probe_weight_decay,
            betas=(self.cfg.optim.adamw_beta1, self.cfg.optim.adamw_beta2),
        )
        return ssl_optimizer, probe_optimizer

    def _apply_optim_scheduler(self, optimizer, lr, wd, last_layer_lr):
        for param_group in optimizer.param_groups:
            is_last_layer = param_group.get('is_last_layer', False)
            lr_multiplier = param_group.get('lr_multiplier', 1.0)
            wd_multiplier = param_group.get('wd_multiplier', 1.0)
            param_group['weight_decay'] = wd * wd_multiplier
            param_group['lr'] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

    def _compute_ssl_val_loss(self, batch, teacher_temp_val, iteration):
        n_global_crops = 2
        n_local_crops = self.ssl_model.n_local_crops
        B = batch['collated_local_crops'].shape[0] // n_local_crops
        device = next(self.ssl_model.student.backbone.parameters()).device
        global_crops = batch['collated_global_crops'].to(device, non_blocking=True)
        local_crops = batch['collated_local_crops'].to(device, non_blocking=True)
        masks = batch['collated_masks'].to(device, non_blocking=True)
        mask_indices_list = batch['mask_indices_list'].to(device, non_blocking=True)
        masks_weight = batch['masks_weight'].to(device, non_blocking=True)
        n_masked_patches_tensor = batch['n_masked_patches'].to(device, non_blocking=True)

        teacher_global = self.ssl_model.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp_val,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=batch['upperbound'],
        )
        student_global, student_local = self.ssl_model.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=batch['upperbound'],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )
        loss, loss_dict = self.ssl_model.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global={},
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
        )
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        ssl_batch = batch['ssl'] if isinstance(batch, dict) and 'ssl' in batch else batch
        probe_batch = batch.get('probe') if isinstance(batch, dict) else None
        log_dino_crops(
            self,
            ssl_batch,
            batch_idx,
            every_n_epochs=3,
            mean=self.cfg.crops.rgb_mean,
            std=self.cfg.crops.rgb_std,
        )

        optimizers = self.optimizers()
        optimizer = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
        it = self.current_iteration

        lr_val = self._schedule_value(self.lr_schedule, it, self.cfg.optim.lr)
        wd_val = self._schedule_value(self.wd_schedule, it, self.cfg.optim.weight_decay)
        mom_val = self._schedule_value(self.momentum_schedule, it, self.cfg.teacher.momentum_teacher)
        teacher_temp_val = self._schedule_value(self.teacher_temp_schedule, it, self.cfg.teacher.teacher_temp)
        last_layer_lr_val = self._schedule_value(self.last_layer_lr_schedule, it, self.cfg.optim.lr)
        self._apply_optim_scheduler(optimizer, lr_val, wd_val, last_layer_lr_val)

        ssl_group_lrs = [float(param_group['lr']) for param_group in optimizer.param_groups]
        ssl_lr_min = min(ssl_group_lrs)
        ssl_lr_max = max(ssl_group_lrs)

        ssl_batch['global_batch_size'] = self.cfg.train.batch_size_per_gpu * self.trainer.world_size

        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics_dict = self.ssl_model.forward_backward(
            ssl_batch,
            teacher_temp=teacher_temp_val,
            iteration=it,
        )

        if self.cfg.optim.clip_grad:
            grad_norms = []
            for _, module in self.ssl_model.student.items():
                grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=self.cfg.optim.clip_grad)
                grad_norms.append(float(grad_norm))
            if grad_norms:
                self.log('train_grad_norm', max(grad_norms), on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)

        optimizer.step()
        self.ssl_model.update_ema(mom_val)

        batch_size = self.cfg.train.batch_size_per_gpu
        self.log('train_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log('train/lr', lr_val, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        self.log('train/wd', wd_val, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        self.log('train/momentum', mom_val, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        self.log('train/teacher_temp', teacher_temp_val, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        self.log('ssl_lr_min', ssl_lr_min, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)
        self.log('ssl_lr_max', ssl_lr_max, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)

        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.detach()
                else:
                    value = value.detach().mean()
            self.log(f'train/{key}', value, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch_size)

        if self.online_probe is not None and probe_batch is not None:
            with torch.no_grad():
                features = self.ssl_model.student.backbone.forward_features(probe_batch['x'])['x_norm_clstoken']
            probe_loss, probas = self.online_probe.step(features, probe_batch['y'])
            self.online_probe.update('train', probas, probe_batch['y'])
            probe_opt = optimizers[1] if isinstance(optimizers, (list, tuple)) else None
            if probe_opt is not None:
                probe_opt.zero_grad(set_to_none=True)
                self.manual_backward(probe_loss)
                probe_opt.step()
                self.log(
                    'probe_lr',
                    float(probe_opt.param_groups[0]['lr']),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                    batch_size=probe_batch['y'].shape[0],
                )
            self.log('train_probe_loss', probe_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=probe_batch['y'].shape[0])

        if it < 3 and self.trainer.is_global_zero:
            probe_lr_msg = 'disabled'
            if isinstance(optimizers, (list, tuple)) and len(optimizers) > 1:
                probe_lr_msg = f"{float(optimizers[1].param_groups[0]['lr']):.3e}"
            self.print(
                "[DINOv3][Debug] "
                f"iter={it} ssl_lr_min={ssl_lr_min:.3e} ssl_lr_max={ssl_lr_max:.3e} "
                f"base_lr={lr_val:.3e} wd={wd_val:.3e} mom={mom_val:.6f} "
                f"teacher_temp={teacher_temp_val:.3e} probe_lr={probe_lr_msg}"
            )

        self.current_iteration += 1
        return total_loss

    def validation_step(self, batch, batch_idx):
        ssl_batch = batch['ssl'] if isinstance(batch, dict) and 'ssl' in batch else batch
        probe_batch = batch.get('probe') if isinstance(batch, dict) else None

        it = min(self.current_iteration, max(int(self.cfg.train.OFFICIAL_EPOCH_LENGTH) * max(int(self.cfg.optim.epochs), 1) - 1, 0))
        teacher_temp_val = self._schedule_value(self.teacher_temp_schedule, it, self.cfg.teacher.teacher_temp)

        ssl_batch['global_batch_size'] = self.cfg.train.batch_size_per_gpu * self.trainer.world_size
        with torch.no_grad():
            val_loss, loss_dict = self._compute_ssl_val_loss(ssl_batch, teacher_temp_val, it)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=self.cfg.train.batch_size_per_gpu)
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().mean() if value.numel() > 1 else value.detach()
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.cfg.train.batch_size_per_gpu)

        if self.online_probe is not None and probe_batch is not None:
            with torch.no_grad():
                features = self.ssl_model.student.backbone.forward_features(probe_batch['x'])['x_norm_clstoken']
            probe_loss, probas = self.online_probe.step(features, probe_batch['y'])
            self.online_probe.update('val', probas, probe_batch['y'])
            self.log('val_probe_loss', probe_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=probe_batch['y'].shape[0])
        return val_loss

    def on_train_epoch_start(self):
        datamodule = getattr(self.trainer, 'datamodule', None)
        if datamodule is not None and hasattr(datamodule, 'set_epoch'):
            datamodule.set_epoch(self.current_epoch)
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
        gc.collect()

    def on_validation_epoch_end(self):
        self._log_probe_metrics('val')

    def forward(self, x):
        return self.ssl_model.teacher.backbone(x, is_training=False)


def main(hparams):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(hparams.seed, workers=True)

    output_dir = build_output_dir(hparams.output_root, hparams.output_name)
    cfg = _build_dinov3_cfg(hparams, output_dir)
    precision_dtype = _precision_to_dino_dtype(hparams.precision)
    mean, std = _default_channel_stats(hparams.input_channels, hparams.normalization)

    data = EMBEDMammoDataModuleDINOv3(
        data_dir=embed_data_dir,
        csv_file=hparams.csv_file,
        image_size=tuple(hparams.image_size),
        test_percent=hparams.test_percent,
        val_percent=hparams.val_percent,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        mmap_path=hparams.mmap_path,
        input_channels=hparams.input_channels,
        global_crops_scale=tuple(hparams.global_crops_scale),
        local_crops_scale=tuple(hparams.local_crops_scale),
        local_crops_number=hparams.local_crops_number,
        local_crops_size=tuple(hparams.local_crops_size),
        patch_size=hparams.patch_size,
        ibot_mask_ratio_min_max=tuple(hparams.ibot_mask_ratio_min_max),
        mask_probability=hparams.mask_probability,
        mask_random_circular_shift=hparams.mask_random_circular_shift,
        precision_dtype=precision_dtype,
        horizontal_flip=hparams.horizontal_flip,
        share_color_jitter=hparams.share_color_jitter,
        random_color_jitter=hparams.random_color_jitter,
        color_jitter_prob=hparams.color_jitter_prob,
        mean=mean,
        std=std,
        online_probe_enabled=hparams.online_probe_enabled,
        tissue_aware_crop=hparams.tissue_aware_crop,
        tissue_aware_mask=hparams.tissue_aware_mask,
    )
    data.setup('fit')

    cfg.train.OFFICIAL_EPOCH_LENGTH = _resolve_official_epoch_length(
        data,
        world_size=hparams.num_devices,
    )

    model = DINOv3Lightning(
        cfg=cfg,
        init_mode=hparams.init_mode,
        init_ckpt=hparams.init_ckpt,
        num_classes=hparams.num_classes,
        online_probe_enabled=hparams.online_probe_enabled,
        online_probe_learning_rate=hparams.online_probe_learning_rate,
        online_probe_weight_decay=hparams.online_probe_weight_decay,
    )

    logger = build_logger(hparams.output_root, hparams.output_name)
    checkpoint_callback = ModelCheckpoint(
        monitor=hparams.checkpoint_metric,
        mode=hparams.checkpoint_mode,
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpoint_callback, TQDMProgressBar(refresh_rate=10)]
    if hparams.token_heatmap_enabled:
        callbacks.append(
            TokenHeatmapCallback(
                enabled=hparams.token_heatmap_enabled,
                every_n_epochs=hparams.token_heatmap_every_n_epochs,
                num_reference_images=hparams.token_heatmap_num_reference_images,
            )
        )
    strategy = build_strategy(hparams.num_devices, find_unused_parameters=True)
    trainer = build_trainer(
        max_epochs=hparams.epochs,
        num_devices=hparams.num_devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        precision=hparams.precision,
        sync_batchnorm=hparams.sync_batchnorm,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
    )
    trainer.fit(model=model, datamodule=data)

    if trainer.is_global_zero:
        best_state_dict = load_best_or_current_prefixed_state_dict(
            checkpoint_callback,
            'ssl_model.',
            model.ssl_model.state_dict(),
        )
        _save_backbone_exports(best_state_dict, output_dir)
        _save_backbone_exports(best_state_dict, output_dir, suffix='_best')
        del best_state_dict

        last_state_dict = load_last_or_current_prefixed_state_dict(
            checkpoint_callback,
            'ssl_model.',
            model.ssl_model.state_dict(),
        )
        _save_backbone_exports(last_state_dict, output_dir, suffix='_last')


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_learning_rate', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--model', type=str, default='vit_base', choices=['vit_small', 'vit_base', 'vit_large'])
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--normalization', type=str, default='imagenet', choices=['imagenet', 'half'])
    parser.add_argument('--drop_path_rate', type=float, default=0.3)
    parser.add_argument('--num_storage_tokens', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--image_size', type=int, nargs=2, default=list(image_size))
    parser.add_argument('--local_crops_size', type=int, nargs=2, default=list(local_image_size))
    parser.add_argument('--global_crops_scale', type=float, nargs=2, default=[0.40, 1.0])
    parser.add_argument('--local_crops_scale', type=float, nargs=2, default=[0.05, 0.20])
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--ibot_mask_ratio_min_max', type=float, nargs=2, default=[0.1, 0.5])
    parser.add_argument('--mask_probability', type=float, default=0.5)
    parser.add_argument('--mask_random_circular_shift', type=parse_bool, default=False)
    parser.add_argument('--head_n_prototypes', type=int, default=65536)
    parser.add_argument('--head_bottleneck_dim', type=int, default=256)
    parser.add_argument('--head_hidden_dim', type=int, default=2048)
    parser.add_argument('--head_nlayers', type=int, default=3)
    parser.add_argument('--teacher_temp', type=float, default=0.07)
    parser.add_argument('--warmup_teacher_temp', type=float, default=0.04)
    parser.add_argument('--momentum_teacher', type=float, default=0.992)
    parser.add_argument('--final_momentum_teacher', type=float, default=1.0)
    parser.add_argument('--koleo_loss_weight', type=float, default=0.1)
    parser.add_argument('--ibot_loss_weight', type=float, default=1.0)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--freeze_last_layer_epochs', type=int, default=1)
    parser.add_argument('--layerwise_decay', type=float, default=0.9)
    parser.add_argument('--patch_embed_lr_mult', type=float, default=0.2)
    parser.add_argument('--dino_head_wd_multiplier', type=float, default=1.0)
    parser.add_argument('--adamw_beta1', type=float, default=0.9)
    parser.add_argument('--adamw_beta2', type=float, default=0.999)
    parser.add_argument('--horizontal_flip', type=parse_bool, default=False)
    parser.add_argument('--share_color_jitter', type=parse_bool, default=False)
    parser.add_argument('--tissue_aware_crop', type=parse_bool, default=False)
    parser.add_argument('--tissue_aware_mask', type=parse_bool, default=False)
    parser.add_argument('--random_color_jitter', type=float, default=0.0)
    parser.add_argument('--color_jitter_prob', type=float, default=0.3)
    parser.add_argument('--compile', type=parse_bool, default=False)
    parser.add_argument('--cudagraphs', type=parse_bool, default=False)
    parser.add_argument('--checkpointing', type=parse_bool, default=False)
    parser.add_argument('--init_mode', type=str, default='random', choices=['random', 'ssl'])
    parser.add_argument('--init_ckpt', type=str, default=None)
    parser.add_argument('--csv_file', type=str, default=str(EMBED_CSV_PATH))
    parser.add_argument('--mmap_path', type=str, default=mmap_file_path)
    parser.add_argument('--output_root', type=str, default=str(OUTPUT_SSL_DIR))
    parser.add_argument('--output_name', type=str, default='dinov3_embed_run')
    parser.add_argument('--test_percent', type=float, default=0.2)
    parser.add_argument('--val_percent', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--sync_batchnorm', type=parse_bool, default=False)
    parser.add_argument('--checkpoint_metric', type=str, default='val_loss')
    parser.add_argument('--checkpoint_mode', type=str, default='min')
    parser.add_argument('--online_probe_enabled', type=parse_bool, default=False)
    parser.add_argument('--online_probe_learning_rate', type=float, default=1e-4)
    parser.add_argument('--online_probe_weight_decay', type=float, default=1e-4)
    parser.add_argument('--limit_train_batches', type=parse_limit_batches, default=None)
    parser.add_argument('--limit_val_batches', type=parse_limit_batches, default=None)
    parser.add_argument('--token_heatmap_enabled', type=parse_bool, default=False)
    parser.add_argument('--token_heatmap_every_n_epochs', type=int, default=4)
    parser.add_argument('--token_heatmap_num_reference_images', type=int, default=3)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
