from __future__ import annotations

import shlex

import hydra
from omegaconf import DictConfig, OmegaConf

from methods.base import get_method_module


def _bool_str(value: bool) -> str:
    return 'true' if value else 'false'


def _extend_arg(args: list[str], flag: str, value):
    if value is None:
        return

    if isinstance(value, bool):
        args.extend([flag, _bool_str(value)])
        return

    if isinstance(value, (str, bytes)):
        args.extend([flag, str(value)])
        return

    if hasattr(value, '__iter__') and not isinstance(value, dict):
        args.append(flag)
        args.extend(str(item) for item in value)
        return

    args.extend([flag, str(value)])


def _build_mae_cli_args(cfg: DictConfig) -> list[str]:
    output_name = cfg.output_name or cfg.experiment_name
    args: list[str] = []
    pairs = [
        ('--epochs', cfg.trainer.epochs),
        ('--batch_size', cfg.data.batch_size),
        ('--learning_rate', cfg.trainer.learning_rate),
        ('--weight_decay', cfg.trainer.weight_decay),
        ('--num_workers', cfg.data.num_workers),
        ('--num_devices', cfg.trainer.num_devices),
        ('--model', cfg.model.name),
        ('--init_mode', cfg.model.init_mode),
        ('--init_ckpt', cfg.model.init_ckpt),
        ('--init_ckpt_format', cfg.model.init_ckpt_format),
        ('--csv_file', cfg.data.csv_file),
        ('--mmap_path', cfg.data.mmap_path),
        ('--output_root', cfg.output_root),
        ('--output_name', output_name),
        ('--test_percent', cfg.data.test_percent),
        ('--val_percent', cfg.data.val_percent),
        ('--seed', cfg.seed),
        ('--precision', cfg.trainer.precision),
        ('--sync_batchnorm', cfg.trainer.sync_batchnorm),
        ('--accumulate_grad_batches', cfg.trainer.get('accumulate_grad_batches', 1)),
        ('--checkpoint_metric', cfg.trainer.checkpoint_metric),
        ('--checkpoint_mode', cfg.trainer.checkpoint_mode),
        ('--online_probe_enabled', cfg.trainer.online_probe.enabled),
        ('--online_probe_learning_rate', cfg.trainer.online_probe.learning_rate),
        ('--online_probe_weight_decay', cfg.trainer.online_probe.weight_decay),
        ('--limit_train_batches', cfg.trainer.get('limit_train_batches', None)),
        ('--limit_val_batches', cfg.trainer.get('limit_val_batches', None)),
        ('--mask_ratio', cfg.trainer.get('mask_ratio', 0.75)),
        ('--masking_mode', cfg.trainer.get('masking_mode', 'full_image')),
        ('--mae_crop_scale', cfg.data.augmentations.get('mae_crop_scale', [0.4, 1.0])),
        ('--mae_horizontal_flip', cfg.data.augmentations.get('mae_horizontal_flip', True)),
        ('--mae_color_jitter', cfg.data.augmentations.get('mae_color_jitter', 0.1)),
        ('--mae_color_jitter_prob', cfg.data.augmentations.get('mae_color_jitter_prob', 0.3)),
        ('--tissue_aware_crop', cfg.trainer.get('tissue_aware_crop', False)),
        ('--min_tissue_fraction', cfg.trainer.get('min_tissue_fraction', 0.3)),
        ('--image_mean', cfg.data.augmentations.get('image_mean', None)),
        ('--image_std', cfg.data.augmentations.get('image_std', None)),
    ]
    for flag, value in pairs:
        _extend_arg(args, flag, value)
    return args


def _build_dinov2_cli_args(cfg: DictConfig) -> list[str]:
    output_name = cfg.output_name or cfg.experiment_name
    args: list[str] = []
    pairs = [
        ('--epochs', cfg.trainer.epochs),
        ('--batch_size', cfg.data.batch_size),
        ('--learning_rate', cfg.trainer.learning_rate),
        ('--min_learning_rate', cfg.trainer.get('min_learning_rate', 1.0e-6)),
        ('--weight_decay', cfg.trainer.weight_decay),
        ('--warmup_epochs', cfg.trainer.warmup_epochs),
        ('--warmup_teacher_temp_epochs', cfg.trainer.get('warmup_teacher_temp_epochs', 30)),
        ('--num_workers', cfg.data.num_workers),
        ('--num_devices', cfg.trainer.num_devices),
        ('--model', cfg.model.name),
        ('--patch_size', cfg.model.patch_size),
        ('--num_register_tokens', cfg.model.num_register_tokens),
        ('--input_channels', cfg.data.input_channels),
        ('--normalization', cfg.trainer.get('normalization', 'imagenet')),
        ('--drop_path_rate', cfg.trainer.get('drop_path_rate', 0.3)),
        ('--image_size', cfg.data.image_size),
        ('--local_crops_size', cfg.trainer.crops.local_crops_size),
        ('--global_crops_scale', cfg.trainer.crops.global_crops_scale),
        ('--local_crops_scale', cfg.trainer.crops.local_crops_scale),
        ('--local_crops_number', cfg.trainer.crops.local_crops_number),
        ('--ibot_mask_ratio_min_max', cfg.trainer.ibot_mask_ratio_min_max),
        ('--mask_probability', cfg.trainer.mask_probability),
        ('--head_n_prototypes', cfg.trainer.head_n_prototypes),
        ('--head_bottleneck_dim', cfg.trainer.head_bottleneck_dim),
        ('--head_hidden_dim', cfg.trainer.head_hidden_dim),
        ('--head_nlayers', cfg.trainer.head_nlayers),
        ('--teacher_temp', cfg.trainer.teacher_temp),
        ('--warmup_teacher_temp', cfg.trainer.warmup_teacher_temp),
        ('--momentum_teacher', cfg.trainer.get('momentum_teacher', 0.992)),
        ('--final_momentum_teacher', cfg.trainer.get('final_momentum_teacher', 1.0)),
        ('--centering', cfg.trainer.get('centering', 'centering')),
        ('--wd_start', cfg.trainer.wd_start),
        ('--wd_end', cfg.trainer.wd_end),
        ('--use_lr_scheduler', cfg.trainer.use_lr_scheduler),
        ('--koleo_loss_weight', cfg.trainer.koleo_loss_weight),
        ('--ibot_loss_weight', cfg.trainer.ibot_loss_weight),
        ('--ibot_separate_head', cfg.trainer.get('ibot_separate_head', False)),
        ('--clip_grad', cfg.trainer.get('clip_grad', 3.0)),
        ('--freeze_last_layer_epochs', cfg.trainer.get('freeze_last_layer_epochs', 1)),
        ('--layerwise_decay', cfg.trainer.get('layerwise_decay', 0.9)),
        ('--patch_embed_lr_mult', cfg.trainer.get('patch_embed_lr_mult', 0.2)),
        ('--adamw_beta1', cfg.trainer.get('adamw_beta1', 0.9)),
        ('--adamw_beta2', cfg.trainer.get('adamw_beta2', 0.999)),
        ('--horizontal_flip', cfg.data.augmentations.horizontal_flip),
        ('--tissue_aware_crop', cfg.trainer.get('tissue_aware_crop', False)),
        ('--tissue_aware_mask', cfg.trainer.get('tissue_aware_mask', False)),
        ('--random_color_jitter', cfg.data.augmentations.random_color_jitter),
        ('--color_jitter_prob', cfg.data.augmentations.color_jitter_prob),
        ('--num_classes', cfg.data.num_classes),
        ('--init_mode', cfg.model.init_mode),
        ('--init_ckpt', cfg.model.init_ckpt),
        ('--csv_file', cfg.data.csv_file),
        ('--mmap_path', cfg.data.mmap_path),
        ('--output_root', cfg.output_root),
        ('--output_name', output_name),
        ('--test_percent', cfg.data.test_percent),
        ('--val_percent', cfg.data.val_percent),
        ('--seed', cfg.seed),
        ('--precision', cfg.trainer.precision),
        ('--sync_batchnorm', cfg.trainer.sync_batchnorm),
        ('--checkpoint_metric', cfg.trainer.checkpoint_metric),
        ('--checkpoint_mode', cfg.trainer.checkpoint_mode),
        ('--online_probe_enabled', cfg.trainer.online_probe.enabled),
        ('--online_probe_learning_rate', cfg.trainer.online_probe.learning_rate),
        ('--online_probe_weight_decay', cfg.trainer.online_probe.weight_decay),
        ('--limit_train_batches', cfg.trainer.get('limit_train_batches', None)),
        ('--limit_val_batches', cfg.trainer.get('limit_val_batches', None)),
    ]
    for flag, value in pairs:
        _extend_arg(args, flag, value)
    return args


def _build_dinov3_cli_args(cfg: DictConfig) -> list[str]:
    output_name = cfg.output_name or cfg.experiment_name
    args: list[str] = []
    pairs = [
        ('--epochs', cfg.trainer.epochs),
        ('--batch_size', cfg.data.batch_size),
        ('--learning_rate', cfg.trainer.learning_rate),
        ('--min_learning_rate', cfg.trainer.min_learning_rate),
        ('--weight_decay', cfg.trainer.weight_decay),
        ('--weight_decay_end', cfg.trainer.weight_decay_end),
        ('--warmup_epochs', cfg.trainer.warmup_epochs),
        ('--warmup_teacher_temp_epochs', cfg.trainer.warmup_teacher_temp_epochs),
        ('--num_workers', cfg.data.num_workers),
        ('--num_devices', cfg.trainer.num_devices),
        ('--model', cfg.model.name),
        ('--patch_size', cfg.model.patch_size),
        ('--input_channels', cfg.data.input_channels),
        ('--normalization', cfg.trainer.get('normalization', 'imagenet')),
        ('--drop_path_rate', cfg.trainer.get('drop_path_rate', 0.3)),
        ('--num_storage_tokens', cfg.model.num_storage_tokens),
        ('--image_size', cfg.data.image_size),
        ('--local_crops_size', cfg.trainer.crops.local_crops_size),
        ('--global_crops_scale', cfg.trainer.crops.global_crops_scale),
        ('--local_crops_scale', cfg.trainer.crops.local_crops_scale),
        ('--local_crops_number', cfg.trainer.crops.local_crops_number),
        ('--ibot_mask_ratio_min_max', cfg.trainer.ibot_mask_ratio_min_max),
        ('--mask_probability', cfg.trainer.mask_probability),
        ('--mask_random_circular_shift', cfg.trainer.mask_random_circular_shift),
        ('--head_n_prototypes', cfg.trainer.head_n_prototypes),
        ('--head_bottleneck_dim', cfg.trainer.head_bottleneck_dim),
        ('--head_hidden_dim', cfg.trainer.head_hidden_dim),
        ('--head_nlayers', cfg.trainer.head_nlayers),
        ('--teacher_temp', cfg.trainer.teacher_temp),
        ('--warmup_teacher_temp', cfg.trainer.warmup_teacher_temp),
        ('--momentum_teacher', cfg.trainer.momentum_teacher),
        ('--final_momentum_teacher', cfg.trainer.final_momentum_teacher),
        ('--koleo_loss_weight', cfg.trainer.koleo_loss_weight),
        ('--ibot_loss_weight', cfg.trainer.ibot_loss_weight),
        ('--clip_grad', cfg.trainer.clip_grad),
        ('--freeze_last_layer_epochs', cfg.trainer.freeze_last_layer_epochs),
        ('--layerwise_decay', cfg.trainer.layerwise_decay),
        ('--patch_embed_lr_mult', cfg.trainer.patch_embed_lr_mult),
        ('--dino_head_wd_multiplier', cfg.trainer.dino_head_wd_multiplier),
        ('--adamw_beta1', cfg.trainer.adamw_beta1),
        ('--adamw_beta2', cfg.trainer.adamw_beta2),
        ('--horizontal_flip', cfg.data.augmentations.horizontal_flip),
        ('--share_color_jitter', cfg.trainer.share_color_jitter),
        ('--tissue_aware_crop', cfg.trainer.get('tissue_aware_crop', False)),
        ('--tissue_aware_mask', cfg.trainer.get('tissue_aware_mask', False)),
        ('--random_color_jitter', cfg.data.augmentations.random_color_jitter),
        ('--color_jitter_prob', cfg.data.augmentations.color_jitter_prob),
        ('--compile', cfg.trainer.compile),
        ('--cudagraphs', cfg.trainer.cudagraphs),
        ('--checkpointing', cfg.trainer.checkpointing),
        ('--init_mode', cfg.model.init_mode),
        ('--init_ckpt', cfg.model.init_ckpt),
        ('--csv_file', cfg.data.csv_file),
        ('--mmap_path', cfg.data.mmap_path),
        ('--output_root', cfg.output_root),
        ('--output_name', output_name),
        ('--test_percent', cfg.data.test_percent),
        ('--val_percent', cfg.data.val_percent),
        ('--seed', cfg.seed),
        ('--precision', cfg.trainer.precision),
        ('--sync_batchnorm', cfg.trainer.sync_batchnorm),
        ('--checkpoint_metric', cfg.trainer.checkpoint_metric),
        ('--checkpoint_mode', cfg.trainer.checkpoint_mode),
        ('--online_probe_enabled', cfg.trainer.online_probe.enabled),
        ('--online_probe_learning_rate', cfg.trainer.online_probe.learning_rate),
        ('--online_probe_weight_decay', cfg.trainer.online_probe.weight_decay),
        ('--limit_train_batches', cfg.trainer.get('limit_train_batches', None)),
        ('--limit_val_batches', cfg.trainer.get('limit_val_batches', None)),
        ('--token_heatmap_enabled', cfg.trainer.token_heatmap.enabled),
        ('--token_heatmap_every_n_epochs', cfg.trainer.token_heatmap.every_n_epochs),
        ('--token_heatmap_num_reference_images', cfg.trainer.token_heatmap.num_reference_images),
    ]
    for flag, value in pairs:
        _extend_arg(args, flag, value)
    return args


def _build_cli_args(cfg: DictConfig) -> list[str]:
    if cfg.method_name == 'dinov2':
        return _build_dinov2_cli_args(cfg)
    if cfg.method_name == 'dinov3':
        return _build_dinov3_cli_args(cfg)
    if cfg.method_name == 'mae':
        return _build_mae_cli_args(cfg)
    raise ValueError(f"Unsupported method_name: {cfg.method_name}")


@hydra.main(version_base=None, config_path='configs', config_name='config')
def hydra_main(cfg: DictConfig) -> None:
    if cfg.method_name is None:
        raise ValueError('Please choose an experiment config, for example: python pre-training/main.py experiment=dinov2_embed')

    OmegaConf.resolve(cfg)
    print('================ EFFECTIVE CONFIG ================')
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print('==================================================')

    module = get_method_module(cfg.method_name)
    if not hasattr(module, 'build_parser') or not hasattr(module, 'main'):
        raise AttributeError(f"Method '{cfg.method_name}' must expose build_parser() and main(hparams).")

    cli_args = _build_cli_args(cfg)
    print('Dispatching to method:')
    print(f"  {cfg.method_name} {' '.join(shlex.quote(arg) for arg in cli_args)}")

    parser = module.build_parser()
    args = parser.parse_args(cli_args)
    module.main(args)


if __name__ == '__main__':
    hydra_main()
