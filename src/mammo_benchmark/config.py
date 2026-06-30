from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

# Per-backbone input normalization. Foundation models expect inputs in the exact distribution
# they were pretrained on — downstream images (raw [0,1]) must be renormalized to match.
# CLIP-family wrappers (BioMedCLIP/UniMedCLIP/MedSigLIP) normalize internally inside their
# forward(), so we leave pretrain_norm=None for those to avoid double-normalizing.
IMAGENET_NORM = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
RAD_DINO_NORM = {"mean": [0.5307, 0.5307, 0.5307], "std": [0.2583, 0.2583, 0.2583]}
# [0.5]/[0.5] normalization for checkpoints that were trained with this input convention.
HALF_NORM = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
# Native Mammo-CLIP/Mammo-FM image normalization recorded in the checkpoint config.
MAMMOCLIP_NORM = {
    "mean": [0.3089279, 0.3089279, 0.3089279],
    "std": [0.25053555408335154, 0.25053555408335154, 0.25053555408335154],
}


DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "BMCD": {
        "image_dataset": "BMCD",
        "label_mappings": {
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
            "Bi-Rads": {"Bi-Rads 1": 0, "Bi-Rads 2": 1, "Bi-Rads 4": 2, "Bi-Rads 5": 3},
        },
    },
    "CBIS-DDSM-breast": {
        "image_dataset": "CBIS-DDSM",
        "label_mappings": {
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
            "Bi-Rads": {
                "Bi-Rads 1": 0,
                "Bi-Rads 2": 1,
                "Bi-Rads 3": 2,
                "Bi-Rads 4": 3,
                "Bi-Rads 5": 4,
            },
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "CDD-CESM": {
        "image_dataset": "CDD-CESM",
        "label_mappings": {
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
            "Bi-Rads": {
                "Bi-Rads 1": 0,
                "Bi-Rads 2": 1,
                "Bi-Rads 3": 2,
                "Bi-Rads 4": 3,
                "Bi-Rads 5": 4,
            },
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "CMMD": {
        "image_dataset": "CMMD",
        "label_mappings": {
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "DBT": {
        "image_dataset": "DBT",
        "label_mappings": {
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "DMID": {
        "image_dataset": "DMID",
        "label_mappings": {
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
            "Bi-Rads": {
                "Bi-Rads 1": 0,
                "Bi-Rads 2": 1,
                "Bi-Rads 3": 2,
                "Bi-Rads 4": 3,
                "Bi-Rads 5": 4,
            },
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "INbreast": {
        "image_dataset": "INbreast",
        "label_mappings": {
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
            "Bi-Rads": {
                "Bi-Rads 1": 0,
                "Bi-Rads 2": 1,
                "Bi-Rads 3": 2,
                "Bi-Rads 4": 3,
                "Bi-Rads 5": 4,
            },
        },
    },
    "KAU-BCMD": {
        "image_dataset": "KAU-BCMD",
        "label_mappings": {
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
            "Bi-Rads": {"Bi-Rads 1": 0, "Bi-Rads 3": 1, "Bi-Rads 4": 2, "Bi-Rads 5": 3},
        },
    },
    "MIAS": {
        "image_dataset": "MIAS",
        "label_mappings": {
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "MM": {
        "image_dataset": "MM",
        "label_mappings": {
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "NLBS": {
        "image_dataset": "NLBS",
        "label_mappings": {
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "RSNA": {
        "image_dataset": "RSNA",
        "label_mappings": {
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
        },
    },
    "RSNA-site2": {
        "image_dataset": "RSNA",
        "metadata_filters": {"site_id": ["2"]},
        "label_mappings": {
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "EMBED": {
        "image_dataset": "EMBED",
        "label_mappings": {
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
            "Bi-Rads": {
                "Bi-Rads 1": 0,
                "Bi-Rads 2": 1,
                "Bi-Rads 3": 2,
                "Bi-Rads 4": 3,
                "Bi-Rads 5": 4,
            },
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "OPTIMAM": {
        "image_dataset": "OPTIMAM",
        "label_mappings": {
            "Composition": {"Level A": 0, "Level B": 1, "Level C": 2, "Level D": 3},
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
    "VinDr-Mammo-breast": {
        "image_dataset": "VinDr-Mammo",
        "label_mappings": {
            "Composition": {
                "Level A": 0,
                "Level B": 1,
                "Level C": 2,
                "Level D": 3,
            },
            "Bi-Rads": {
                "Bi-Rads 1": 0,
                "Bi-Rads 2": 1,
                "Bi-Rads 3": 2,
                "Bi-Rads 4": 3,
                "Bi-Rads 5": 4,
            },
            "CancerStatus": {"Non-cancer": 0, "Cancer": 1},
        },
    },
}


MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "dinov2_vitb": {
        "display_name": "DINOv2 (ViT-B)",
        "weight_file": "dinov2/dinov2_vitb14_reg4_pretrain.pth",
        "default_input_size": (1008, 756),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": IMAGENET_NORM,
    },
    "dinov3_vitb": {
        "display_name": "DINOv3 (ViT-B)",
        "weight_file": "dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "default_input_size": (1024, 768),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": IMAGENET_NORM,
    },
    "biomedclip_vitb_1024x768": {
        "display_name": "BioMedCLIP (ViT-B/16, 1024x768)",
        "weight_file": "biomedclip",
        "default_input_size": (1024, 768),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": None,
    },
    "unimedclip_vitl14_base_1008x756_interp": {
        "display_name": "UniMedCLIP (ViT-L/14, 1008x756 interp)",
        "weight_file": "unimedclip/unimed_clip_vit_l14_base_text_encoder.pt",
        "default_input_size": (1008, 756),
        "feature_layer_ids": (4, 11, 17, 23),
        "pretrain_norm": None,
    },
    "medsiglip_vit_1008x756_interp": {
        "display_name": "MedSigLIP (ViT-So400M/14, 1008x756 interp)",
        "weight_file": "medsiglip",
        "default_input_size": (1008, 756),
        "feature_layer_ids": (6, 13, 19, 26),
        "pretrain_norm": None,
    },
    "rad_dino_vitb": {
        "display_name": "RAD-DINO (ViT-B)",
        "weight_file": "rad_dino/backbone_compatible.safetensors",
        "default_input_size": (1008, 756),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": RAD_DINO_NORM,
    },
    "xray_dino_vitl": {
        "display_name": "RayDINO / X-ray DINO (ViT-L)",
        "weight_file": "xray_dino/xray_dino_vitl16_pretrained-ad31c2b0.pth",
        "default_input_size": (1024, 768),
        "feature_layer_ids": (4, 11, 17, 23),
        "pretrain_norm": IMAGENET_NORM,
    },
    "dinov2_embed_vitb_in": {
        "display_name": "DINOv2-EMBED (ViT-B)",
        "weight_file": "dinov2_embed_continued_in_e60_imagenet_norm_b32/dinov2_student_backbone_last.pth",
        "checkpoint_group": "adapted",
        "default_input_size": (1008, 756),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": IMAGENET_NORM,
    },
    "dinov3_embed_vitb_in": {
        "display_name": "DINOv3-EMBED (ViT-B)",
        "weight_file": "dinov3_embed_continued_preddp20_scaled_e60_dinov3_vit_base_continued_512x384_b28x4_gpu4_lr2e-05_e60_ssl/dinov3_student_backbone_last.pth",
        "checkpoint_group": "adapted",
        "default_input_size": (1024, 768),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": IMAGENET_NORM,
    },
    "mammo_mae_vitb_in": {
        "display_name": "MAE-EMBED (ViT-B)",
        "weight_file": "mae_embed_in_e500_mr020_inffirst_tissue_wdfix_bicubic/mae_custom_pretrained_model.pth",
        "checkpoint_group": "adapted",
        "default_input_size": (1024, 768),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": IMAGENET_NORM,
    },
    "mama_vitb": {
        "display_name": "MaMA (ViT-B)",
        "weight_file": "mama/MAMA (ViT-B).ckpt",
        "default_input_size": (1008, 756),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": HALF_NORM,
    },
    "glam_vitb": {
        "display_name": "GLAM (ViT-B)",
        "weight_file": "glam/glam_official_2024_12_13_vitb14_reg_backbone.pth",
        "default_input_size": (1008, 756),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": HALF_NORM,
    },
    "mammo_clip_effnet_b5": {
        "display_name": "Mammo-CLIP (EfficientNet-B5)",
        "weight_file": "mammo-clip/Mammo-CLIP (Enb5).tar",
        "default_input_size": (1024, 768),
        "feature_layer_ids": (-1,),
        "pretrain_norm": MAMMOCLIP_NORM,
        "backbone_family": "mammo_clip_cnn",
    },
    "mammo_fm_batman_effnet_b5": {
        "display_name": "Mammo-FM Batmanlab CLIP (EfficientNet-B5)",
        "weight_file": "mammo-fm/Mammo-FM_BatmanlabTrained_CLIP.tar",
        "default_input_size": (1024, 768),
        "feature_layer_ids": (-1,),
        "pretrain_norm": MAMMOCLIP_NORM,
        "backbone_family": "mammo_clip_cnn",
    },
    "versamm_vitb": {
        "display_name": "VersaMammo stage-1 (ViT-B)",
        "weight_file": "versamm/VersaMammo (ViT-B).pth",
        "default_input_size": (1008, 756),
        "feature_layer_ids": (2, 5, 8, 11),
        "pretrain_norm": IMAGENET_NORM,
    },
}


def resolve_model_specs(
    checkpoint_root: str | Path,
    adapted_checkpoint_root: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Return model specs with absolute checkpoint paths.

    Public/third-party backbones are resolved under ``checkpoint_root``. The
    three EMBED-adapted checkpoints are resolved under
    ``adapted_checkpoint_root`` when provided, otherwise under
    ``checkpoint_root``.
    """
    checkpoint_root = Path(checkpoint_root).expanduser()
    adapted_root = Path(adapted_checkpoint_root).expanduser() if adapted_checkpoint_root else checkpoint_root
    specs = deepcopy(MODEL_SPECS)
    for spec in specs.values():
        root = adapted_root if spec.get("checkpoint_group") == "adapted" else checkpoint_root
        spec["weight_file"] = str(root / spec["weight_file"])
    return specs
