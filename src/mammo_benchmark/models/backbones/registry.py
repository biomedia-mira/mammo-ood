import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from .biomedclip import BioMedClipVisionBackbone, build_biomedclip_visual_model
from .common import BackboneBundle, flatten_nested_block_keys, load_state_dict_with_report, torch_load_compat
from .dino_backbones import build_dinov2_model, build_dinov3_model, build_xray_dino_model
from .generic_vit import DinoBackbone, MAEBackbone
from .mammo_cnn import build_mammo_clip_cnn_backbone, build_versamammo_effnet_backbone
from .medsiglip import MedSiglipVisionBackbone
from .unimedclip import UniMedClipVisionBackbone, build_unimedclip_visual_model


PACKAGE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PACKAGE_DIR.parent
PRETRAINING_DIR = PACKAGE_DIR / "pretraining"
for path in (SRC_DIR, PRETRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_backbone_bundle(
    *,
    model_key: str,
    model_spec: Dict[str, Any],
    input_size: Tuple[int, int],
) -> BackboneBundle:
    checkpoint_path_obj = Path(model_spec["weight_file"])
    if not checkpoint_path_obj.is_absolute():
        raise ValueError(f"weight_file must be an absolute path for {model_key}: {checkpoint_path_obj}")
    checkpoint_path = str(checkpoint_path_obj)
    layer_ids = tuple(int(layer_id) for layer_id in model_spec["feature_layer_ids"])

    if model_spec.get("backbone_family") == "mammo_clip_cnn":
        if layer_ids != (-1,):
            raise ValueError(f"Mammo-CLIP CNN backbones support only feature_layer_ids=(-1,), got {layer_ids}")
        backbone = build_mammo_clip_cnn_backbone(checkpoint_path, input_size=input_size)

    elif model_spec.get("backbone_family") == "versamammo_effnet_cnn":
        if layer_ids != (-1,):
            raise ValueError(f"VersaMammo EfficientNet supports only feature_layer_ids=(-1,), got {layer_ids}")
        backbone = build_versamammo_effnet_backbone(checkpoint_path, input_size=input_size)

    elif model_key == "mama_vitb":
        from MaMA.load_weight import load_model

        mama_model = load_model(pretrained_model_path=checkpoint_path)
        backbone = DinoBackbone(mama_model.img_encoder_q.model)

    elif model_key == "glam_vitb":
        # The configured GLAM checkpoint is an exported DINOv2 ViT-B/14-reg
        # backbone state dict, so load it directly and avoid depending on the
        # copied GLAM package.
        model = build_dinov2_model("dinov2_vitb14_reg", img_size=518)
        checkpoint = torch_load_compat(checkpoint_path)
        state_dict = checkpoint.get("state_dict", checkpoint)
        load_state_dict_with_report(model, state_dict, context="GLAM EMBED ViT-B backbone", strict=False)
        backbone = DinoBackbone(model)

    elif model_key in {"mammo_mae_vitb", "mammo_mae_vitb_in"}:
        from architectures.vit import mae_vit_base_patch16

        model = mae_vit_base_patch16(img_size=input_size, norm_pix_loss=True)
        checkpoint = torch_load_compat(checkpoint_path)
        # pos_embed and decoder_pos_embed are fixed sin-cos embeddings (not learned).
        # The model already initialised correct sin-cos for the new grid size,
        # so we drop the old ones from the checkpoint and load the rest.
        state_dict = checkpoint.get("model", checkpoint)
        if "state_dict" in checkpoint:
            prefixed = {
                key.replace("model.", "", 1): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("model.")
            }
            if prefixed:
                state_dict = prefixed
        state_dict = dict(state_dict)
        state_dict.pop("pos_embed", None)
        state_dict.pop("decoder_pos_embed", None)
        load_state_dict_with_report(
            model, state_dict, context=f"Mammo-MAE ViT-B img_size={input_size}", strict=False,
        )
        backbone = MAEBackbone(model)

    elif model_key == "versamm_vitb":
        model = build_dinov2_model("dinov2_vitb14", img_size=224)
        checkpoint = torch_load_compat(checkpoint_path)
        state_dict = flatten_nested_block_keys(checkpoint["teacher"], prefix="backbone.")
        load_state_dict_with_report(model, state_dict, context="VersaMammo ViT-B", strict=False)
        backbone = DinoBackbone(model)

    elif model_key == "dinov2_vitb":
        model = build_dinov2_model("dinov2_vitb14_reg")
        checkpoint = torch_load_compat(checkpoint_path)
        load_state_dict_with_report(model, checkpoint, context="DINOv2 ViT-B", strict=False)
        backbone = DinoBackbone(model)

    elif model_key in {"dinov2_embed_vitb", "dinov2_embed_vitb_in"}:
        # Build with the source pretraining size so learned non-square pos_embed
        # loads cleanly; the model will interpolate at runtime for downstream inputs.
        model = build_dinov2_model("dinov2_vitb14_reg", img_size=(504, 378))
        checkpoint = torch_load_compat(checkpoint_path)
        load_state_dict_with_report(model, checkpoint, context="DINOv2 EMBED ViT-B", strict=False)
        backbone = DinoBackbone(model)

    elif model_key == "dinov3_vitb":
        model = build_dinov3_model("vitb")
        checkpoint = torch_load_compat(checkpoint_path)
        load_state_dict_with_report(model, checkpoint, context="DINOv3 ViT-B", strict=False)
        backbone = DinoBackbone(model)

    elif model_key in {"dinov3_embed_vitb", "dinov3_embed_vitb_in"}:
        model = build_dinov3_model("vitb")
        checkpoint = torch_load_compat(checkpoint_path)
        load_state_dict_with_report(model, checkpoint, context="DINOv3 EMBED ViT-B", strict=False)
        backbone = DinoBackbone(model)

    elif model_key == "dinov3_vitl":
        model = build_dinov3_model("vitl")
        checkpoint = torch_load_compat(checkpoint_path)
        load_state_dict_with_report(model, checkpoint, context="DINOv3 ViT-L", strict=False)
        backbone = DinoBackbone(model)

    elif model_key == "rad_dino_vitb":
        model = build_dinov2_model("dinov2_vitb14")
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
        load_state_dict_with_report(model, state_dict, context="RAD-DINO ViT-B", strict=False)
        backbone = DinoBackbone(model)

    elif model_key == "xray_dino_vitl":
        # Build with img_size matching the checkpoint (512); the model's
        # interpolate_pos_encoding() handles larger inputs at inference.
        model = build_xray_dino_model(img_size=512)
        checkpoint = torch_load_compat(checkpoint_path)
        state_dict = flatten_nested_block_keys(checkpoint)
        # X-ray DINO uses chunked blocks (blocks.0.LAYER.*) in the DINOv2 model
        chunked_dict = {
            (f"blocks.0.{k[len('blocks.'):]}" if k.startswith("blocks.") else k): v
            for k, v in state_dict.items()
        }
        load_state_dict_with_report(model, chunked_dict, context="X-ray DINO ViT-L", strict=False)
        backbone = DinoBackbone(model)

    elif model_key == "medsiglip_vit_1008x756_interp":
        from transformers import SiglipVisionModel

        model, loading_info = SiglipVisionModel.from_pretrained(
            checkpoint_path,
            local_files_only=True,
            output_loading_info=True,
        )
        unexpected = loading_info.get("unexpected_keys", [])
        if unexpected:
            print(f"[Backbone] MedSigLIP ignored non-vision checkpoint keys: {len(unexpected)}")
        backbone = MedSiglipVisionBackbone(
            model.vision_model,
            input_size=input_size,
            interpolate_pos_encoding=True,
        )

    elif model_key == "biomedclip_vitb_1024x768":
        model, image_mean, image_std = build_biomedclip_visual_model(checkpoint_path, input_size=input_size)
        backbone = BioMedClipVisionBackbone(model, image_mean=image_mean, image_std=image_std)

    elif model_key == "unimedclip_vitl14_base_1008x756_interp":
        model, image_mean, image_std = build_unimedclip_visual_model(checkpoint_path, input_size=input_size)
        backbone = UniMedClipVisionBackbone(model, image_mean=image_mean, image_std=image_std)

    else:
        raise ValueError(f"Unsupported model_key={model_key}")

    native_feature_dim = int(getattr(backbone, "native_feature_dim", backbone.token_dim))
    print(
        f"[Backbone] Loaded model={model_key} token_dim={backbone.token_dim} layers={layer_ids} "
        f"supports_cls={backbone.supports_cls_token} native_feature_dim={native_feature_dim}"
    )
    return BackboneBundle(
        backbone=backbone,
        token_dim=backbone.token_dim,
        layer_ids=layer_ids,
        supports_cls_token=backbone.supports_cls_token,
        native_feature_dim=native_feature_dim,
    )
