from mammo_benchmark.config import MODEL_SPECS, resolve_model_specs


def test_release_exposes_only_paper_models():
    assert len(MODEL_SPECS) == 15
    assert "dinov3_vitl" not in MODEL_SPECS
    assert "mammo_clip_effnet_b2" not in MODEL_SPECS
    assert "mammo_fm_asu_effnet_b5" not in MODEL_SPECS


def test_adapted_checkpoints_resolve_under_adapted_root():
    specs = resolve_model_specs("/ckpt", "/adapted")
    assert specs["dinov3_vitb"]["weight_file"].startswith("/ckpt/")
    assert specs["dinov3_embed_vitb_in"]["weight_file"].startswith("/adapted/")

