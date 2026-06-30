from .common import BackboneBundle, LayerFeatures


def load_backbone_bundle(*args, **kwargs):
    from .registry import load_backbone_bundle as _load_backbone_bundle

    return _load_backbone_bundle(*args, **kwargs)

__all__ = ["BackboneBundle", "LayerFeatures", "load_backbone_bundle"]
