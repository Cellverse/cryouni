from fvcore.common.registry import Registry
from omegaconf import DictConfig

from .backbone import Backbone

__all__ = [
    "BACKBONE_REGISTRY",
    "build_backbone",
]

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for the backbone, which extract features from the input images.
"""


def build_backbone(cfg: DictConfig) -> Backbone:
    """
    Build the backbone defined by `cfg.MODEL.BACKBONE.NAME`.
    """

    backbone_name = cfg.MODEL.BACKBONE.NAME
    try:
        backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    except KeyError as e:
        raise KeyError(BACKBONE_REGISTRY) from e

    return backbone
