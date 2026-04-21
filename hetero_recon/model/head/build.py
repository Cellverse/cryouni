from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch.nn as nn

from hetero_recon.model.utils.shape_specification import ShapeSpecification

__all__ = [
    "HEAD_REGISTRY",
    "build_head",
]

HEAD_REGISTRY = Registry("HEAD")
HEAD_REGISTRY.__doc__ = """
Registry for the head, which predict required output.
"""


def build_head(cfg: DictConfig, input_shape: ShapeSpecification) -> nn.Module:
    """
    Build the head defined by `cfg.MODEL.HEAD.NAME`.
    """

    head_name = cfg.MODEL.HEAD.NAME
    try:
        head = HEAD_REGISTRY.get(head_name)(cfg, input_shape)
    except KeyError as e:
        raise KeyError(HEAD_REGISTRY) from e

    return head
