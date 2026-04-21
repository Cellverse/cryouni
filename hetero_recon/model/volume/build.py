from fvcore.common.registry import Registry
from omegaconf import DictConfig

from .volume import Volume

__all__ = [
    "VOLUME_REGISTRY",
    "build_volume",
]

VOLUME_REGISTRY = Registry("VOLUME")
VOLUME_REGISTRY.__doc__ = "Registry for the volume."


def build_volume(cfg: DictConfig) -> Volume:
    volume_name = getattr(cfg, "NAME", cfg.MODEL.VOLUME.NAME)
    try:
        volume = VOLUME_REGISTRY.get(volume_name)(cfg)
    except KeyError as e:
        raise KeyError(VOLUME_REGISTRY) from e

    return volume
