from dataclasses import dataclass


@dataclass
class ShapeSpecification:
    """
    Help for shape inference among layers.
    """

    in_channels: int | None = None
    out_channels: int | None = None
    stride: int | None = None
