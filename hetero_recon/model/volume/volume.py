from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class Volume(nn.Module, metaclass=ABCMeta):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, rotations: torch.Tensor, conformations: torch.Tensor, radius: int = None) -> torch.Tensor:
        """
        Args:
            `rotations` (torch.Tensor): Rotation matrices of shape [B, 3, 3].
            `conformations` (torch.Tensor): Conformations of shape [B, C].
            `radius` (int): Radius of the Hartley domain mask.

        Returns:
            `images_ht` (torch.Tensor): Hartley domain images of shape [B, D, D].
        """
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def eval_volume(self, conformations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `conformations` (torch.Tensor): Conformations of shape [B, C].

        Returns:
            `volume_sp` (torch.Tensor): Spatial domain volume of shape [D, D, D].
        """
        raise NotImplementedError
