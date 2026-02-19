"""Abstract base class for differentiable attack models."""

from abc import ABC, abstractmethod
from typing import Union, List

import torch
from torch import Tensor
from PIL import Image


class BaseAttack(ABC):
    """Interface for differentiable attack models used in DiffVax training."""

    @abstractmethod
    def attack(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        batch_size: int = 1,
    ) -> Tensor:
        """Run differentiable forward pass of the attack model.

        Args:
            prompt: Text prompt(s) for the attack.
            image: Input image tensor (adversarially perturbed).
            mask: Binary mask tensor.
            height: Output height.
            width: Output width.
            num_inference_steps: Number of diffusion steps.
            batch_size: Batch size.

        Returns:
            Generated image tensor with gradient flow from input image.
        """
        ...

    @abstractmethod
    def to_device(self, device: str):
        """Move model to specified device."""
        ...

    @abstractmethod
    def to_cpu(self):
        """Move model to CPU and free GPU memory."""
        ...

    @property
    @abstractmethod
    def loss_uses_mask_weighting(self) -> bool:
        """Whether loss should be weighted by mask (True) or computed over full image (False)."""
        ...
