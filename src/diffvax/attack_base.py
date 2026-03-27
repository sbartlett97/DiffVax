"""Abstract base class for differentiable attack models."""

from abc import ABC, abstractmethod
from typing import Union, List, Optional

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
        mask: Optional[Union[torch.FloatTensor, Image.Image]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        batch_size: int = 1,
        strength: float = 1.0,
    ) -> Tensor:
        """Run differentiable forward pass of the attack model.

        Args:
            prompt: Text prompt(s) for the attack.
            image: Input image tensor (adversarially perturbed).
            mask: Binary mask tensor. Required for inpainting models; None
                for img2img / full-image models which do not use a mask.
            height: Output height.
            width: Output width.
            num_inference_steps: Number of diffusion steps.
            batch_size: Batch size.
            strength: Denoising strength (1.0 = full noise, <1.0 = partial).

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

    @property
    def vae_channels(self) -> int:
        """Number of latent channels (4 for SD1.x, 16 for SD3/FLUX).

        Override in subclasses that use a non-standard VAE.
        Default returns 4 for backward compatibility with SD 1.5 subclasses.
        """
        return 4

    @property
    def native_resolution(self) -> int:
        """Preferred training resolution for this model (width == height).

        The training loop uses this to decide whether to differentiably
        resize the input before passing it to the attack forward pass.
        Default returns 512 for backward compatibility with SD 1.5 subclasses.
        """
        return 512

    def get_vae(self):
        """Return the VAE encoder module for latent-space loss computation.

        Returns None by default.  Subclasses that expose a VAE should override
        to return the diffusers ``AutoencoderKL`` (or equivalent) module so
        that the training loop can compute a latent-space disruption loss without
        having to inspect the concrete attack class.

        The returned module must be on the same device as the attack model
        and must not require gradients (frozen weights).
        """
        return None

    @property
    def is_inpainting(self) -> bool:
        """Whether this model is a mask-conditioned inpainting model.

        Inpainting models (e.g. SD 1.5 inpaint) expect the masked image and
        the raw mask as separate inputs, and the loss is weighted by the mask
        region. Full-image img2img models (SD3, FLUX) receive no mask.

        Default returns False. Override to True in inpainting subclasses.
        """
        return False
