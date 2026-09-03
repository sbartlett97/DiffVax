"""Abstract base class for differentiable attack models."""

import warnings
from abc import ABC, abstractmethod
from typing import Union, List, Optional

import torch
from torch import Tensor
from PIL import Image


def suppress_full_backward_hook_kwarg_warning() -> None:
    """Silence PyTorch's benign "no inputs require gradients" backward-hook
    warning for callers that register register_full_backward_(pre_)hook on
    modules invoked with all-keyword arguments (e.g. H4 TGR hooks on real
    diffusers transformer blocks, called as
    ``block(hidden_states=..., encoder_hidden_states=..., temb=...)``).

    PyTorch's full-backward-hook input-tracking cannot see a gradient-
    requiring tensor passed purely as a keyword argument (no positional
    args at all) and defensively assumes none of the module's inputs
    require grad — even though the module's output correctly does, and the
    hook receives the correct grad_output regardless. Verified empirically:
    the hook's captured grad_output matches the analytically expected value
    and downstream .grad values are correct in both cases; see
    tests/test_attack_gradient_flow.py (A6) for the gradient-correctness
    regression tests this warning has no bearing on.
    """
    warnings.filterwarnings(
        "ignore",
        message="Full backward hook is firing when gradients are computed "
        "with respect to module outputs",
        category=UserWarning,
    )


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

    @property
    def supports_masked_attack(self) -> bool:
        """Whether this model can additionally run a mask-conditioned attack
        while remaining fundamentally a full-image model (distinct from
        is_inpainting, which means masked-ONLY, no full-image capability).

        Default returns False. Override to True in subclasses that implement
        optional mask-conditioned blending (e.g. SD3Attack's RePaint mode).
        """
        return False
