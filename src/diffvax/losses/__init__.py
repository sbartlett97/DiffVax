"""Loss composition for DiffVax v2.

Provides a LossComposer that aggregates optional CLIP, flat-minima, and
attention disruption loss terms based on configuration flags. Every term
is gated by an 'enabled' flag so that disabling all new features restores
the original single-term (loss1 + loss2) behaviour.
"""

import torch
from torch import Tensor
from typing import Dict, Tuple, Any


class LossComposer:
    """Accumulates and computes optional loss terms with configured weights.

    Constructed from the full training config dict. Only terms with
    'enabled: true' in their respective config sections are activated.

    Usage in the training loop:
        extra_loss, breakdown = loss_composer.compute(
            img_orig=img_batch,
            img_adv=img_adv,
            img_out=img_out,
            prompts=cur_prompt,
        )
        total_loss = loss1 + loss2 + extra_loss

    Args:
        config: Full training config dict (same object passed to the
                DiffVaxImmunization constructor).
    """

    def __init__(self, config: dict):
        self._terms: Dict[str, Tuple[Any, float]] = {}

        # Phase 2: CLIP disruption loss
        if config.get("clip_loss", {}).get("enabled", False):
            from diffvax.losses.clip_loss import CLIPDisruptionLoss

            beta = float(config.get("beta", 0.5))
            self._terms["clip"] = (CLIPDisruptionLoss(config), beta)

    def has_terms(self) -> bool:
        """Return True if any optional loss terms are active."""
        return bool(self._terms)

    def compute(
        self,
        img_orig: Tensor,
        img_adv: Tensor,
        img_out: Tensor,
        prompts,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute all active loss terms.

        Args:
            img_orig: Original clean image (B, 3, H, W), [-1, 1].
            img_adv:  Adversarial image, same shape.
            img_out:  Attack model output, same shape.
            prompts:  Edit prompts (list of strings).

        Returns:
            (total_extra_loss, breakdown_dict) where breakdown_dict maps
            term names to their unweighted scalar values for logging.
        """
        total = torch.tensor(0.0, device="cuda")
        breakdown: Dict[str, float] = {}

        for name, (loss_fn, weight) in self._terms.items():
            val = loss_fn(
                img_orig=img_orig,
                img_adv=img_adv,
                img_out=img_out,
                prompts=prompts,
            )
            total = total + weight * val
            breakdown[name] = val.item()

        return total, breakdown
