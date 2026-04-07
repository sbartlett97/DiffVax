"""Multi-model attack wrapper for DiffVax training.

Randomly routes each training batch to one of several attack models according
to user-specified probabilities. This forces the immunization network to learn
perturbations that transfer across architectures (SD 1.5 UNet, FLUX DiT, SD 3.5
MM-DiT) rather than overfitting to a single model family.

Usage (matches attack.py interface)::

    from diffvax.attack_multi import MultiAttack
    attack = MultiAttack(
        models=[
            {"type": "sd15",  "link": "runwayml/stable-diffusion-inpainting",  "prob": 0.2},
            {"type": "flux",  "link": "black-forest-labs/FLUX.2-klein-4B",     "prob": 0.6},
            {"type": "sd3",   "link": "stabilityai/stable-diffusion-3.5-large-inpainting", "prob": 0.2},
        ]
    )
    edited = attack.attack(prompt, masked_image, mask, height=512, width=512)
"""

import random
import torch
from typing import Union, List


class MultiAttack:
    """Routes each forward call to a randomly-sampled attack model.

    Args:
        models: list of dicts with keys:
            - type: "sd15" | "flux" | "sd3"
            - link: HuggingFace repo ID
            - prob: sampling probability (will be normalised)
        seed: RNG seed for reproducibility (None = non-deterministic).
    """

    def __init__(self, models: list, seed: int = None):
        if not models:
            raise ValueError("At least one model spec must be provided.")

        total_prob = sum(m["prob"] for m in models)
        self._specs = [dict(m, prob=m["prob"] / total_prob) for m in models]
        self._loaded: dict = {}  # type -> attack instance (lazy load)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self, spec: dict):
        """Lazily load and cache an attack model instance."""
        key = spec["link"]
        if key not in self._loaded:
            t = spec["type"]
            if t == "sd15":
                from diffvax.attack import Attack
                self._loaded[key] = Attack(spec["link"])
            elif t == "flux":
                from diffvax.attack_flux import FluxAttack
                # Default to 0.0 — FluxAttack auto-detects distilled vs non-distilled.
                # Explicit override via spec["guidance_scale"] takes precedence.
                guidance = spec.get("guidance_scale", 0.0)
                self._loaded[key] = FluxAttack(spec["link"], guidance_scale=guidance)
            elif t == "sd3":
                from diffvax.attack_sd3 import SD3Attack
                guidance = spec.get("guidance_scale", 7.0)
                self._loaded[key] = SD3Attack(spec["link"], guidance_scale=guidance)
            else:
                raise ValueError(f"Unknown attack model type: {t!r}")
        return self._loaded[key]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_model(self):
        """Sample a model spec according to specified probabilities."""
        r = self._rng.random()
        cumulative = 0.0
        for spec in self._specs:
            cumulative += spec["prob"]
            if r < cumulative:
                return spec
        return self._specs[-1]

    # ------------------------------------------------------------------
    # Attack interface
    # ------------------------------------------------------------------

    def attack(
        self,
        prompt: Union[str, List[str]],
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        batch_size: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Route one batch to a randomly-selected attack model.

        Args and returns match the attack.py interface for drop-in use
        inside DiffVaxImmunization.
        """
        spec = self._sample_model()
        model = self._load(spec)

        t = spec["type"]
        if t == "sd15":
            return model.attack(
                prompt=prompt,
                masked_image=masked_image,
                mask=mask,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                batch_size=batch_size,
            )
        else:
            # FLUX / SD3 share the same extended signature
            strength = kwargs.get("strength", 0.85)
            return model.attack(
                prompt=prompt,
                masked_image=masked_image,
                mask=mask,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                strength=strength,
                batch_size=batch_size,
            )

    @property
    def model(self):
        """Return the first-loaded attack model (for baseline compat)."""
        if self._loaded:
            return next(iter(self._loaded.values())).pipe
        self._load(self._specs[0])
        return next(iter(self._loaded.values())).pipe

    def tokenize_prompt(self, *args, **kwargs):
        """Delegate tokenization to first SD15 model (baseline compat)."""
        for spec in self._specs:
            if spec["type"] == "sd15":
                return self._load(spec).tokenize_prompt(*args, **kwargs)
        raise RuntimeError("No SD 1.5 model loaded — tokenize_prompt not available.")
