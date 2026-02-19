"""GPU-swapping model manager for multi-model training."""

import random
import gc
from typing import Dict, Tuple

import torch

from diffvax.attack_base import BaseAttack


class AttackModelManager:
    """Manages multiple attack models, swapping them on/off GPU by probability.

    All models start on CPU. select_and_load() picks one by configured probability,
    offloads the current model to CPU, and loads the selected one to GPU.
    """

    def __init__(
        self,
        models: Dict[str, BaseAttack],
        probabilities: Dict[str, float],
    ):
        self.models = models
        self.probabilities = probabilities
        self._current_name: str | None = None

        # Validate probabilities sum to ~1.0
        total = sum(probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Model probabilities must sum to 1.0, got {total}: {probabilities}"
            )

        # Validate same keys
        if set(models.keys()) != set(probabilities.keys()):
            raise ValueError(
                f"Model keys {set(models.keys())} don't match "
                f"probability keys {set(probabilities.keys())}"
            )

    def select_and_load(self) -> Tuple[str, BaseAttack]:
        """Randomly select a model and load it to GPU.

        Returns:
            Tuple of (model_name, attack_model) with the model on GPU.
        """
        names = list(self.probabilities.keys())
        weights = [self.probabilities[n] for n in names]
        selected_name = random.choices(names, weights=weights, k=1)[0]

        if selected_name == self._current_name:
            return selected_name, self.models[selected_name]

        # Offload current model to CPU
        if self._current_name is not None:
            self.models[self._current_name].to_cpu()

        # Load selected model to GPU
        self.models[selected_name].to_device("cuda")
        torch.cuda.empty_cache()
        gc.collect()

        self._current_name = selected_name
        return selected_name, self.models[selected_name]

    def offload_all(self):
        """Move all models to CPU."""
        for name, model in self.models.items():
            model.to_cpu()
        self._current_name = None
        torch.cuda.empty_cache()
        gc.collect()
