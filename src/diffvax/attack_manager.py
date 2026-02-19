"""Probability-weighted model selector for multi-model training."""

import random
from typing import Dict, Tuple

from diffvax.attack_base import BaseAttack


class AttackModelManager:
    """Manages multiple attack models, selecting between them by probability.

    All models are loaded to GPU at construction time and stay there.
    """

    def __init__(
        self,
        models: Dict[str, BaseAttack],
        probabilities: Dict[str, float],
    ):
        self.models = models
        self.probabilities = probabilities

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

        # Load all models to GPU upfront
        for model in models.values():
            model.to_device("cuda")

    def select_and_load(self) -> Tuple[str, BaseAttack]:
        """Randomly select a model by configured probability.

        Returns:
            Tuple of (model_name, attack_model).
        """
        names = list(self.probabilities.keys())
        weights = [self.probabilities[n] for n in names]
        selected_name = random.choices(names, weights=weights, k=1)[0]
        return selected_name, self.models[selected_name]
