"""Probability-weighted model selector for multi-model training (Phase 5: adaptive).

Phase 5 adds gradient-aware adaptive ensemble weighting (AdaEA-inspired,
arXiv:2308.02897). The core idea: if two surrogates produce highly correlated
gradients, one is redundant and should be downweighted. If their gradients are
orthogonal, both contribute unique signal and should be upweighted.

Adaptive weighting is fully opt-in: passing `adaptive=False` (or omitting the
adaptive_ensemble config section) restores the original static random.choices
behaviour exactly.
"""

import random
from typing import Dict, Tuple, Optional

import torch
from torch import Tensor

from diffvax.attack_base import BaseAttack


class AttackModelManager:
    """Manages multiple attack models, selecting between them by probability.

    All models are loaded to GPU at construction time and stay there.
    With adaptive=True the selection probabilities are updated every
    `update_period` iterations based on gradient disparity between models.

    Args:
        models:       Dict mapping name → BaseAttack instance.
        probabilities:Dict mapping name → float (must sum to 1.0).
        adaptive:     Enable gradient-aware adaptive weighting (Phase 5).
        adaptive_cfg: Config dict for adaptive ensemble parameters:
                        update_period (int, default 50)
                        min_weight    (float, default 0.1)
                        smoothing     (float, default 0.9)  EMA coefficient
    """

    def __init__(
        self,
        models: Dict[str, BaseAttack],
        probabilities: Dict[str, float],
        adaptive: bool = False,
        adaptive_cfg: Optional[Dict] = None,
    ):
        self.models = models
        self.probabilities = dict(probabilities)  # mutable copy
        self.adaptive = adaptive
        adaptive_cfg = adaptive_cfg or {}
        self.update_period: int = int(adaptive_cfg.get("update_period", 50))
        self.min_weight: float = float(adaptive_cfg.get("min_weight", 0.1))
        self.smoothing: float = float(adaptive_cfg.get("smoothing", 0.9))

        # Validate probabilities
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

        # Adaptive weighting state
        # gradient_history: name → latest flattened gradient vector (detached)
        self.gradient_history: Dict[str, Optional[Tensor]] = {
            name: None for name in models
        }

    def select_and_load(self) -> Tuple[str, BaseAttack]:
        """Randomly select a model by the current (possibly adaptive) weights.

        Returns:
            Tuple of (model_name, attack_model).
        """
        names = list(self.probabilities.keys())
        weights = [self.probabilities[n] for n in names]
        selected_name = random.choices(names, weights=weights, k=1)[0]
        return selected_name, self.models[selected_name]

    # ------------------------------------------------------------------
    # Adaptive ensemble (Phase 5)
    # ------------------------------------------------------------------

    def record_gradient(self, model_name: str, grad_vec: Tensor) -> None:
        """Store the flattened NestedUNet gradient produced when model_name was active.

        Call this after each backward pass while adaptive=True.

        Args:
            model_name: Name of the model used for this backward step.
            grad_vec:   Flattened gradient tensor (1-D), already detached.
        """
        if not self.adaptive:
            return
        if model_name not in self.gradient_history:
            return
        # Use EMA to smooth gradient history
        prev = self.gradient_history[model_name]
        if prev is None or prev.shape != grad_vec.shape:
            self.gradient_history[model_name] = grad_vec.detach().clone()
        else:
            self.gradient_history[model_name] = (
                self.smoothing * prev + (1.0 - self.smoothing) * grad_vec.detach()
            )

    def update_weights(self) -> None:
        """Recompute model selection probabilities based on gradient disparity.

        Models whose gradients are more dissimilar to other models' gradients
        receive higher weight (they contribute unique signal). Models with
        high pairwise similarity are downweighted (redundant).

        Uses the AdaEA disparity-based principle: weight ∝ mean gradient
        disparity to all other models. After normalization, applies a
        floor of min_weight to prevent starvation.
        """
        if not self.adaptive:
            return

        names = [n for n in self.models if self.gradient_history.get(n) is not None]
        if len(names) < 2:
            return  # Not enough history yet

        grads = {n: self.gradient_history[n] for n in names}

        # Compute pairwise cosine disparity via batched matrix multiply — O(N*D)
        # instead of O(N^2) sequential cosine_similarity calls.
        grad_matrix = torch.stack([grads[n] for n in names], dim=0)  # (N, D)
        norms = grad_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
        grad_normed = grad_matrix / norms  # unit vectors (N, D)
        cos_sim_matrix = torch.mm(grad_normed, grad_normed.t())  # (N, N)
        n = len(names)
        # Diagonal is self-similarity = 1.0; sum off-diagonal per row
        mean_cos = (cos_sim_matrix.sum(dim=1) - 1.0) / max(n - 1, 1)
        mean_disp = (1.0 - mean_cos).tolist()  # disparity = 1 - mean_cos_sim
        disparity: Dict[str, float] = dict(zip(names, mean_disp))

        # Higher disparity → more unique gradient → higher weight
        total_disparity = sum(disparity.values())
        if total_disparity < 1e-8:
            return  # All gradients identical — keep current weights

        raw_weights = {n: disparity[n] / total_disparity for n in names}

        # Apply min_weight floor and renormalize
        floored = {n: max(raw_weights[n], self.min_weight) for n in names}
        # Also include models with no gradient history yet (keep their weight)
        for n in self.models:
            if n not in floored:
                floored[n] = self.probabilities[n]

        total = sum(floored.values())
        self.probabilities = {n: floored[n] / total for n in floored}
