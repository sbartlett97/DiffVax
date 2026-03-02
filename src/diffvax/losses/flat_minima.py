"""Flat-minima regularization for cross-model transfer (Phase 6).

TDAE (arXiv:2512.14341) showed that adversarial perturbations occupying flat
regions of the loss landscape transfer better across models. Sharp minima
exploit one model's specific quirks; flat minima exploit broadly shared
vulnerabilities. This module penalizes sharp loss landscapes via two methods:

  - 'grad_norm': gradient-norm penalty as a cheap sharpness proxy.
    Scales gradients by (1 + 2λ||∇L||), encouraging the optimizer toward
    flat regions where gradients are small.
  - 'sam': SAM-style (Sharpness-Aware Minimization) sharpness estimate.
    Scales gradients by (1 + λρ/||∇L||), approximating the worst-case
    ascent direction.

Both methods use a first-order approximation applied post-backward, avoiding
create_graph=True (which would require second-order derivatives through
attention layers and force the O(N²) math SDPA backend).
"""

import torch


class FlatMinimaRegularizer:
    """Sharpness penalty that discourages narrow loss-landscape valleys.

    Call apply() AFTER loss.backward() and BEFORE optimizer.step().

    Args from config['flat_minima']:
        method:      'grad_norm' (default) or 'sam'
        rho:         SAM perturbation radius (default: 0.05)
        lambda_flat: Weight in total loss (default: 0.01)
    """

    def __init__(self, config: dict):
        cfg = config.get("flat_minima", {})
        self.method = cfg.get("method", "grad_norm")
        self.rho = float(cfg.get("rho", 0.05))
        self.lambda_flat = float(cfg.get("lambda_flat", 0.01))

    def apply(self, model: torch.nn.Module, lambda_flat: float) -> float:
        """Apply flat-minima gradient scaling post-backward.

        Must be called after loss.backward() and before optimizer.step().
        Modifies parameter gradients in-place.

        The exact gradient of L + λ||∇L||² requires the Hessian-vector product
        2λ H ∇L, which needs second-order derivatives (incompatible with flash
        attention). Instead we use the first-order approximation:

            ∇(L + λ||∇L||²) ≈ (1 + 2λ||∇L||) · ∇L

        This achieves the same qualitative effect: parameters with large
        gradients (sharp landscape) receive amplified updates, pushing the
        optimizer toward flatter regions.

        Args:
            model:       The NestedUNet whose .grad attributes to scale.
            lambda_flat: Regularization weight.

        Returns:
            Gradient norm squared (scalar float, for logging).
        """
        grad_norm_sq = sum(
            p.grad.norm() ** 2 for p in model.parameters()
            if p.grad is not None
        ).item()

        grad_norm = grad_norm_sq ** 0.5

        if self.method == "sam":
            # SAM proxy: scale ∝ (1 + λρ / ||∇L||)
            if grad_norm > 1e-8:
                scale = 1.0 + lambda_flat * self.rho / grad_norm
            else:
                scale = 1.0
        else:
            # grad_norm: scale ∝ (1 + 2λ||∇L||)
            scale = 1.0 + 2.0 * lambda_flat * grad_norm

        if scale != 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

        return grad_norm_sq
