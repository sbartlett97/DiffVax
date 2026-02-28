"""Flat-minima regularization for cross-model transfer (Phase 6).

TDAE (arXiv:2512.14341) showed that adversarial perturbations occupying flat
regions of the loss landscape transfer better across models. Sharp minima
exploit one model's specific quirks; flat minima exploit broadly shared
vulnerabilities. This module penalizes sharp loss landscapes via two methods:

  - 'grad_norm': gradient-norm penalty as a cheap sharpness proxy.
    Minimizing ||∇L||² encourages the optimizer toward flat regions.
  - 'sam': SAM-style (Sharpness-Aware Minimization) sharpness estimate.
    Computes the worst-case loss increase under a weight perturbation of
    radius rho (more expensive but theoretically principled).
"""

import torch
from torch import Tensor


class FlatMinimaRegularizer:
    """Sharpness penalty that discourages narrow loss-landscape valleys.

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

    def compute(self, loss: Tensor, model: torch.nn.Module) -> Tensor:
        """Compute the flat-minima regularization term.

        Args:
            loss:  The current base loss (scalar tensor, in the graph).
            model: The NestedUNet whose parameters define the landscape.

        Returns:
            Scalar regularization term. Multiply by lambda_flat before
            adding to the total loss.
        """
        if self.method == "sam":
            return self._sam(loss, model)
        return self._grad_norm(loss, model)

    def _grad_norm(self, loss: Tensor, model: torch.nn.Module) -> Tensor:
        """Gradient-norm penalty: loss += lambda * ||∇L||².

        Penalizes large gradients, which correlate with sharp loss landscapes,
        without second-order computation.
        """
        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            create_graph=True,
            allow_unused=True,
        )
        grad_norm_sq = sum(
            g.norm() ** 2 for g in grads if g is not None
        )
        return grad_norm_sq

    def _sam(self, loss: Tensor, model: torch.nn.Module) -> Tensor:
        """SAM-style sharpness estimate via gradient-norm proxy for rho-ball.

        Returns rho * ||∇L|| as an approximation of the worst-case loss
        increase under a weight perturbation of radius rho.
        """
        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            create_graph=True,
            allow_unused=True,
        )
        grad_norm = torch.sqrt(
            sum(g.norm() ** 2 for g in grads if g is not None)
        )
        return self.rho * grad_norm
