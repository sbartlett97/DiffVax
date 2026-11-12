"""Perceptual imperceptibility loss: LPIPS(original, protected).

The soft L1 on the perturbation (loss2) does not track what people notice:
the v1 Klein run reached its full 16/255 budget as a structured stripe texture
with LPIPS(orig, protected) = 0.35 and SSIM 0.77, plainly visible. LPIPS is the
same quantity the evaluation reports, and it is differentiable, so it can be
minimised directly.
"""
import torch
from torch import Tensor

from diffvax.utils import resolve_device


class LPIPSLoss:
    """Mean LPIPS between the clean and protected image (lower = less visible).

    Config ``lpips_loss``: ``net`` (alex|vgg|squeeze, default alex). The weight is
    applied by LossComposer. Inputs are in [-1, 1], which is what lpips expects.
    """

    def __init__(self, config: dict):
        import lpips  # pip install lpips

        cfg = config.get("lpips_loss", {})
        self._device = resolve_device()
        self.net = lpips.LPIPS(net=cfg.get("net", "alex"), verbose=False).to(self._device).eval()
        self.net.requires_grad_(False)

    def __call__(self, img_orig: Tensor, img_adv: Tensor, **kwargs) -> Tensor:
        a = img_orig.detach().float().to(self._device)
        b = img_adv.float().to(self._device)
        return self.net(a, b).mean()
