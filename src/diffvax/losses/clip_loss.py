"""CLIP-based architecture-agnostic disruption loss (Phase 2).

SITA (IEEE TIFS 2025) demonstrated that CLIP-based losses disrupt style
representations without any diffusion model in the loop, with strong
cross-architecture transferability. Since virtually all modern generative
models share a CLIP-like vision-language backbone, this loss attacks the
one representation layer that is genuinely architecture-agnostic.

Two terms:
  - Feature disruption: maximize cosine distance between CLIP features of
    the original image and the adversarial image.
  - Semantic disruption: minimize CLIP similarity between the edit output
    and the edit prompt (the model fails to follow the instruction).
"""

from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor


class CLIPDisruptionLoss:
    """CLIP feature disruption loss for architecture-agnostic immunization.

    Loads an OpenCLIP model (ViT-B/32 by default) once at construction;
    all its parameters are frozen. Handles [-1, 1] input convention
    internally by converting to [0, 1] before CLIP preprocessing.

    Args from config['clip_loss']:
        model:          OpenCLIP model name (default: 'ViT-B/32')
        pretrained:     OpenCLIP weight set (default: 'laion2b_s34b_b79k')
        feature_weight: Weight on feature disruption term (default: 1.0)
        semantic_weight:Weight on semantic disruption term (default: 0.5)
    """

    def __init__(self, config: dict):
        cfg = config.get("clip_loss", {})
        self.feature_weight = float(cfg.get("feature_weight", 1.0))
        self.semantic_weight = float(cfg.get("semantic_weight", 0.5))
        model_name = cfg.get("model", "ViT-B/32")
        pretrained = cfg.get("pretrained", "laion2b_s34b_b79k")

        import open_clip  # pip install open-clip-torch

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.cuda().half()
        self.model.requires_grad_(False)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # CLIP normalization constants (ImageNet stats, shared across CLIP variants)
        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device="cuda"
        ).view(1, 3, 1, 1).half()
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device="cuda"
        ).view(1, 3, 1, 1).half()

    def _preprocess(self, x: Tensor) -> Tensor:
        """Convert [-1, 1] tensor to CLIP-normalized [0, 1] 224×224 input."""
        x_01 = (x.half() + 1.0) / 2.0
        x_resized = F.interpolate(
            x_01, (224, 224), mode="bicubic", align_corners=False
        )
        return (x_resized - self.mean) / self.std

    def encode_image(self, x: Tensor) -> Tensor:
        """Encode image tensor ([-1, 1]) into CLIP feature space."""
        return self.model.encode_image(self._preprocess(x))

    def encode_text(self, prompts: Union[str, List[str]]) -> Tensor:
        """Encode text prompts into CLIP feature space."""
        if isinstance(prompts, (tuple, list)):
            prompt_list = list(prompts)
        else:
            prompt_list = [prompts]
        tokens = self.tokenizer(prompt_list).cuda()
        return self.model.encode_text(tokens)

    def forward(
        self,
        img_orig: Tensor,
        img_adv: Tensor,
        img_out: Tensor,
        prompts: Union[str, List[str]],
    ) -> Tensor:
        """Compute CLIP disruption loss.

        Args:
            img_orig: Original clean image, shape (B, 3, H, W), in [-1, 1].
            img_adv:  Adversarially perturbed image, same shape.
            img_out:  Edited output from attack model, same shape.
            prompts:  Edit prompts (list of strings, length B).

        Returns:
            Scalar loss tensor. Minimizing this loss maximizes feature
            distance (disrupting the representation) and minimizes
            prompt–output alignment (disrupting the semantic edit).
        """
        # Feature disruption: minimize cosine similarity orig <-> adv
        # (we want the adversarial image to look different in CLIP space)
        feat_orig = self.encode_image(img_orig).detach()
        feat_adv = self.encode_image(img_adv)
        loss_feat = F.cosine_similarity(feat_orig, feat_adv, dim=-1).mean()

        # Semantic disruption: minimize prompt–output alignment
        feat_out = self.encode_image(img_out)
        feat_text = self.encode_text(prompts).detach()
        loss_sem = F.cosine_similarity(feat_out, feat_text, dim=-1).mean()

        return self.feature_weight * loss_feat + self.semantic_weight * loss_sem

    def __call__(
        self,
        img_orig: Tensor,
        img_adv: Tensor,
        img_out: Tensor,
        prompts: Union[str, List[str]],
    ) -> Tensor:
        return self.forward(img_orig, img_adv, img_out, prompts)
