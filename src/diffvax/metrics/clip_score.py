"""Clip Score Metric"""

import open_clip
import torch

from .base import Metric
from diffvax.utils import resolve_device


class ClipScore(Metric):
    """CLIP score metric for evaluating the quality of an image."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        # CUDA > MPS > CPU. Previously hard-coded to run on CPU tensors while
        # wrapped in a CUDA-only autocast context — harmless on recent torch
        # (autocast silently disables itself without a CUDA context) but
        # fragile, and left the model on CPU unconditionally even when a GPU
        # was available.
        self._device = resolve_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            kwargs["model"], pretrained=kwargs["pretrained_on"])
        self.model = self.model.to(self._device).eval()
        self.tokenizer = open_clip.get_tokenizer(kwargs["model"])

    def __call__(self, edited_images, prompts):
        clip_scores = []
        for img, prompt in zip(edited_images, prompts):
            clip_score = self.calculate_clip_score(img, prompt)
            clip_scores.append(clip_score)
        return clip_scores

    def calculate_clip_score(self, img, prompt):
        """Calculate the CLIP score between an image and a prompt."""
        image = self.preprocess(img).unsqueeze(0).to(self._device)
        text = self.tokenizer([prompt]).to(self._device)

        with torch.no_grad(), torch.autocast(
            self._device.type, enabled=self._device.type != "cpu"
        ):
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            clip_score = (100 * image_features @ text_features.T).mean()
        return clip_score.item()
