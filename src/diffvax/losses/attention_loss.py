"""Cross-attention disruption loss for DiT models (Phase 7).

DeContext (arXiv:2512.16625) confirmed that standard adversarial attacks fail
on DiT models because they ignore how contextual information propagates through
dual-stream attention. MM-DiT uses joint bidirectional attention where text and
image tokens interact directly. Disrupting cross-attention alignment causes
edits to produce semantically incoherent outputs regardless of the edit prompt.

This module hooks into transformer blocks during the attack forward pass and
maximizes attention entropy — making the model unable to focus on semantically
relevant image regions for any text prompt.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Union


class AttentionDisruptionLoss:
    """Maximizes attention entropy in DiT transformer blocks.

    Hooks into the attention layers of an MM-DiT or similar transformer,
    captures attention-related activations during the forward pass, and
    computes a penalty that encourages uniform (high-entropy) attention
    distributions, thereby disrupting the model's ability to focus on
    semantically relevant regions.

    Args from config['attention_loss']:
        target_blocks: 'early' (default), 'late', or list of block indices.
        num_hooks:     Number of blocks to hook (default: 4).
        weight:        Loss weight in total loss (default: 0.3).
        only_with_dit: If True, only activate for DiT models (default: True).
    """

    def __init__(self, config: dict):
        cfg = config.get("attention_loss", {})
        self.weight = float(cfg.get("weight", 0.3))
        self.num_hooks = int(cfg.get("num_hooks", 4))
        self.target_blocks = cfg.get("target_blocks", "early")
        self.only_with_dit = bool(cfg.get("only_with_dit", True))
        self._hooks: List = []
        self._attention_maps: List[Tensor] = []

    def _should_hook(self, block_idx: int, total_blocks: int) -> bool:
        """Decide whether to hook block at index block_idx."""
        if self.target_blocks == "early":
            return block_idx < self.num_hooks
        elif self.target_blocks == "late":
            return block_idx >= (total_blocks - self.num_hooks)
        elif isinstance(self.target_blocks, list):
            return block_idx in self.target_blocks
        # fallback: hook first num_hooks blocks
        return block_idx < self.num_hooks

    def _make_hook(self):
        """Factory for forward hook that captures attention outputs."""
        def hook_fn(module, input, output):
            # output may be a tuple (attn_out, weights) or just attn_out tensor
            if isinstance(output, (tuple, list)):
                attn_out = output[0]
            else:
                attn_out = output
            if isinstance(attn_out, Tensor):
                self._attention_maps.append(attn_out)
        return hook_fn

    def register_hooks(self, transformer) -> None:
        """Register forward hooks on attention layers of the transformer.

        Tries transformer_blocks, single_transformer_blocks, and blocks
        attribute names to support SD3, FLUX, and other DiT variants.
        Hooks the 'attn' or 'attention' sub-module of each selected block.

        Args:
            transformer: The DiT transformer module from the attack pipeline.
        """
        self.remove_hooks()

        blocks = None
        for attr in ["transformer_blocks", "single_transformer_blocks", "blocks"]:
            blocks = getattr(transformer, attr, None)
            if blocks is not None:
                break

        if blocks is None:
            return

        total = len(blocks)
        for i, block in enumerate(blocks):
            if not self._should_hook(i, total):
                continue
            # Try to find the attention sub-module
            attn = (
                getattr(block, "attn", None)
                or getattr(block, "attention", None)
                or getattr(block, "self_attn", None)
            )
            if attn is not None:
                h = attn.register_forward_hook(self._make_hook())
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered hooks and clear captured maps."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._attention_maps = []

    def compute(self) -> Tensor:
        """Compute attention entropy loss from captured activation maps.

        Higher entropy = more uniform attention = disrupted semantic focus.
        We negate because we minimize the loss and want to maximize entropy.

        Returns:
            Scalar loss tensor (negated mean entropy across hooked blocks).
            Returns 0.0 if no maps were captured (e.g., no hooks registered).
        """
        if not self._attention_maps:
            return torch.tensor(0.0, device="cuda")

        total_entropy = torch.tensor(0.0, device="cuda")
        count = 0

        for attn_out in self._attention_maps:
            # attn_out shape: (B, seq_len, dim)
            # Use token-level activation magnitude distribution as proxy
            # for attention focus: high variance = focused, low = diffuse.
            acts = attn_out.abs().float()  # (B, seq_len, dim)
            # Aggregate across feature dim -> token importance scores
            token_scores = acts.mean(dim=-1)  # (B, seq_len)
            probs = F.softmax(token_scores, dim=-1)  # (B, seq_len)
            # Entropy: H = -sum(p * log(p))
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            total_entropy = total_entropy + entropy
            count += 1

        # Clear for next iteration
        self._attention_maps = []

        if count == 0:
            return torch.tensor(0.0, device="cuda")

        # Negate: we minimize loss, so -entropy achieves entropy maximization
        return -(total_entropy / count)
