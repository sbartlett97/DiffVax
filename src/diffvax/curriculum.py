"""Multi-resolution training curriculum for DiffVax v2 (Phase 4).

The current pipeline trains at 512×512 only. For higher-resolution images
the naive approach of bilinearly downsampling to 512 loses high-frequency
perturbation detail in the gradient signal. SDXL (1024), SD3 (1024), and
FLUX (variable) all operate natively at higher resolutions.

This module schedules a sequence of training resolutions over the course of
training, starting low (low VRAM, fast iteration) and progressively growing
to the target resolution. The NestedUNet is fully convolutional and
resolution-agnostic, so no architecture changes are needed — only the input
resolution and DataLoader batch size need adjustment at each stage boundary.

All resolutions are rounded to the nearest multiple of 16 (required by the
4-level NestedUNet max-pooling chain) to avoid padding artefacts.
"""

import math
import random
from typing import Dict, List, Optional


class ResolutionCurriculum:
    """Schedules training resolutions as a function of training iteration.

    Stages are checked in order; the first stage whose ``until_epoch`` is
    greater than the current iteration index is used.  Iterations beyond all
    stage boundaries fall back to the last stage.

    Args from config['curriculum']:
        enabled:           Toggle (default: False — use config resolution).
        stages:            List of dicts with keys:
                             resolution  (int) – target resolution for stage
                             until_epoch (int) – exclusive upper iteration bound
                             batch_size  (int) – suggested batch size
        resolution_jitter: Float in [0, 1].  If > 0, applies a random
                           ± jitter percentage to the target resolution
                           each call. The result is always rounded to the
                           nearest multiple of 16.
    """

    _DEFAULT_STAGES: List[Dict] = [
        {"resolution": 512,  "until_epoch": 500_000, "batch_size": 5},
        {"resolution": 768,  "until_epoch": 750_000, "batch_size": 3},
        {"resolution": 1024, "until_epoch": 1_000_000, "batch_size": 2},
    ]

    def __init__(self, config: dict):
        cfg = config.get("curriculum", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.stages: List[Dict] = cfg.get("stages", self._DEFAULT_STAGES)
        self.resolution_jitter: float = float(cfg.get("resolution_jitter", 0.0))
        self._default_resolution: int = int(config.get("resolution", 512))
        self._default_batch_size: int = int(config.get("batch_size", 5))

    @staticmethod
    def _round16(value: int) -> int:
        """Round to the nearest multiple of 16 (NestedUNet requirement)."""
        return max(16, (value // 16) * 16)

    def _get_stage(self, iteration: int) -> Dict:
        """Return the active curriculum stage for a given iteration."""
        if not self.enabled:
            return {
                "resolution": self._default_resolution,
                "batch_size": self._default_batch_size,
            }
        for stage in self.stages:
            if iteration < stage["until_epoch"]:
                return stage
        return self.stages[-1]

    def get_resolution(self, iteration: int) -> int:
        """Return the target resolution (multiple of 16) for this iteration.

        If ``resolution_jitter`` is set, applies a random ± jitter.
        """
        stage = self._get_stage(iteration)
        res = int(stage["resolution"])
        if self.resolution_jitter > 0.0:
            jitter = random.uniform(-self.resolution_jitter, self.resolution_jitter)
            res = int(res * (1.0 + jitter))
        return self._round16(res)

    def get_batch_size(self, iteration: int) -> int:
        """Return the suggested batch size for the current curriculum stage."""
        return int(self._get_stage(iteration).get("batch_size", self._default_batch_size))
