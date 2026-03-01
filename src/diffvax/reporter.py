"""Training progress reporter with optional webhook notifications.

Configure via the 'reporting' section of the training YAML:

    reporting:
      webhook_url: "https://discord.com/api/webhooks/..."  # or Slack incoming-webhook
      checkpoint_every_n_epochs: 10000

The JSON event log is always written to <output_dir>/training_log.json.
Webhook notifications fire for new best-model checkpoints and training
completion. All network failures are silently ignored so a bad webhook URL
never aborts training.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional


class TrainingReporter:
    """Records training events to a JSON log and optionally notifies a webhook.

    Args:
        config:     Full training config dict (reads ``config["reporting"]``).
        output_dir: Directory where ``training_log.json`` is written.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str) -> None:
        cfg: Dict[str, Any] = config.get("reporting", {})
        self.webhook_url: Optional[str] = cfg.get("webhook_url") or None
        self.checkpoint_every: int = int(cfg.get("checkpoint_every_n_epochs", 10000))
        self.log_path: str = os.path.join(output_dir, "training_log.json")
        self._events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report_epoch(
        self,
        epoch: int,
        avg_loss: float,
        per_model: Dict[str, float],
    ) -> None:
        """Record average epoch loss (written to log; no webhook notification)."""
        self._append(
            {
                "type": "epoch",
                "epoch": epoch,
                "avg_loss": avg_loss,
                "per_model": per_model,
                "ts": time.time(),
            }
        )

    def report_checkpoint(
        self,
        epoch: int,
        avg_loss: float,
        path: str,
        is_best: bool = False,
    ) -> None:
        """Record a checkpoint save and optionally notify via webhook.

        Webhook fires only when *is_best* is True, to avoid noisy periodic pings.
        """
        tag = "best" if is_best else "periodic"
        self._append(
            {
                "type": "checkpoint",
                "tag": tag,
                "epoch": epoch,
                "avg_loss": avg_loss,
                "path": path,
                "ts": time.time(),
            }
        )
        if is_best:
            self._send(
                f"**[DiffVax] New best checkpoint** — epoch {epoch}, "
                f"loss={avg_loss:.5f}\n`{path}`"
            )

    def report_complete(
        self, total_epochs: int, final_loss: float, path: str
    ) -> None:
        """Record training completion and notify via webhook (always, if configured)."""
        self._append(
            {
                "type": "complete",
                "total_epochs": total_epochs,
                "final_loss": final_loss,
                "path": path,
                "ts": time.time(),
            }
        )
        self._send(
            f"**[DiffVax] Training complete** — {total_epochs} epochs, "
            f"final loss={final_loss:.5f}\n`{path}`"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(self, event: Dict[str, Any]) -> None:
        """Append event to the in-memory list and flush to JSON file."""
        self._events.append(event)
        try:
            with open(self.log_path, "w") as fh:
                json.dump(self._events, fh, indent=2)
        except OSError:
            pass  # Non-fatal; log write failures must not abort training

    def _send(self, message: str) -> None:
        """POST *message* to the configured webhook URL; failures are silenced."""
        if not self.webhook_url:
            return
        try:
            import requests  # lazy import — only needed when webhook is configured

            requests.post(
                self.webhook_url,
                json={"content": message},
                timeout=10,
            )
        except Exception:
            pass
