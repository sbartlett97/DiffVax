"""Training progress reporter with optional webhook notifications.

Configure via the 'reporting' section of the training YAML:

    reporting:
      webhook_url: "https://discord.com/api/webhooks/..."  # or Slack incoming-webhook
      checkpoint_every_n_epochs: 10000

The JSON event log is always written to <output_dir>/training_log.json.
Webhook notifications fire for:
  - New best-model checkpoints
  - Training completion
  - Critical errors (CUDA OOM, NaN loss, unhandled exceptions)

All network failures are silently ignored so a bad webhook URL never
aborts training.
"""

import io
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
        # Webhook: config value, else the DISCORD_WEBHOOK env var (keep the URL
        # out of committed configs).
        self.webhook_url: Optional[str] = (
            cfg.get("webhook_url")
            or os.environ.get("DISCORD_WEBHOOK")
            or os.environ.get("DISCORD_WEBHOOK_URL")
            or None
        )
        self.checkpoint_every: int = int(cfg.get("checkpoint_every_n_epochs", 50))
        self.log_path: str = os.path.join(output_dir, "training_log.json")
        self._events: List[Dict[str, Any]] = []

        # Periodic progress pings (time-based, so they work for any epoch length).
        self.progress_every_minutes: float = float(cfg.get("progress_every_minutes", 10))
        self.label: str = str(cfg.get("label") or config.get("project_name") or "diffvax")
        self.price_per_hour: Optional[float] = cfg.get("price_per_hour")
        self._t_start = time.time()
        self._last_progress = time.time()
        self._step_hist: List[Dict[str, float]] = []   # downsampled per-step terms
        self._epoch_hist: List[tuple] = []             # (epoch, avg_loss)

        # Optional Hugging Face Trackio experiment tracking.
        tcfg = cfg.get("trackio", {}) or {}
        self._trackio_cfg = tcfg
        self._trackio = None            # the module once initialised
        self._trackio_every = int(tcfg.get("log_every_n_steps", 10))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Live tracking: Trackio + periodic Discord pings
    # ------------------------------------------------------------------

    def disable_remote(self) -> None:
        """Turn off webhook and Trackio (non-main ranks)."""
        self.webhook_url = None
        self._trackio_cfg = {}
        self._trackio = None

    def start_run(self, summary: Dict[str, Any]) -> None:
        """Announce the run (Discord) and open the Trackio run. Never raises."""
        self._t_start = self._last_progress = time.time()
        if self._trackio_cfg.get("enabled", False):
            try:
                import trackio

                trackio.init(
                    project=self._trackio_cfg.get("project") or self.label,
                    name=self._trackio_cfg.get("run_name"),
                    config={k: str(v) for k, v in summary.items()},
                    space_id=self._trackio_cfg.get("space_id"),
                )
                self._trackio = trackio
            except Exception as exc:  # tracking must never stop training
                print(f"[reporter] Trackio disabled: {exc}")
                self._trackio = None
        lines = [f"**[{self.label}] training started**"]
        lines += [f"- {k}: {v}" for k, v in summary.items()]
        self._send("\n".join(lines)[:1900])

    def log_step(
        self,
        step: int,
        terms: Dict[str, float],
        epoch: int,
        batch: int,
        n_batches: int,
        step_seconds: float,
    ) -> None:
        """Per-step hook: Trackio logging (every N steps) and a throttled
        Discord progress ping with a loss chart. Cheap when nothing is due."""
        if self._trackio is not None and step % max(1, self._trackio_every) == 0:
            try:
                self._trackio.log(
                    {**terms, "step_seconds": step_seconds, "epoch": epoch}, step=step
                )
            except Exception:
                self._trackio = None
        if step % 5 == 0:
            self._step_hist.append(dict(terms))
            if len(self._step_hist) > 4000:
                self._step_hist = self._step_hist[::2]
        if (
            self.webhook_url
            and time.time() - self._last_progress >= self.progress_every_minutes * 60
        ):
            self._last_progress = time.time()
            hrs = (time.time() - self._t_start) / 3600
            shown = ", ".join(f"{k}={v:.3f}" for k, v in terms.items() if k != "scale")
            cost = f" = ${hrs * self.price_per_hour:.2f}" if self.price_per_hour else ""
            avgs = ", ".join(f"{a:.3f}" for _, a in self._epoch_hist[-4:]) or "n/a"
            self._send(
                f"**[{self.label}]** epoch {epoch} step {batch}/{n_batches} | "
                f"{step_seconds:.2f} s/it | {hrs:.1f} h{cost}\n{shown}\n"
                f"last epoch avgs: {avgs}",
                image=self._chart(),
            )

    def _chart(self) -> Optional[bytes]:
        """PNG of recent per-step terms and epoch averages; None if matplotlib is absent."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
            for key in ("loss", "loss1", "hs", "latent"):
                ys = [h[key] for h in self._step_hist if key in h]
                if ys:
                    ax[0].plot(ys, label=key, lw=1)
            ax[0].set_title("per-step terms"); ax[0].legend(fontsize=7)
            if self._epoch_hist:
                ax[1].plot([e for e, _ in self._epoch_hist], [a for _, a in self._epoch_hist], "o-")
            ax[1].set_title("epoch average loss")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
            return buf.getvalue()
        except Exception:
            return None

    def report_epoch(
        self,
        epoch: int,
        avg_loss: float,
        per_model: Dict[str, float],
    ) -> None:
        """Record average epoch loss (log + Trackio; Discord gets it via progress pings)."""
        self._epoch_hist.append((epoch, avg_loss))
        if self._trackio is not None:
            try:
                self._trackio.log(
                    {"epoch_avg_loss": avg_loss, **{f"epoch_loss_{k}": v for k, v in per_model.items()}},
                    step=None,
                )
            except Exception:
                self._trackio = None
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
            f"final loss={final_loss:.5f}\n`{path}`",
            image=self._chart(),
        )
        if self._trackio is not None:
            try:
                self._trackio.finish()
            except Exception:
                pass

    def report_error(
        self,
        error_type: str,
        message: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
    ) -> None:
        """Record a critical error to the log and notify via webhook immediately.

        Unlike report_epoch/report_checkpoint, this always fires the webhook
        regardless of any thresholds, since critical errors require immediate
        attention.

        Args:
            error_type: Short category string, e.g. ``"cuda_oom"``,
                        ``"nan_loss"``, ``"fatal"``.
            message:    Full error message or traceback (truncated to 1 500 chars
                        in the webhook payload).
            epoch:      Epoch index at time of error (if known).
            batch:      Batch index within the epoch (if known).
        """
        event: Dict[str, Any] = {
            "type": "error",
            "error_type": error_type,
            "message": message,
            "ts": time.time(),
        }
        if epoch is not None:
            event["epoch"] = epoch
        if batch is not None:
            event["batch"] = batch
        self._append(event)

        location = ""
        if epoch is not None:
            location += f" epoch={epoch}"
        if batch is not None:
            location += f" batch={batch}"
        # Truncate message so oversized tracebacks don't blow past webhook limits
        snippet = message[:1500] + ("…" if len(message) > 1500 else "")
        self._send(
            f"**[DiffVax] \u26a0\ufe0f {error_type.upper()}**{location}\n"
            f"```\n{snippet}\n```"
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

    def _send(self, message: str, image: Optional[bytes] = None) -> None:
        """POST *message* (and an optional PNG) to the webhook; failures are silenced."""
        if not self.webhook_url:
            return
        try:
            import requests  # lazy import — only needed when webhook is configured

            if image:
                requests.post(
                    self.webhook_url,
                    data={"payload_json": json.dumps({"content": message[:1900]})},
                    files={"file": ("loss.png", image, "image/png")},
                    timeout=20,
                )
            else:
                requests.post(self.webhook_url, json={"content": message[:1900]}, timeout=10)
        except Exception:
            pass
