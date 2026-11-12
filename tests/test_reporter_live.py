"""Live tracking in TrainingReporter: Discord pings and Trackio logging."""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffvax.reporter import TrainingReporter  # noqa: E402


class _Req:
    def __init__(self): self.calls = []
    def post(self, url, **kw): self.calls.append((url, kw))


def _reporter(tmp_path, monkeypatch, **cfg):
    req = _Req()
    monkeypatch.setitem(sys.modules, "requests", req)
    monkeypatch.delenv("DISCORD_WEBHOOK", raising=False)
    rep = TrainingReporter({"project_name": "t", "reporting": cfg}, str(tmp_path))
    return rep, req


def test_webhook_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK", "https://discord.com/api/webhooks/x")
    rep = TrainingReporter({"reporting": {}}, str(tmp_path))
    assert rep.webhook_url == "https://discord.com/api/webhooks/x"


def test_progress_ping_is_throttled_and_carries_chart(tmp_path, monkeypatch):
    rep, req = _reporter(tmp_path, monkeypatch, webhook_url="https://w", progress_every_minutes=10)
    rep.start_run({"gpu": "A100"})
    n0 = len(req.calls)                      # the start message
    terms = {"loss": 1.0, "loss1": 0.5, "hs": 0.9, "latent": -0.3}
    for s in range(1, 30):
        rep.log_step(s, terms, epoch=0, batch=s, n_batches=100, step_seconds=1.9)
    assert len(req.calls) == n0              # not due yet: no spam
    rep._last_progress -= 11 * 60            # pretend 11 min passed
    rep.log_step(30, terms, 0, 30, 100, 1.9)
    assert len(req.calls) == n0 + 1
    url, kw = req.calls[-1]
    assert "files" in kw and kw["files"]["file"][1][:4] == b"\x89PNG"   # chart attached


def test_errors_and_complete_still_notify(tmp_path, monkeypatch):
    rep, req = _reporter(tmp_path, monkeypatch, webhook_url="https://w")
    rep.report_error("nan_loss", "boom", epoch=1, batch=2)
    rep.report_complete(3, 0.5, "/x.pth")
    assert len(req.calls) == 2


def test_trackio_logged_every_n_steps_and_finished(tmp_path, monkeypatch):
    logged, state = [], {}
    fake = types.SimpleNamespace(
        init=lambda **kw: state.setdefault("init", kw),
        log=lambda m, step=None: logged.append((step, m)),
        finish=lambda: state.setdefault("finished", True),
    )
    monkeypatch.setitem(sys.modules, "trackio", fake)
    rep, _ = _reporter(tmp_path, monkeypatch,
                       trackio={"enabled": True, "project": "p", "log_every_n_steps": 5})
    rep.start_run({"gpu": "A100"})
    assert state["init"]["project"] == "p"
    for s in range(1, 21):
        rep.log_step(s, {"loss": 1.0}, 0, s, 100, 1.0)
    assert [st for st, _ in logged] == [5, 10, 15, 20]
    rep.report_complete(1, 0.1, "/x")
    assert state.get("finished")


def test_tracking_failures_never_raise(tmp_path, monkeypatch):
    def boom(**kw): raise RuntimeError("no network")
    monkeypatch.setitem(sys.modules, "trackio", types.SimpleNamespace(init=boom))
    class _Bad:
        def post(self, *a, **k): raise OSError("down")
    monkeypatch.setitem(sys.modules, "requests", _Bad())
    rep = TrainingReporter({"reporting": {"webhook_url": "https://w", "trackio": {"enabled": True}}}, str(tmp_path))
    rep.start_run({"a": 1})
    rep.log_step(1, {"loss": 1.0}, 0, 1, 1, 1.0)
    rep.report_error("x", "y")


def test_non_main_rank_has_no_remote(tmp_path, monkeypatch):
    rep, req = _reporter(tmp_path, monkeypatch, webhook_url="https://w", trackio={"enabled": True})
    rep.disable_remote()
    rep.start_run({"a": 1}); rep.report_error("x", "y")
    assert req.calls == [] and rep._trackio is None
