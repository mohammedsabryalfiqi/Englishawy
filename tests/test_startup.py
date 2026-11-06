import socket

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def test_gradio_launch_keeps_health_and_webhook_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGLISHAWY_AUTO_SETUP", "0")
    monkeypatch.setenv("ENGLISHAWY_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)
    import app as application

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    application.demo.launch(server_name="127.0.0.1", server_port=port,
                            prevent_thread_lock=True, quiet=True, ssr_mode=False, show_api=False)
    try:
        base = f"http://127.0.0.1:{application.demo.server_port}"
        health = httpx.get(base + "/health", timeout=10)
        assert health.status_code == 200
        assert health.json()["assets"]["phase"] == "disabled"  # proves startup lifespan ran
        assert httpx.post(base + "/telegram/webhook", json={"update_id": 1}, timeout=10).status_code == 503
    finally:
        application.demo.close()


def test_lifespan_starts_and_stops_configured_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGLISHAWY_AUTO_SETUP", "0")
    monkeypatch.setenv("ENGLISHAWY_DATA", str(tmp_path / "data"))
    import app as application

    class FakeBridge:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.base_url = "https://example.hf.space"
            self.error = ""
        async def start(self): self.started = True
        async def stop(self): self.stopped = True; self.started = False
        def status(self): return {"configured": True, "connected": self.started, "mode": "webhook", "error": ""}
        async def webhook(self, request): return JSONResponse({"ok": True})

    fake = FakeBridge()
    monkeypatch.setattr(application, "bridge", fake)
    monkeypatch.setattr(application, "bridge_error", "")
    active = FastAPI(lifespan=application.make_lifespan())
    with TestClient(active) as client:
        assert fake.started
        assert client.get("/health").json()["telegram"]["connected"] is True
        assert client.post("/telegram/webhook", json={"update_id": 1}).status_code == 200
    assert fake.stopped
