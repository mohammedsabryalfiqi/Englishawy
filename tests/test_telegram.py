import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bot import handle_document, post_init
from services import Services
from telegram_webhook import TelegramBridge, mount_telegram


def test_webhook_requires_secret_and_enqueues_only_valid_updates(tmp_path, monkeypatch):
    monkeypatch.delenv("OWNER_TELEGRAM_ID", raising=False)
    services = Services.create(tmp_path)
    bridge = TelegramBridge(services, "123456:TEST_TOKEN_NOT_REAL", "a-long-test-webhook-secret", "https://test.hf.space")
    bridge.started = True
    bridge.queue = asyncio.Queue(maxsize=4)
    app = FastAPI()
    mount_telegram(app, bridge)
    with TestClient(app) as client:
        assert client.post("/telegram/webhook", json={"update_id": 1}).status_code == 403
        headers = {"X-Telegram-Bot-Api-Secret-Token": bridge.secret}
        assert client.post("/telegram/webhook", json={"update_id": 1}, headers=headers).status_code == 200
        assert bridge.queue.qsize() == 1
        assert client.post("/telegram/webhook", json=[], headers=headers).status_code == 400
        assert client.post("/telegram/webhook", content=b"x" * (1024 * 1024 + 1), headers=headers).status_code == 413


def test_telegram_defaults_to_polling_without_webhook_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN_NOT_REAL")
    monkeypatch.delenv("TELEGRAM_MODE", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)
    bridge = TelegramBridge.from_env(Services.create(tmp_path))
    assert bridge is not None
    assert bridge.mode == "polling"
    assert bridge.secret == "" and bridge.base_url == ""
    assert bridge.application.updater is not None
    assert bridge.application.post_init is post_init
    asyncio.run(bridge.application.shutdown())


def test_polling_start_preserves_pending_updates_and_stops_cleanly(tmp_path):
    async def run():
        bridge = TelegramBridge(Services.create(tmp_path), "123456:TEST_TOKEN_NOT_REAL", "", "", mode="polling")
        calls = []

        class Bot:
            async def delete_webhook(self, **kwargs):
                calls.append(("delete_webhook", kwargs))

        class Updater:
            running = False
            async def start_polling(self, **kwargs):
                calls.append(("start_polling", kwargs))
                self.running = True
            async def stop(self):
                calls.append(("updater_stop", {}))
                self.running = False

        async def fake_post_init(app):
            calls.append(("post_init", {}))

        class App:
            initialized = False
            running = False
            bot = Bot()
            updater = Updater()
            post_init = staticmethod(fake_post_init)
            async def initialize(self):
                calls.append(("initialize", {}))
                self.initialized = True
            async def start(self):
                calls.append(("start", {}))
                self.running = True
            async def stop(self):
                calls.append(("stop", {}))
                self.running = False
            async def shutdown(self):
                calls.append(("shutdown", {}))
                self.initialized = False

        bridge.application = App()
        await bridge.start()
        assert bridge.started and bridge.status()["connected"] is True
        assert calls[0] == ("initialize", {})
        assert calls[1] == ("post_init", {})
        assert calls[2] == ("delete_webhook", {"drop_pending_updates": False})
        assert calls[3] == ("start", {})
        name, kwargs = calls[4]
        assert name == "start_polling"
        assert kwargs["allowed_updates"] == ["message", "callback_query"]
        assert kwargs["drop_pending_updates"] is False
        assert kwargs["timeout"] == 30
        assert kwargs["bootstrap_retries"] == -1
        assert callable(kwargs["error_callback"])
        await bridge.stop()
        assert not bridge.started and bridge.status()["connected"] is False
        assert [name for name, _ in calls[-3:]] == ["updater_stop", "stop", "shutdown"]

    asyncio.run(run())


def test_invalid_transport_and_webhook_requirements_fail_closed(monkeypatch, tmp_path):
    services = Services.create(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:TEST_TOKEN_NOT_REAL")
    monkeypatch.setenv("TELEGRAM_MODE", "invalid")
    with pytest.raises(ValueError):
        TelegramBridge.from_env(services)
    monkeypatch.setenv("TELEGRAM_MODE", "webhook")
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)
    with pytest.raises(ValueError):
        TelegramBridge.from_env(services)


def test_non_owner_font_upload_is_rejected_before_download(tmp_path):
    asyncio.run(_non_owner_case(tmp_path))


async def _non_owner_case(tmp_path):
    services = Services.create(tmp_path)
    calls = []
    replies = []
    async def forbidden_get_file(*args, **kwargs):
        calls.append("download")
        raise AssertionError("must not download")
    async def reply_text(text):
        replies.append(text)
    document = SimpleNamespace(file_name="fonts.zip", file_size=100, mime_type="application/zip", get_file=forbidden_get_file)
    update = SimpleNamespace(effective_user=SimpleNamespace(id=123),
                             effective_chat=SimpleNamespace(id=123, type="private"),
                             message=SimpleNamespace(document=document, reply_text=reply_text))
    context = SimpleNamespace(bot_data={"services": services})
    await handle_document(update, context)
    assert calls == []
    assert "للمالك فقط" in replies[0]
