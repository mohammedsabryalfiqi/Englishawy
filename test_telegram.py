import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bot import handle_document
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


def test_image_sessions_awaiting_does_not_deadlock():
    import threading
    from services import ImageSessions

    sessions = ImageSessions()
    sessions.put((1, 1), b"img")

    def work():
        sessions.awaiting((1, 1), True)
        assert sessions.is_awaiting((1, 1))
        sessions.awaiting((1, 1), False)
        assert not sessions.is_awaiting((1, 1))

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(3)
    assert not thread.is_alive(), "ImageSessions deadlocked (non re-entrant lock)"
