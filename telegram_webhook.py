"""Telegram webhook mounted on the same FastAPI/Gradio port (no second public port)."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from telegram import Update
from telegram.error import InvalidToken, NetworkError, TelegramError, TimedOut
from telegram.ext import Application

from bot import register_handlers
from config import public_base_url
from services import Services

logger = logging.getLogger(__name__)
WEBHOOK_PATH = "/telegram/webhook"
MAX_UPDATE_BYTES = 1024 * 1024  # JSON metadata only; files are downloaded separately.


def describe_telegram_error(exc: Exception) -> str:
    """Never include exception text that may embed the bot-token URL."""
    if isinstance(exc, InvalidToken):
        return "InvalidToken: توكن البوت غير صحيح؛ راجع TELEGRAM_BOT_TOKEN في Space secrets"
    if isinstance(exc, TimedOut):
        return "TimedOut: انتهت مهلة الاتصال بتليجرام؛ أعد تشغيل Space أو تحقق من الشبكة"
    if isinstance(exc, NetworkError):
        return "NetworkError: تعذر الوصول إلى api.telegram.org من الـSpace"
    if isinstance(exc, TelegramError):
        return f"{type(exc).__name__}: رفضت Telegram الطلب؛ تحقق من التوكن وإعدادات Webhook"
    return f"{type(exc).__name__}: تحقق من التوكن ورابط HTTPS والاتصال بتليجرام"


class TelegramBridge:
    def __init__(self, services: Services, token: str, secret: str, base_url: str):
        if len(secret) < 16:
            raise ValueError("TELEGRAM_WEBHOOK_SECRET must contain at least 16 characters")
        if not base_url.startswith("https://"):
            raise ValueError("A public HTTPS URL is required for Telegram webhook mode")
        self.token, self.secret, self.base_url = token, secret, base_url
        self.application = Application.builder().token(token).updater(None).concurrent_updates(4).build()
        register_handlers(self.application, services)
        self.queue: asyncio.Queue | None = None
        self.tasks: list[asyncio.Task] = []
        self.started = False
        self.error = ""

    @classmethod
    def from_env(cls, services: Services) -> "TelegramBridge | None":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return None
        return cls(services, token, os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip(), public_base_url())

    async def start(self) -> None:
        await self.application.initialize()
        await self.application.start()
        self.queue = asyncio.Queue(maxsize=128)
        self.tasks = [asyncio.create_task(self._consumer(), name=f"telegram-worker-{i}") for i in range(4)]
        try:
            await self.application.bot.set_webhook(
                url=self.base_url + WEBHOOK_PATH, secret_token=self.secret,
                allowed_updates=["message", "callback_query"], drop_pending_updates=False,
                max_connections=40,
            )
            self.started = True
        except Exception:
            await self.stop()
            raise

    async def _consumer(self) -> None:
        assert self.queue is not None
        while True:
            update = await self.queue.get()
            try:
                await self.application.process_update(update)
            except Exception as exc:
                logger.error("Telegram update failed: %s", type(exc).__name__)
            finally:
                self.queue.task_done()

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        if self.application.running:
            await self.application.stop()
        if self.application.initialized:
            await self.application.shutdown()
        self.started = False

    async def webhook(self, request: Request) -> JSONResponse:
        supplied = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not hmac.compare_digest(supplied, self.secret):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_UPDATE_BYTES):
            return JSONResponse({"error": "update too large"}, status_code=413)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_UPDATE_BYTES:
                return JSONResponse({"error": "update too large"}, status_code=413)
            body.extend(chunk)
        try:
            data = json.loads(body)
            if not isinstance(data, dict) or not isinstance(data.get("update_id"), int):
                raise ValueError("invalid update")
            update = Update.de_json(data, self.application.bot)
        except (ValueError, TypeError, KeyError):
            return JSONResponse({"error": "invalid update"}, status_code=400)
        if not self.started or self.queue is None:
            return JSONResponse({"error": "bot not ready"}, status_code=503)
        try:
            self.queue.put_nowait(update)
        except asyncio.QueueFull:
            return JSONResponse({"error": "busy"}, status_code=503)
        return JSONResponse({"ok": True})

    def status(self) -> dict:
        return {"configured": True, "connected": self.started, "mode": "webhook", "error": self.error}


def mount_telegram(app: FastAPI, bridge: TelegramBridge | None) -> None:
    @app.post(WEBHOOK_PATH)
    async def telegram_webhook(request: Request):
        if bridge is None:
            return JSONResponse({"error": "not configured"}, status_code=503)
        return await bridge.webhook(request)
