"""Telegram transport: long polling by default; optional webhook on Gradio's port."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from telegram import Update
from telegram.error import Conflict, InvalidToken, NetworkError, TelegramError, TimedOut
from telegram.ext import Application

from bot import post_init, register_handlers
from config import public_base_url
from services import Services

logger = logging.getLogger(__name__)
WEBHOOK_PATH = "/telegram/webhook"
MAX_UPDATE_BYTES = 1024 * 1024  # JSON metadata only; files are downloaded separately.


def describe_telegram_error(exc: Exception) -> str:
    """Never include exception text that may embed the bot-token URL."""
    if isinstance(exc, Conflict):
        return "Conflict: نسخة أخرى تستهلك نفس التوكن؛ أوقف البوت المكرر ثم أعد التشغيل"
    if isinstance(exc, InvalidToken):
        return "InvalidToken: توكن البوت غير صحيح؛ راجع TELEGRAM_BOT_TOKEN في Space secrets"
    if isinstance(exc, TimedOut):
        return "TimedOut: انتهت مهلة الاتصال بتليجرام؛ أعد تشغيل Space أو تحقق من الشبكة"
    if isinstance(exc, NetworkError):
        return "NetworkError: تعذر الوصول إلى api.telegram.org من الـSpace"
    if isinstance(exc, TelegramError):
        return f"{type(exc).__name__}: رفضت Telegram الطلب؛ تحقق من التوكن والصلاحيات وإعدادات الاتصال"
    return f"{type(exc).__name__}: تحقق من التوكن والاتصال بتليجرام"


class TelegramBridge:
    def __init__(self, services: Services, token: str, secret: str, base_url: str, mode: str = "webhook"):
        self.mode = mode
        if mode == "webhook":
            if len(secret) < 16:
                raise ValueError("TELEGRAM_WEBHOOK_SECRET must contain at least 16 characters")
            if not base_url.startswith("https://"):
                raise ValueError("A public HTTPS URL is required for Telegram webhook mode")
        self.token, self.secret, self.base_url = token, secret, base_url
        builder = (Application.builder().token(token).concurrent_updates(int(os.getenv("BOT_CONCURRENT_UPDATES", "16")))
                   .post_init(post_init).connect_timeout(15).read_timeout(30)
                   .write_timeout(30).pool_timeout(15).get_updates_read_timeout(35))
        if mode == "webhook":
            builder = builder.updater(None)  # polling mode keeps the default Updater
        self.application = builder.build()
        register_handlers(self.application, services)
        self.queue: asyncio.Queue | None = None
        self.tasks: list[asyncio.Task] = []
        self.started = False
        self.error = ""
        self.hits = 0            # POST requests that reached /telegram/webhook
        self.accepted = 0        # valid updates queued
        self.processed = 0       # updates fully handled
        self.failed = 0          # handler crashes
        self.last_hit_at = 0.0
        self.last_reject = ""

    @classmethod
    def from_env(cls, services: Services) -> "TelegramBridge | None":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return None
        mode = os.getenv("TELEGRAM_MODE", "polling").strip().lower()
        if mode not in {"webhook", "polling"}:
            raise ValueError("TELEGRAM_MODE must be 'webhook' or 'polling'")
        base = public_base_url() if mode == "webhook" else ""
        return cls(services, token, os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip(), base, mode)

    async def start(self) -> None:
        await self.application.initialize()
        # PTB only runs post_init automatically in run_polling/run_webhook.
        # This bridge starts the Application manually, so invoke it here too.
        callback = getattr(self.application, "post_init", None)
        if callback is not None:
            await callback(self.application)
        if self.mode == "polling":
            # Works on PRIVATE Spaces too: only outbound calls to api.telegram.org are needed.
            try:
                await self.application.bot.delete_webhook(drop_pending_updates=False)
                await self.application.start()
                await self.application.updater.start_polling(
                    allowed_updates=["message", "callback_query"], drop_pending_updates=False,
                    timeout=30, bootstrap_retries=-1, error_callback=self._polling_error)
                self.started = True
                self.error = ""
                logger.info("Telegram polling started")
            except Exception:
                await self.stop()
                raise
            return
        await self.application.start()
        self.queue = asyncio.Queue(maxsize=128)
        self.tasks = [asyncio.create_task(self._consumer(), name=f"telegram-worker-{i}") for i in range(4)]
        try:
            await self.application.bot.set_webhook(
                url=self.base_url + WEBHOOK_PATH, secret_token=self.secret,
                allowed_updates=["message", "callback_query"], drop_pending_updates=True,
                max_connections=40,
            )
            self.started = True
            self.error = ""
            info = await self.webhook_info()
            logger.info("Telegram getWebhookInfo: %s", info)
        except Exception:
            await self.stop()
            raise

    def _polling_error(self, exc: TelegramError) -> None:
        # PTB calls this from the polling retry loop; never leak token-bearing text.
        self.error = describe_telegram_error(exc)
        logger.error("Telegram polling error: %s", self.error)

    async def _consumer(self) -> None:
        assert self.queue is not None
        while True:
            update = await self.queue.get()
            try:
                await self.application.process_update(update)
                self.processed += 1
            except Exception as exc:
                self.failed += 1
                logger.error("Telegram update %s failed: %s", update.update_id, type(exc).__name__)
            finally:
                self.queue.task_done()

    async def stop(self) -> None:
        updater = getattr(self.application, "updater", None)
        if self.mode == "polling" and updater is not None and updater.running:
            await updater.stop()
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

    def _reject(self, reason: str, status: int) -> JSONResponse:
        self.last_reject = f"{reason} @ {time.strftime('%H:%M:%S')}"
        logger.warning("Telegram webhook rejected: %s", reason)
        return JSONResponse({"error": reason}, status_code=status)

    async def webhook_info(self) -> dict:
        """What Telegram itself says about delivery (never includes the token)."""
        try:
            info = await self.application.bot.get_webhook_info()
        except Exception as exc:
            return {"error": describe_telegram_error(exc)}
        return {"url": info.url, "pending_update_count": info.pending_update_count,
                "last_error_message": info.last_error_message,
                "last_error_date": info.last_error_date.isoformat() if info.last_error_date else None,
                "ip_address": info.ip_address}

    async def webhook(self, request: Request) -> JSONResponse:
        self.hits += 1
        self.last_hit_at = time.time()
        supplied = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not hmac.compare_digest(supplied, self.secret):
            return self._reject("bad or missing secret header", 403)
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_UPDATE_BYTES):
            return self._reject("update too large", 413)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_UPDATE_BYTES:
                return self._reject("update too large", 413)
            body.extend(chunk)
        try:
            data = json.loads(body)
            if not isinstance(data, dict) or not isinstance(data.get("update_id"), int):
                raise ValueError("invalid update")
            update = Update.de_json(data, self.application.bot)
        except (ValueError, TypeError, KeyError):
            return self._reject("invalid update", 400)
        if not self.started or self.queue is None:
            return self._reject("bot not ready", 503)
        try:
            self.queue.put_nowait(update)
        except asyncio.QueueFull:
            return self._reject("queue full", 503)
        self.accepted += 1
        logger.info("Telegram update %s queued", update.update_id)
        return JSONResponse({"ok": True})

    def status(self) -> dict:
        connected = self.started
        updater = getattr(self.application, "updater", None)
        if self.mode == "polling" and updater is not None:
            connected = connected and bool(updater.running)
        return {"configured": True, "connected": connected, "mode": self.mode, "error": self.error,
                "hits": self.hits, "accepted": self.accepted, "processed": self.processed,
                "failed": self.failed, "last_reject": self.last_reject,
                "last_hit_secs_ago": round(time.time() - self.last_hit_at) if self.last_hit_at else None}


def mount_telegram(app: FastAPI, bridge: TelegramBridge | None) -> None:
    @app.post(WEBHOOK_PATH)
    async def telegram_webhook(request: Request):
        if bridge is None:
            return JSONResponse({"error": "not configured"}, status_code=503)
        return await bridge.webhook(request)
