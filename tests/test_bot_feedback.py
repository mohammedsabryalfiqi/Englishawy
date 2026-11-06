import asyncio
import io
import time
from types import SimpleNamespace

import pytest
from PIL import Image

import bot
from config import admin_telegram_ids, is_admin, reply_delay_seconds
from font_matcher import FontMatch
from pipeline import Identification
from services import Services


class Message:
    def __init__(self):
        self.replies = []
        self.edits = []
        self.deleted = False
        self.markup_removed = False

    async def reply_text(self, text, **kwargs):
        message = Message()
        self.replies.append((text, message))
        return message

    async def edit_text(self, text, **kwargs):
        self.edits.append(text)

    async def delete(self):
        self.deleted = True

    async def edit_reply_markup(self, **kwargs):
        self.markup_removed = True


class Chat:
    id = 10
    type = "private"

    def __init__(self):
        self.photos = []

    async def send_action(self, action):
        pass

    async def send_photo(self, **kwargs):
        self.photos.append(kwargs)


def result(owner=False):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    match = FontMatch("owner:1", "Owner Script", "owner", "", 0.9, 0.8, 0.1, 0.1, 0.8,
                      source="owner" if owner else "external:WhatFontIs")
    return Identification("welcome baby", SimpleNamespace(confidence=43), SimpleNamespace(),
                          [match], buffer.getvalue(), 1, 0.2, False,
                          winning_stage="owner_library" if owner else "whatfontis", fallback_used=owner)


def test_admin_ids_are_explicit_and_private(monkeypatch):
    monkeypatch.setenv("OWNER_TELEGRAM_ID", "100")
    monkeypatch.setenv("ADMIN_TELEGRAM_IDS", "200, 300;200")
    assert admin_telegram_ids() == {200, 300}
    assert is_admin(100, 100, "private")
    assert is_admin(200, 200, "private")
    assert not is_admin(200, -100, "supergroup")
    assert not is_admin(400, 400, "private")
    monkeypatch.setenv("ADMIN_TELEGRAM_IDS", "200,bad")
    with pytest.raises(ValueError):
        admin_telegram_ids()


def test_delay_defaults_to_twenty_and_is_bounded(monkeypatch):
    monkeypatch.delenv("BOT_REPLY_DELAY_SECONDS", raising=False)
    assert reply_delay_seconds() == 20
    monkeypatch.setenv("BOT_REPLY_DELAY_SECONDS", "-1")
    assert reply_delay_seconds() == 0
    monkeypatch.setenv("BOT_REPLY_DELAY_SECONDS", "500")
    assert reply_delay_seconds() == 120
    monkeypatch.setenv("BOT_REPLY_DELAY_SECONDS", "nan")
    assert reply_delay_seconds() == 20
    monkeypatch.setenv("BOT_REPLY_DELAY_SECONDS", "inf")
    assert reply_delay_seconds() == 20


def test_only_two_feedback_buttons_and_short_caption():
    keyboard = bot._keyboard("abc123")
    assert len(keyboard.inline_keyboard) == 1
    buttons = keyboard.inline_keyboard[0]
    assert [button.text for button in buttons] == ["✅ صحيحة", "❌ خاطئة"]
    assert buttons[0].callback_data == "vote:abc123:correct"
    assert buttons[1].callback_data == "vote:abc123:wrong"
    assert buttons[0].style == "success" and buttons[1].style == "danger"
    payload = keyboard.to_dict()["inline_keyboard"][0]
    assert payload[0]["style"] == "success" and payload[1]["style"] == "danger"
    caption = bot._caption(result())
    assert caption == "🎯 Owner Script\n📌 WhatFontIs"
    assert len(caption.splitlines()) == 2


def test_command_menus_default_empty_and_scoped(monkeypatch):
    monkeypatch.setenv("OWNER_TELEGRAM_ID", "100")
    monkeypatch.setenv("ADMIN_TELEGRAM_IDS", "200")
    calls = []

    class FakeBot:
        async def set_my_commands(self, commands, scope=None, **kwargs):
            calls.append((scope.type, getattr(scope, "chat_id", None), [c.command for c in commands]))

    asyncio.run(bot.configure_command_menus(FakeBot()))
    assert calls[:4] == [("default", None, []), ("all_private_chats", None, []),
                         ("all_group_chats", None, []), ("all_chat_administrators", None, [])]
    assert calls[4][0:2] == ("chat", 100)
    assert "start" in calls[4][2] and "removefont" in calls[4][2]
    assert calls[5] == ("chat", 200, ["start", "help", "status"])


def test_unreachable_admin_chat_does_not_block_command_menus(monkeypatch):
    monkeypatch.setenv("OWNER_TELEGRAM_ID", "100")
    monkeypatch.setenv("ADMIN_TELEGRAM_IDS", "200")
    calls = []

    class FakeBot:
        async def set_my_commands(self, commands, scope=None, **kwargs):
            if getattr(scope, "chat_id", None) == 200:
                raise bot.TelegramError("chat not found")
            calls.append((scope.type, getattr(scope, "chat_id", None)))

    asyncio.run(bot.configure_command_menus(FakeBot()))
    assert ("chat", 100) in calls
    assert len(calls) == 5


def test_wrong_button_uses_owner_library_without_external_search(tmp_path, monkeypatch):
    async def run():
        services = Services.create(tmp_path)
        token = services.sessions.put((10, 10), b"image")
        services.sessions.finish((10, 10), token, "welcome baby", False)
        message = Message()
        answers = []

        async def answer(text="", **kwargs):
            answers.append(text)

        query = SimpleNamespace(data=f"vote:{token}:wrong", message=message, answer=answer)
        update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=10),
                                 effective_chat=Chat())
        context = SimpleNamespace(bot_data={"services": services})
        calls = []

        async def fake_identify(svc, image, mode, text, owner_only=False):
            calls.append((image, text, owner_only))
            return result(owner=True)

        monkeypatch.setattr(bot, "_identify_async", fake_identify)
        monkeypatch.setenv("BOT_REPLY_DELAY_SECONDS", "0")
        await bot.handle_callback(update, context)
        assert calls == [(b"image", "welcome baby", True)]
        assert message.markup_removed
        assert "مكتبة المالك" in answers[0]
        state = services.sessions.current((10, 10))
        assert state.owner_only and not state.busy
        assert state.text == "welcome baby"

    asyncio.run(run())


def test_stale_card_cannot_reanalyze_latest_image(tmp_path):
    async def run():
        services = Services.create(tmp_path)
        old = services.sessions.put((10, 10), b"old")
        services.sessions.put((10, 10), b"new")
        answers = []

        async def answer(text="", **kwargs):
            answers.append(text)

        query = SimpleNamespace(data=f"vote:{old}:wrong", message=Message(), answer=answer)
        update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=10),
                                 effective_chat=Chat())
        await bot.handle_callback(update, SimpleNamespace(bot_data={"services": services}))
        assert "انتهت جلسة الصورة" in answers[0]
        assert services.sessions.get((10, 10)) == b"new"

    asyncio.run(run())


def test_minimum_delay_is_nonblocking_and_counts_real_work(tmp_path, monkeypatch):
    async def run():
        services = Services.create(tmp_path)
        token = services.sessions.put((10, 10), b"image")
        status = Message()
        chat = Chat()
        update = SimpleNamespace(effective_user=SimpleNamespace(id=10), effective_chat=chat)
        context = SimpleNamespace(bot_data={"services": services})
        monkeypatch.setenv("BOT_REPLY_DELAY_SECONDS", "0.12")
        heartbeats = 0

        async def fake_identify(*args, **kwargs):
            await asyncio.sleep(0.04)
            return result()

        async def other_task():
            nonlocal heartbeats
            for _ in range(4):
                await asyncio.sleep(0.01)
                heartbeats += 1

        monkeypatch.setattr(bot, "_identify_async", fake_identify)
        started = time.monotonic()
        await asyncio.gather(bot._analyze_and_respond(update, context, status, b"image", "auto", None, token),
                             other_task())
        assert time.monotonic() - started >= 0.11
        assert heartbeats == 4
        assert status.deleted
        assert not services.sessions.current((10, 10)).busy

    asyncio.run(run())
