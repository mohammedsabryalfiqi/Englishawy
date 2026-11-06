"""Englishawy Telegram bot v5 — cascade search + background GitHub ingest.

Search for EVERY photo — both sources at the same time, one merged list:
  1) External provider search (network)
  2) Owner library + Google faces + DINOv2 rerank (local)
  Reciprocal Rank Fusion merges the two ranked lists. A family both sources
  name outranks anything a single source likes alone.

Owner GitHub import (bot downloads it HIMSELF in background):
  /import_github https://github.com/<owner>/<repo>[/tree/<branch>/<subpath>]
  — or simply paste the github.com link in the owner private chat.
  The download/analysis/install runs in a daemon thread: leaving the chat or
  closing the browser NEVER stops it. Track with /jobs.

Owner identity always comes from update.effective_user.id (private chat only).
Results contain names + rendered previews; the font file itself is sent only
when the owner taps "📄 ابعت ملف الخط" on a card.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import threading
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from telegram import BotCommand, BotCommandScopeAllChatAdministrators, BotCommandScopeAllGroupChats, BotCommandScopeAllPrivateChats, BotCommandScopeChat, BotCommandScopeDefault, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

import assets_runtime
from diag import dump_threads
from comparison_card import render_card
from config import FONT_SUFFIXES, admin_telegram_ids, is_admin, is_owner, owner_telegram_id, reply_delay_seconds, storage_status, whatfontis_api_key
from font_store import format_import_report
from github_ingestor import parse_github_url
from pipeline import Identification, identify
from services import Services

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs full request URLs, which contain the bot token -> keep it quiet.
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
FONT_LIMIT = int(os.getenv("FONT_LIMIT", "0")) or None
CLOUD_DOWNLOAD_LIMIT = 20 * 1024 * 1024
IMAGE_LIMIT = 10 * 1024 * 1024

GITHUB_URL_RE = re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(?:/[^\s]*)?", re.I)

def _services(context) -> Services:
    return context.bot_data["services"]


def _key(update: Update) -> tuple[int, int] | None:
    if not update.effective_chat or not update.effective_user:
        return None
    return update.effective_chat.id, update.effective_user.id


def _owner(update: Update) -> bool:
    chat, user = update.effective_chat, update.effective_user
    return bool(chat and user and is_owner(user.id, chat.id, chat.type))


def _admin(update: Update) -> bool:
    chat, user = update.effective_chat, update.effective_user
    return bool(chat and user and is_admin(user.id, chat.id, chat.type))


def _keyboard(token: str) -> InlineKeyboardMarkup:
    # Bot API 9.4+ supports real button colors. Keep the emoji as a fallback
    # for older Telegram clients that ignore `style`.
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ صحيحة", callback_data=f"vote:{token}:correct", style="success"),
         InlineKeyboardButton("❌ خاطئة", callback_data=f"vote:{token}:wrong", style="danger")],
        [InlineKeyboardButton("✏️ الكلمة غلط — عدّلها", callback_data=f"fix:{token}")],
        # The file is sent only after an explicit tap — never with the card.
        [InlineKeyboardButton("📄 ابعت ملف الخط", callback_data=f"file:{token}")],
    ])


def _source_label(match) -> str:
    src = str(getattr(match, "source", ""))
    if src == "merged":
        return "بحث خارجي + مكتبتك"
    if src.startswith("external:"):
        return "بحث خارجي"
    if src == "owner":
        return "مكتبتك"
    return "Google"


def _caption(result: Identification) -> str:
    if not result.matches:
        return "لم أجد خطاً مطابقاً."
    best = result.matches[0]
    caption = f"🎯 {best.family}\n📌 {_source_label(best)}"
    # A merged result is not a fallback, but its notices still matter (low
    # confidence, neural rerank skipped) — show them either way.
    if result.notices:
        caption += f"\n⚠️ {result.notices[0]}"
    return caption[:1024]


IDENTIFY_TIMEOUT_S = float(os.getenv("IDENTIFY_TIMEOUT_S", "300"))
SLOW_DUMP_S = float(os.getenv("SLOW_DUMP_S", "45"))
MAX_FONT_FILE = 45 * 1024 * 1024  # Bot API upload ceiling


def _font_file_name(match) -> str:
    suffix = Path(str(getattr(match, "font_path", "") or "")).suffix or ".ttf"
    name = f"{getattr(match, 'family', 'font')} {getattr(match, 'style', '')}".strip()
    name = re.sub(r'[\\/:*?"<>|\r\n]+', "-", name).strip(" .-") or "font"
    return f"{name[:80]}{suffix}"


async def _send_font_file(query, context, match) -> bool:
    """Send the real TTF/OTF — only ever after the user taps the button.

    Google and owner-library faces both carry a real file on disk. An
    external-only family has no file unless it also lives in the library, and
    that is said plainly instead of pretending to send something.
    """
    path = Path(str(getattr(match, "font_path", "") or ""))
    if not path.is_file():
        await query.message.reply_text(
            "⚠️ الخط ده معنديش ملفه — الاسم جاي من البحث الخارجي بس، "
            "وملفات الخطوط بتتبعت لما تكون موجودة في مكتبتك.")
        return False
    try:
        if path.stat().st_size > MAX_FONT_FILE:
            await query.message.reply_text("⚠️ ملف الخط أكبر من حد تيليجرام (45MB).")
            return False
    except OSError:
        await query.message.reply_text("⚠️ مش قادر أقرأ ملف الخط ده.")
        return False
    name = _font_file_name(match)
    caption = f"📄 {getattr(match, 'family', 'font')}"
    bot = getattr(context, "bot", None)
    chat = query.message.chat
    owner_id = owner_telegram_id()
    try:
        with path.open("rb") as handle:
            if (chat.type == "private" and chat.id == owner_id) or bot is None:
                await query.message.reply_document(document=handle, filename=name, caption=caption)
            else:
                # The button can be tapped from a group card; the file itself
                # goes to the owner's private chat, not to the group.
                await bot.send_document(chat_id=owner_id, document=handle, filename=name, caption=caption)
                await query.message.reply_text("📄 بعتلك ملف الخط في الخاص.")
    except TelegramError:
        logger.error("Font file delivery failed")
        await query.message.reply_text("⚠️ تعذر إرسال ملف الخط؛ جرّب تاني.")
        return False
    return True


async def threads_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner-only: where is every thread right now (used to find a stuck analysis)."""
    if not _owner(update):
        return
    await update.message.reply_text(("🧵 Threads:\n" + dump_threads())[:4000])


async def _identify_async(services: Services, image: bytes, mode: str, text: str | None,
                          owner_only: bool = False, cancel: threading.Event | None = None) -> Identification:
    fonts = await asyncio.to_thread(services.catalog.fonts, "owner" if owner_only else "all")
    ready, total = services.catalog.mirror_progress() if hasattr(services.catalog, "mirror_progress") else (0, 0)
    async with services.analysis_semaphore:
        result = await asyncio.to_thread(identify, image, top_n=MAX_RESULTS, candidates=CANDIDATES,
                                       force_invert=mode == "invert", force_binarize=mode == "binarize",
                                       text_override=text, font_limit=FONT_LIMIT,
                                       fonts=fonts, external_first=not owner_only, cancel=cancel)
    if total and ready < total:
        result.notices.append(f"⏳ لسه بجهّز مكتبتك على القرص المحلي ({ready}/{total}) — النتيجة من الجاهز حالياً.")
    return result


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = ("👋 ابعت صورة كلمة إنجليزية وهرجّعلك أقرب الخطوط بالمعاينة.\n"
            "✅ صحيحة لتأكيد النتيجة · ❌ خاطئة للبحث في مكتبة المالك فقط.")
    if _owner(update):
        text += "\n\n🔐 الإدارة: /login /owner /fonts /removefont /import_github /jobs"
    elif _admin(update):
        text += "\n\n🛠️ أوامر المدير: /help /status"
    await update.message.reply_text(text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _admin(update):
        await update.message.reply_text("🔐 الحالة متاحة للمالك والمديرين في محادثة خاصة.")
        return
    stats = await asyncio.to_thread(_services(context).catalog.stats)
    await update.message.reply_text(
        f"🔎 بحث خارجي ← مكتبتك ← Google\n"
        f"البحث الخارجي: {'مفتاح مضبوط' if whatfontis_api_key() else 'المفتاح غير مضبوط'}\n"
        f"Google: {stats['google']} خط\nمكتبة المالك: {stats['owner']['faces']} وجه / {stats['owner']['families']} عائلة\n"
        f"⏱️ وقت الرد المضبوط: {reply_delay_seconds():.0f} ثانية على الأقل\n"
        f"{storage_status()['label']}")


async def login(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _owner(update):
        await update.message.reply_text("🔐 دخول لوحة المالك متاح لحساب المالك في محادثة خاصة فقط.")
        return
    code = _services(context).auth.issue_code(update.effective_user.id, update.effective_chat.id, update.effective_chat.type)
    await update.message.reply_text(f"🔑 كود دخول لوحة الخطوط:\n{code}\n\nصالح 5 دقائق ولمرة واحدة. لا تشاركه مع أي شخص.")


async def owner_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _owner(update):
        return
    services = _services(context)
    stats = services.store.stats()
    await update.message.reply_text(f"🔐 المالك: {owner_telegram_id()}\nالوجوه: {stats['faces']}\nالعائلات: {stats['families']}\nالحجم: {stats['bytes']/1048576:.1f}MB\n{storage_status()['label']}\n\n/fonts [صفحة] لعرض الخطوط و/removefont ID للحذف.\n/import_github <رابط> لسحب مستودع خطوط ضخم في الخلفية.")


async def list_fonts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _owner(update):
        return
    page = max(1, int(context.args[0])) if context.args and context.args[0].isdigit() else 1
    entries = _services(context).store.entries(limit=20, offset=(page - 1) * 20)
    lines = [f"📚 مكتبة المالك — صفحة {page}"]
    lines += [f"{e['id']}. {e['family']} / {e['style']}" for e in entries]
    if not entries:
        lines.append("فارغة أو انتهت الصفحات. ابعت ZIP أو رابط GitHub لإضافة خطوط.")
    await update.message.reply_text("\n".join(lines)[:4000])


async def remove_font(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _owner(update):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("استخدم /removefont ID — الأرقام ظاهرة في /fonts")
        return
    removed = await asyncio.to_thread(_services(context).store.delete, int(context.args[0]))
    await update.message.reply_text("✅ اتحذف الخط من البحث والمعاينة." if removed else "لم أجد هذا ID.")


async def import_github(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Owner: /import_github <github-url> — bot downloads + installs in background."""
    if not _owner(update):
        await update.message.reply_text("🔐 سحب مستودعات GitHub متاح للمالك فقط في محادثة خاصة.")
        return
    raw = " ".join(context.args or []).strip()
    if not raw:
        await update.message.reply_text("الصق رابط المستودع بعد الأمر. مثال:\n/import_github https://github.com/google/fonts")
        return
    m = GITHUB_URL_RE.search(raw)
    url = m.group(0) if m else raw
    try:
        owner, repo, branch, _sub = parse_github_url(url)
    except ValueError as exc:
        await update.message.reply_text(f"⚠️ {exc}")
        return
    services = _services(context)
    if services.ingestor is None:
        await update.message.reply_text("⚠️ عامل السحب الخلفي غير مهيأ على الخادم.")
        return
    job = await asyncio.to_thread(services.ingestor.start_background_job, url)
    await update.message.reply_text(
        f"🚀 بدأ البوت التحميل في الخلفية (البوت يحمّل بنفسه):\n"
        f"📦 {owner}/{repo} (فرع {job.get('branch', branch)}) — مهمة #{job.get('id')}\n"
        f"⚙️ يعمل حتى لو خرجت من المحادثة أو أغلقت الصفحة.\n"
        f"تابع التقدم بـ /jobs أو من تبويب مكتبة المالك في الموقع.")


async def jobs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _owner(update):
        return
    services = _services(context)
    if services.ingestor is None:
        await update.message.reply_text("⚠️ عامل السحب الخلفي غير مهيأ.")
        return
    jobs = await asyncio.to_thread(services.ingestor.list_jobs, 10)
    if not jobs:
        await update.message.reply_text("لا توجد مهام سحب من GitHub بعد. ابعت رابط مستودع أو /import_github <رابط>.")
        return
    lines = ["⚙️ مهام سحب GitHub الخلفية:"]
    for j in jobs:
        lines.append(f"• #{j['id']} [{j['status']}] {j['repo_owner']}/{j['repo_name']} — {j['stage_label'][:90]}")
    await update.message.reply_text("\n".join(lines)[:4000])


async def _start_github_from_text(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> bool:
    """If owner pastes a bare GitHub URL, start background ingest. Returns True if handled."""
    if not _owner(update):
        return False
    services = _services(context)
    if services.ingestor is None:
        return False
    try:
        owner, repo, _b, _s = parse_github_url(url)
    except ValueError:
        return False
    job = await asyncio.to_thread(services.ingestor.start_background_job, url)
    await update.message.reply_text(
        f"🚀 التقطت رابط GitHub وبدأت التحميل الخلفي:\n📦 {owner}/{repo} — مهمة #{job.get('id')}\n"
        f"البوت يحمّل ويثبّت ويحلّل بنفسه في الخلفية. تابع بـ /jobs.")
    return True


async def _analyze_and_respond(update: Update, context: ContextTypes.DEFAULT_TYPE, status,
                               image: bytes, mode: str, text: str | None, token: str,
                               owner_only: bool = False, started: float | None = None):
    services = _services(context)
    key = _key(update)
    if key is None or not services.sessions.begin(key, token):
        await status.edit_text("⏳ فيه تحليل شغال لنفس الصورة؛ استنى النتيجة.")
        return
    started = time.monotonic() if started is None else started
    delay = reply_delay_seconds()
    cancel = threading.Event()
    task = asyncio.create_task(_identify_async(services, image, mode, text, owner_only, cancel=cancel))
    result = None
    try:
        await status.edit_text("🔎 بفحص الصورة ومراجعة الخطوط..." if not owner_only else "📚 بفحص مكتبة المالك فقط...")
        dumped = False
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=4)
            elapsed = time.monotonic() - started
            if not done and elapsed > SLOW_DUMP_S and not dumped:
                dumped = True  # مرة واحدة: يبيّن في اللوج أين يقف التحليل
                logger.warning("Analysis still running after %.0fs. Threads:\n%s", elapsed, dump_threads())
            if not done and elapsed > IDENTIFY_TIMEOUT_S:
                logger.error("Analysis TIMEOUT after %.0fs. Threads:\n%s", elapsed, dump_threads())
                cancel.set()
                raise TimeoutError("identification timeout")
            if not done:
                try:
                    await status.edit_text("🔎 لسه بفحص أشكال الحروف وأجهّز المعاينة...")
                except TelegramError:
                    pass
        result = task.result()
        # This is an explicit owner-requested minimum response time. Time spent
        # searching/rendering counts toward it; never add 20s on top of real work.
        remaining = delay - (time.monotonic() - started)
        while remaining > 0:
            try:
                await status.edit_text("🔎 مراجعة أخيرة للنتيجة والمعاينة...")
            except TelegramError:
                pass
            await asyncio.sleep(min(4, remaining))
            remaining = delay - (time.monotonic() - started)
    except TimeoutError:
        await status.edit_text("⏱️ التحليل خد وقت أطول من المسموح فأوقفته. ابعت الصورة تاني، "
                               "ولو اتكرر ابعت /threads (للمالك) أو اللوج.")
        return
    except Exception as exc:
        logger.error("Identification failed: %s", type(exc).__name__)
        await status.edit_text("⚠️ تعذر التحليل. جرّب صورة أوضح أو تأكد من تجهيز المكتبة.")
        return
    finally:
        if not task.done():
            cancel.set()  # يوقف خيط المسح الفعلي؛ task.cancel وحده لا يوقفه
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        services.sessions.finish(key, token, result.text if result else (text or ""), owner_only, result=result)
    if not result.text or not result.matches:
        message = ("📚 مكتبة المالك فارغة أو مافيها نتيجة مطابقة. ابعت صورة أوضح." if owner_only
                   else "⚠️ لم أجد نصاً/خطاً واضحاً. جرّب قصّ الكلمة بصورة أوضح.")
        await status.edit_text(message)
        return
    chat = update.effective_chat
    await chat.send_action(ChatAction.UPLOAD_PHOTO)
    await chat.send_photo(photo=io.BytesIO(result.card_png), caption=_caption(result), reply_markup=_keyboard(token))
    try:
        await status.delete()
    except TelegramError:
        pass


async def _receive_image(update: Update, context: ContextTypes.DEFAULT_TYPE, tg_file, file_size: int | None):
    key = _key(update)
    if key is None:
        return
    if file_size and file_size > IMAGE_LIMIT:
        await update.message.reply_text("الصورة كبيرة للتحليل؛ ابعت قصّاً أصغر للكلمة.")
        return
    try:
        buffer = io.BytesIO()
        await tg_file.download_to_memory(out=buffer)
        image = buffer.getvalue()
    except Exception as exc:
        logger.error("Image download failed: %s", type(exc).__name__)
        await update.message.reply_text("⚠️ تعذر تنزيل الصورة. ابعتها تاني.")
        return
    if len(image) > IMAGE_LIMIT:
        await update.message.reply_text("الصورة أكبر من الحد المخصص للتحليل؛ ابعت قصّاً للكلمة.")
        return
    started = time.monotonic()
    token = _services(context).sessions.put(key, image)
    override = (update.message.caption or "").strip() or None
    status = await update.message.reply_text("🔎 استلمت الصورة، جاري الفحص والمراجعة...")
    await _analyze_and_respond(update, context, status, image, "auto", override, token, started=started)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    await _receive_image(update, context, await photo.get_file(), photo.file_size)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if not document:
        return
    suffix = Path(document.file_name or "").suffix.lower()
    if suffix in FONT_SUFFIXES | {".zip"}:
        # Authorize BEFORE get_file or any download/disk work.
        if not _owner(update):
            await update.message.reply_text("🔐 إضافة الخطوط متاحة للمالك فقط في محادثة خاصة.")
            return
        local_api = os.getenv("TELEGRAM_LOCAL_API", "0").lower() in {"1", "true", "yes"}
        if not local_api and document.file_size and document.file_size > CLOUD_DOWNLOAD_LIMIT:
            await login_hint(update)
            return
        status = await update.message.reply_text("📦 جاري تنزيل وفحص الخطوط على CPU...")
        services = _services(context)
        try:
            tg_file = await document.get_file()
            with tempfile.TemporaryDirectory(prefix="telegram-font-", dir=services.store.tmp_dir) as work:
                path = Path(work) / f"upload{suffix}"
                await tg_file.download_to_drive(custom_path=str(path))
                if suffix == ".zip":
                    report = await asyncio.to_thread(services.store.import_zip, path)
                    message = format_import_report(report)
                else:
                    result = await asyncio.to_thread(services.store.install_font, path, document.file_name or "telegram")
                    message = f"✅ وجوه جديدة: {result['added']} | مكررة: {result['duplicates']}\n" + "، ".join(result["families"])
            await status.edit_text((message + "\n\n" + storage_status()["label"])[:4000])
        except Exception as exc:
            logger.error("Font import failed: %s", type(exc).__name__)
            await status.edit_text(f"⚠️ تعذر استيراد الملف: {str(exc)[:300]}")
        return
    if (document.mime_type or "").startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        await _receive_image(update, context, await document.get_file(), document.file_size)
    else:
        await update.message.reply_text("ابعت صورة لتحديد الخط، أو ZIP/خطوط من حساب المالك، أو رابط GitHub لسحبه في الخلفية.")


async def login_hint(update: Update):
    await update.message.reply_text("📦 Telegram Bot API العادي يسمح بتنزيل ملفات حتى 20MB فقط.\n"
                                    "🚀 للمستودعات/الأرشيفات الضخمة ابعت رابط GitHub هنا (أو /import_github <رابط>) — البوت يحمّله ويثبّته ويحلّله بنفسه في الخلفية.\n"
                                    "لرفع ZIP ضخم يدوياً: اطلب /login هنا، وافتح تبويب «مكتبة المالك» في Gradio وأدخل الكود. "
                                    "الرفع هناك بلا حدّ حجم مفروض في التطبيق، حسب سعة التخزين والمنصة.\n"
                                    "البديل: شغّل Local Bot API server واضبط TELEGRAM_LOCAL_API=1.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    incoming = (update.message.text or "").strip()
    # After "✏️ الكلمة غلط", the next plain message is the corrected word.
    fix = context.user_data.pop("word_fix", None) if context.user_data else None
    if fix and incoming and time.monotonic() < fix.get("until", 0) and not GITHUB_URL_RE.search(incoming):
        await _apply_word_fix(update, context, fix, incoming)
        return
    # Owner pastes a GitHub repo link directly (no command needed).
    m = GITHUB_URL_RE.search(incoming)
    if m and _owner(update):
        handled = await _start_github_from_text(update, context, m.group(0))
        if handled:
            return
    await update.message.reply_text(
        "ابعت 📸 صورة فيها كلمة إنجليزية، وهرد باسم الخط ومعاينته فقط.\n"
        "لو عايز تصحح النص، ابعت الصورة ومعاها الكابتشن الصحيح.\n"
        "(البحث: خارجي + مكتبتك في نفس الوقت، وملف الخط يتبعت لما تطلبه)")


async def _apply_word_fix(update: Update, context: ContextTypes.DEFAULT_TYPE, fix: dict, word: str):
    """Redraw the card with the corrected word, keeping the same candidates."""
    key = _key(update)
    services = _services(context)
    state = services.sessions.state(key, fix.get("token", "")) if key else None
    if state is None:
        await update.message.reply_text("⌛ جلسة الصورة انتهت؛ ابعتها تاني وأنا أعرض البطاقة بالكلمة الصحيحة.")
        return
    word = (word or "").strip()[:160]
    if not word:
        await update.message.reply_text("✏️ ابعت الكلمة الصحيحة نفسها.")
        return
    result = getattr(state, "result", None)
    if result is None:
        # No cached render (session from an older version): redo the search.
        fresh = services.sessions.put(key, state.image)
        status = await update.message.reply_text("✏️ بحدّث الكلمة وأعيد المقارنة...")
        await _analyze_and_respond(update, context, status, state.image, "auto", word, fresh)
        return
    try:
        card = await asyncio.to_thread(render_card, result.prep.original_crop, word,
                                       result.matches, len(result.matches), None)
    except Exception as exc:
        logger.error("Word refresh failed: %s", type(exc).__name__)
        await update.message.reply_text("⚠️ تعذر تحديث الكلمة. ابعت الصورة تاني.")
        return
    state.text = word
    caption = (_caption(result) + f"\n✏️ الكلمة في المعاينة: {word}")[:1024]
    chat = update.effective_chat
    await chat.send_action(ChatAction.UPLOAD_PHOTO)
    media = InputMediaPhoto(media=io.BytesIO(card), caption=caption)
    bot = getattr(context, "bot", None)
    if bot is not None and fix.get("chat_id") and fix.get("message_id"):
        try:
            await bot.edit_message_media(chat_id=fix["chat_id"], message_id=fix["message_id"],
                                         media=media, reply_markup=_keyboard(state.token))
            return
        except TelegramError:
            pass
    await chat.send_photo(photo=io.BytesIO(card), caption=caption, reply_markup=_keyboard(state.token))


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    key = _key(update)
    data = (query.data if query else "") or ""
    kind, _, rest = data.partition(":")
    token = rest.split(":")[0] if rest else ""
    state = _services(context).sessions.state(key, token) if key and token else None
    if not query or not query.message or state is None:
        if query:
            await query.answer("انتهت جلسة الصورة؛ ابعتها تاني.", show_alert=True)
        return
    if kind == "fix":
        await query.answer()
        context.user_data["word_fix"] = {
            "token": state.token,
            "chat_id": query.message.chat_id,
            "message_id": query.message.message_id,
            "until": time.monotonic() + 600,
        }
        await query.message.reply_text(
            "✏️ اكتب الكلمة الصحيحة كما تظهر في الصورة، وأنا أحدّث المعاينة على طول.")
        return
    if kind in {"file", "filepick"}:
        user = update.effective_user
        if user is None or user.id != owner_telegram_id():
            await query.answer("📄 إرسال ملفات الخطوط متاح لصاحب البوت فقط.", show_alert=True)
            return
        cached = getattr(state, "result", None)
        matches = list(getattr(cached, "matches", []) or [])
        if kind == "filepick":
            try:
                index = int(rest.split(":", 1)[1])
            except (IndexError, ValueError):
                await query.answer("⌛ حدّد الخط من القائمة.", show_alert=True)
                return
            if not 0 <= index < len(matches):
                await query.answer("⌛ النتيجة دي قديمة؛ ابعث الصورة تاني.", show_alert=True)
                return
            await query.answer("📄 جاري إرسال ملف الخط...")
            await _send_font_file(query, context, matches[index])
            return
        await query.answer()
        if not matches:
            await query.message.reply_text("⚠️ مفيش نتائج محفوظة للصورة دي؛ ابعتها تاني.")
            return
        choices = [(index, match) for index, match in enumerate(matches)
                   if getattr(match, "font_path", "")]
        if not choices:
            await query.message.reply_text(
                "⚠️ مفيش ملف خط موجود عندي للنتيجة دي — الأسماء دي جاية من البحث الخارجي بس. "
                "لو ضفت الخط للمكتبة هقدر أبعتهولك كملف.")
            return
        # One row per match, one short label: no giant text in a group card.
        rows = [[InlineKeyboardButton((f"{index + 1}. {match.family}"[:58]),
                                      callback_data=f"filepick:{state.token}:{index}")]
                for index, match in choices]
        await query.message.reply_text("📄 أنهي خط أبعتهولك ملفه TTF/OTF؟",
                                       reply_markup=InlineKeyboardMarkup(rows))
        return
    if kind != "vote":
        await query.answer()
        return
    action = rest.partition(":")[2]
    if action == "correct":
        await query.answer("✅ تم تأكيد النتيجة.")
        return
    if action != "wrong":
        await query.answer()
        return
    if state.owner_only:
        await query.answer("دي بالفعل نتائج مكتبة المالك. جرّب صورة أوضح أو أضف خطوطاً للمكتبة.", show_alert=True)
        return
    if state.busy:
        await query.answer("⏳ استنى نتيجة التحليل الحالي.")
        return
    await query.answer("📚 هبحث في مكتبة المالك فقط.")
    try:
        await query.message.edit_reply_markup(reply_markup=None)
    except TelegramError:
        pass
    status = await query.message.reply_text("📚 تركت نتائج المواقع؛ جاري فحص مكتبة المالك...")
    # Reuse the OCR text from the rejected card, without calling either website.
    await _analyze_and_respond(update, context, status, state.image, "auto", state.text or None,
                               state.token, owner_only=True)


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    # PTB HTTP exceptions can embed the bot-token URL; do not log their full text.
    name = type(context.error).__name__
    if name == "Conflict":
        logger.error("Telegram Conflict: another process is polling with this token; stop the duplicate bot")
    else:
        logger.error("Telegram handler error: %s", name)


PUBLIC_COMMANDS: list[tuple[str, str]] = []
ADMIN_COMMANDS = [("start", "بدء استخدام البوت"), ("help", "طريقة الاستخدام"), ("status", "حالة المكتبة والبحث")]
OWNER_COMMANDS = ADMIN_COMMANDS + [
    ("login", "كود دخول لوحة المالك"), ("owner", "حالة مكتبة المالك"),
    ("fonts", "عرض الخطوط"), ("removefont", "حذف خط بالرقم"),
    ("import_github", "استيراد مستودع خطوط"), ("jobs", "متابعة الاستيراد"),
]


async def configure_command_menus(bot) -> None:
    """Hide the menu from everyone except the owner and explicit bot admins.

    Menus are visibility, not authorization: every management handler still
    checks the authenticated Telegram sender and private chat independently.
    """
    for scope in (BotCommandScopeDefault(), BotCommandScopeAllPrivateChats(),
                  BotCommandScopeAllGroupChats(), BotCommandScopeAllChatAdministrators()):
        await bot.set_my_commands(PUBLIC_COMMANDS, scope=scope)
    # Per-chat menus are best-effort: an admin who has never opened the bot can
    # make Telegram reject that chat scope, and must not take polling down.
    try:
        await bot.set_my_commands([BotCommand(*item) for item in OWNER_COMMANDS],
                                  scope=BotCommandScopeChat(chat_id=owner_telegram_id()))
    except TelegramError:
        logger.warning("Owner command menu could not be set; owner should open/start the bot once")
    for admin_id in sorted(admin_telegram_ids()):
        try:
            await bot.set_my_commands([BotCommand(*item) for item in ADMIN_COMMANDS],
                                      scope=BotCommandScopeChat(chat_id=admin_id))
        except TelegramError:
            logger.warning("Admin command menu could not be set for %s", admin_id)


async def post_init(application: Application) -> None:
    try:
        await configure_command_menus(application.bot)
    except TelegramError:
        # Menu visibility is cosmetic; handler authorization is not. A transient
        # Telegram API failure here must not prevent polling from answering users.
        logger.warning("Telegram command menus could not be fully configured; authorization remains enforced")


def register_handlers(application: Application, services: Services) -> None:
    application.bot_data["services"] = services
    for command, handler in [("start", start), ("help", start), ("status", status_command),
                             ("login", login), ("owner", owner_status), ("fonts", list_fonts),
                             ("removefont", remove_font), ("import_github", import_github),
                             ("jobs", jobs_command), ("threads", threads_command)]:
        application.add_handler(CommandHandler(command, handler))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_error_handler(handle_error)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env or the environment")
    builder = (Application.builder().token(token).concurrent_updates(int(os.getenv("BOT_CONCURRENT_UPDATES", "16")))
               .post_init(post_init).connect_timeout(15).read_timeout(30)
               .write_timeout(30).pool_timeout(15).get_updates_read_timeout(35))
    if os.getenv("TELEGRAM_LOCAL_API", "0").lower() in {"1", "true", "yes"}:
        builder = builder.base_url(os.getenv("TELEGRAM_API_BASE_URL", "http://127.0.0.1:8081/bot")).local_mode(True)
    application = builder.build()
    register_handlers(application, Services.create())
    assets_runtime.start()
    logger.info("Englishawy polling starting; owner=%s", owner_telegram_id())
    # run_polling removes an old webhook before getUpdates. Keep queued photos
    # (drop_pending_updates=False), retry bootstrap, and allow slow mobile links.
    application.run_polling(allowed_updates=["message", "callback_query"],
                            drop_pending_updates=False, timeout=30, bootstrap_retries=-1)


if __name__ == "__main__":
    main()
