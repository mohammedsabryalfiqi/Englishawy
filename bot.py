"""Englishawy Telegram bot — polling locally, webhook inside the Gradio Space.

Owner identity always comes from update.effective_user.id (and a private chat for
management). Browser login codes are issued only after that server-side check.
Recognition results contain names and rendered previews, never download links.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from config import FONT_SUFFIXES, is_owner, owner_telegram_id, storage_status
from font_store import format_import_report
from pipeline import Identification, identify
from services import Services

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
FONT_LIMIT = int(os.getenv("FONT_LIMIT", "0")) or None
CLOUD_DOWNLOAD_LIMIT = 20 * 1024 * 1024
IMAGE_LIMIT = 10 * 1024 * 1024


def _services(context) -> Services:
    return context.bot_data["services"]


def _key(update: Update) -> tuple[int, int] | None:
    if not update.effective_chat or not update.effective_user:
        return None
    return update.effective_chat.id, update.effective_user.id


def _owner(update: Update) -> bool:
    chat, user = update.effective_chat, update.effective_user
    return bool(chat and user and is_owner(user.id, chat.id, chat.type))


def _keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🌓 عكس الخلفية", callback_data="reanalyze:invert"),
         InlineKeyboardButton("🔲 عزل ثنائي حاد", callback_data="reanalyze:binarize")],
        [InlineKeyboardButton("✏️ تصحيح النص يدوياً", callback_data="correct:text"),
         InlineKeyboardButton("🔄 إعادة تلقائية", callback_data="reanalyze:auto")],
    ])


def _caption(result: Identification) -> str:
    lines = ["🎯 أقرب الخطوط — Englishawy",
             f"📝 النص: {result.text} · ثقة OCR: {result.ocr.confidence:.0f}%",
             f"🗂️ {result.font_count} خط · {result.neural_backend} · {result.elapsed_s:.0f}s", ""]
    for index, match in enumerate(result.matches, 1):
        source = "مكتبتك" if match.source == "owner" else "Google"
        lines.append(f"{index}. {match.family} — {source}")
    lines += ["", "معاينة حيّة لكل خط — بدون روابط تحميل."]
    return "\n".join(lines)[:1024]


async def _identify_async(services: Services, image: bytes, mode: str, text: str | None) -> Identification:
    async with services.analysis_semaphore:
        return await asyncio.to_thread(identify, image, top_n=MAX_RESULTS, candidates=CANDIDATES,
                                       force_invert=mode == "invert", force_binarize=mode == "binarize",
                                       text_override=text, font_limit=FONT_LIMIT,
                                       fonts=services.catalog.fonts())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = ("أهلاً بيك في Englishawy 👋\n\n"
            "ابعت صورة فيها كلمة إنجليزية، وهقرأها بـTesseract (تصويت + قاموس)، "
            "وأقارن شكل الحروف والسُمك والنسبة عبر Google Fonts ومكتبة المالك، "
            "وأعيد ترتيب أفضل 30 مرشحاً بـDINOv2.\n\n"
            "هتستلم بطاقة مقارنة بالاسم والمعاينة فقط — بدون روابط تحميل.\n"
            "لو القراءة غلط، ابعت الصورة ومعاها النص الصحيح أو استخدم زر التصحيح.\n"
            "الأوامر: /help /status")
    if _owner(update):
        text += ("\n\n🔐 أوامر المالك: /login /owner /fonts /removefont\n"
                 "ابعت ZIP أو ملفات خطوط هنا لإضافتها للمكتبة. حدّ تنزيل Bot API العادي 20MB؛ "
                 "للأرشيفات الأكبر استخدم تبويب المالك في Gradio بكود /login.")
    await update.message.reply_text(text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = _services(context).catalog.stats()
    await update.message.reply_text(f"Google: {stats['google']} خط\nمكتبة المالك: {stats['owner']['faces']} وجه / {stats['owner']['families']} عائلة\n{storage_status()['label']}")


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
    await update.message.reply_text(f"🔐 المالك: {owner_telegram_id()}\nالوجوه: {stats['faces']}\nالعائلات: {stats['families']}\nالحجم: {stats['bytes']/1048576:.1f}MB\n{storage_status()['label']}\n\n/fonts [صفحة] لعرض الخطوط و/removefont ID للحذف.")


async def list_fonts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _owner(update):
        return
    page = max(1, int(context.args[0])) if context.args and context.args[0].isdigit() else 1
    entries = _services(context).store.entries(limit=20, offset=(page - 1) * 20)
    lines = [f"📚 مكتبة المالك — صفحة {page}"]
    lines += [f"{e['id']}. {e['family']} / {e['style']}" for e in entries]
    if not entries:
        lines.append("فارغة أو انتهت الصفحات. ابعت ZIP لإضافة خطوط.")
    await update.message.reply_text("\n".join(lines)[:4000])


async def remove_font(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _owner(update):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("استخدم /removefont ID — الأرقام ظاهرة في /fonts")
        return
    removed = await asyncio.to_thread(_services(context).store.delete, int(context.args[0]))
    await update.message.reply_text("✅ اتحذف الخط من البحث والمعاينة." if removed else "لم أجد هذا ID.")


async def _analyze_and_respond(update: Update, context: ContextTypes.DEFAULT_TYPE, status,
                               image: bytes, mode: str, text: str | None):
    await status.edit_text("🔬 Tesseract + تصويت وقاموس → مطابقة Google/مكتبة المالك → DINOv2 ⏳")
    try:
        result = await _identify_async(_services(context), image, mode, text)
    except Exception as exc:
        logger.error("Identification failed: %s", type(exc).__name__)
        await status.edit_text("⚠️ تعذر التحليل. تأكد من تجهيز المكتبة/النماذج أو جرّب صورة أوضح.")
        return
    if not result.text or not result.matches:
        await status.edit_text("⚠️ لم أجد نصاً/مرشحين واضحين. جرّب التصحيح اليدوي أو قصّ الكلمة.", reply_markup=_keyboard())
        return
    try:
        await status.delete()
    except Exception:
        pass
    chat = update.effective_chat
    await chat.send_action(ChatAction.UPLOAD_PHOTO)
    await chat.send_photo(photo=io.BytesIO(result.card_png), caption=_caption(result), reply_markup=_keyboard())


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
    _services(context).sessions.put(key, image)
    override = (update.message.caption or "").strip() or None
    status = await update.message.reply_text("🔬 استلمت الصورة، جاري التحليل...")
    await _analyze_and_respond(update, context, status, image, "auto", override)


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
        await update.message.reply_text("ابعت صورة لتحديد الخط، أو ZIP/خطوط من حساب المالك لإضافتها.")


async def login_hint(update: Update):
    await update.message.reply_text("📦 Telegram Bot API العادي يسمح بتنزيل ملفات حتى 20MB فقط.\n"
                                    "لرفع ZIP بحجم 200MB أو أكبر: اطلب /login هنا، وافتح تبويب «مكتبة المالك» في Gradio وأدخل الكود. "
                                    "الرفع هناك بلا حدّ حجم مفروض في التطبيق، حسب سعة التخزين والمنصة.\n"
                                    "البديل: شغّل Local Bot API server واضبط TELEGRAM_LOCAL_API=1 و--local في البوت المحلي.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    key = _key(update)
    services = _services(context)
    if key and services.sessions.is_awaiting(key):
        services.sessions.awaiting(key, False)
        word = update.message.text.strip()[:160]
        status = await update.message.reply_text(f"🔬 إعادة المطابقة بالنص: {word}")
        await _analyze_and_respond(update, context, status, services.sessions.get(key), "auto", word)
    else:
        await update.message.reply_text("ابعت 📸 صورة فيها كلمة إنجليزية، وهرد باسم الخط ومعاينته فقط.")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    key = _key(update)
    image = _services(context).sessions.get(key) if key else None
    if not query or not query.message or image is None:
        if query:
            await query.answer("انتهت جلسة الصورة؛ ابعتها تاني.", show_alert=True)
        return
    data = query.data or ""
    if data == "correct:text":
        _services(context).sessions.awaiting(key, True)
        await query.answer()
        await query.message.reply_text("✏️ اكتب النص الصحيح كما يظهر في الصورة، وهعيد المطابقة.")
    elif data in {"reanalyze:invert", "reanalyze:binarize", "reanalyze:auto"}:
        await query.answer("جاري إعادة التحليل...")
        status = await query.message.reply_text("🔄 إعادة التحليل...")
        await _analyze_and_respond(update, context, status, image, data.split(":", 1)[1], None)
    else:
        await query.answer()


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    # PTB HTTP exceptions can embed the bot-token URL; do not log their full text.
    logger.error("Telegram handler error: %s", type(context.error).__name__)


def register_handlers(application: Application, services: Services) -> None:
    application.bot_data["services"] = services
    for command, handler in [("start", start), ("help", start), ("status", status_command),
                             ("login", login), ("owner", owner_status), ("fonts", list_fonts),
                             ("removefont", remove_font)]:
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
    builder = Application.builder().token(token).concurrent_updates(4)
    if os.getenv("TELEGRAM_LOCAL_API", "0").lower() in {"1", "true", "yes"}:
        builder = builder.base_url(os.getenv("TELEGRAM_API_BASE_URL", "http://127.0.0.1:8081/bot")).local_mode(True)
    application = builder.build()
    register_handlers(application, Services.create())
    logger.info("Englishawy polling bot started; owner=%s", owner_telegram_id())
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
