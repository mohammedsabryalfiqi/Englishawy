"""
بوت تليجرام Englishawy v3 — تحديد الخطوط محلياً بالذكاء الاصطناعي (بدون روابط)
=============================================================================
مستودع المشروع: https://github.com/mohammedsabryalfiqi/Englishawy.git

سير العمل (محلي بالكامل — لا WhatFontIs ولا DaFont ولا روابط تحميل):
  1) قراءة الحروف: نموذج Tesseract العصبي يُمرَّر على الصورة بعدة معالجات مسبقة،
     وتُدمج النتائج بالتصويت، ثم تُصحَّح الكلمات بقاموس إنجليزي ضخم.
  2) مطابقة الأشكال: يُرسم النص المقروء بمكتبة خطوط Google المحلية ويُقارن بالصورة
     من حيث شكل الحروف وسُمك الخط ونسبة العرض للارتفاع.
  3) إعادة الترتيب العصبي: نموذج الرؤية DINOv2 يعيد ترتيب أفضل 30 مرشحاً.
النتيجة: بطاقة مقارنة بصرية (اسم الخط + معاينة حيّة) مثل مثال MUQLA — بدون أي روابط.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from pipeline import Identification, identify

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
FONT_LIMIT = int(os.getenv("FONT_LIMIT", "0")) or None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# آخر صورة لكل مستخدم + وضع انتظار تصحيح النص اليدوي
USER_LAST_IMAGE: dict[int, bytes] = {}
AWAITING_TEXT: set[int] = set()


def _keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🌓 عكس الخلفية", callback_data="reanalyze:invert"),
                InlineKeyboardButton("🔲 عزل ثنائي حاد", callback_data="reanalyze:binarize"),
            ],
            [
                InlineKeyboardButton("✏️ تصحيح النص يدوياً", callback_data="correct:text"),
                InlineKeyboardButton("🔄 إعادة تلقائية", callback_data="reanalyze:auto"),
            ],
        ]
    )


def _md_escape(text: str) -> str:
    """يهرب محارف Markdown الخاصة حتى لا يفشل إرسال التعليق لو احتوى النص/الاسم عليها."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, "\\" + ch)
    return text


def _caption(result: Identification, title: str = "أقرب الخطوط (Englishawy v3)") -> str:
    lines = [
        f"🎯 *{title}*",
        f"📝 النص: *{_md_escape(result.text)}* · ثقة OCR: {result.ocr.confidence:.0f}%",
        f"🗂️ {result.font_count} خط Google · {'DINOv2 ✓' if result.neural_used else 'شكل فقط'} · {result.elapsed_s:.0f}s",
        "",
    ]
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for i, m in enumerate(result.matches, 1):
        lines.append(f"{medals.get(i, f'{i}.')} *{_md_escape(m.family)}*")
    lines += ["", "👆 معاينة حيّة لكل خط في البطاقة — بدون أي روابط تحميل."]
    return "\n".join(lines)


async def _identify_async(image_bytes: bytes, mode: str, text_override: str | None) -> Identification:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: identify(
            image_bytes,
            top_n=MAX_RESULTS,
            candidates=CANDIDATES,
            force_invert=(mode == "invert"),
            force_binarize=(mode == "binarize"),
            text_override=text_override,
            font_limit=FONT_LIMIT,
        ),
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بيك في *Englishawy v3* 👋\n"
        "بحدد لك الخط الإنجليزي من الصورة *محلياً بالذكاء الاصطناعي* — من غير أي روابط تحميل.\n\n"
        "🔬 *طريقة العمل:*\n"
        "1️⃣ قراءة الحروف بنموذج *Tesseract* العصبي على عدة معالجات + تصويت + تصحيح قاموسي.\n"
        "2️⃣ رسم النص بمكتبة *Google Fonts* ومطابقة شكل الحروف والسُمك والنسبة.\n"
        "3️⃣ إعادة ترتيب أفضل 30 مرشحاً بنموذج الرؤية *DINOv2*.\n\n"
        "📸 ابعت صورة (أو ملف صورة) فيها كلمة/جملة بخط إنجليزي، وهرد عليك بـ*بطاقة مقارنة* "
        "فيها اسم كل خط مع معاينة حيّة للكلمة بيه — زي المثال بالظبط.\n\n"
        "💡 لو الـOCR قرأ الكلمة غلط، اضغط *تصحيح النص يدوياً* واكتب الكلمة الصحيحة، "
        "أو ابعت الصورة ومعاها تعليق بالكلمة الصحيحة.",
        parse_mode="Markdown",
    )


async def _analyze_and_respond(chat, status, image_bytes: bytes, mode: str, text_override: str | None):
    await status.edit_text(
        "🔬 جاري التحليل المحلي:\n"
        "① قراءة الحروف (Tesseract + تصويت + قاموس)\n"
        "② مطابقة الأشكال على مكتبة Google Fonts\n"
        "③ إعادة الترتيب العصبي (DINOv2) ⏳"
    )
    try:
        result = await _identify_async(image_bytes, mode, text_override)
    except Exception:
        logger.exception("identification failed")
        await status.edit_text("⚠️ حصل خطأ أثناء التحليل. جرّب صورة أوضح أو مقصوصة على الكلمة.")
        return

    if not result.text or not result.matches:
        await status.edit_text(
            "⚠️ معرفتش أقرأ نصاً واضحاً من الصورة.\n"
            "جرّب الأزرار بالأسفل، أو ابعت الصورة ومعاها تعليق بالكلمة المكتوبة.",
            reply_markup=_keyboard(),
        )
        return

    try:
        await status.delete()
    except Exception:
        pass
    await chat.send_action(ChatAction.UPLOAD_PHOTO)
    await chat.send_photo(
        photo=io.BytesIO(result.card_png),
        caption=_caption(result),
        parse_mode="Markdown",
        reply_markup=_keyboard(),
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tg_file = await update.message.photo[-1].get_file()
        buf = io.BytesIO()
        await tg_file.download_to_memory(out=buf)
        image_bytes = buf.getvalue()
    except Exception:
        logger.exception("photo download failed")
        await update.message.reply_text("⚠️ معرفتش أنزّل الصورة، ابعتها تاني.")
        return
    chat_id = update.effective_chat.id
    USER_LAST_IMAGE[chat_id] = image_bytes
    AWAITING_TEXT.discard(chat_id)
    override = (update.message.caption or "").strip() or None
    status = await update.message.reply_text("🔬 استلمت الصورة، جاري التحليل... ⏳")
    await _analyze_and_respond(update.effective_chat, status, image_bytes, "auto", override)


async def handle_image_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc or not doc.mime_type or not doc.mime_type.startswith("image/"):
        return
    try:
        tg_file = await doc.get_file()
        buf = io.BytesIO()
        await tg_file.download_to_memory(out=buf)
        image_bytes = buf.getvalue()
    except Exception:
        logger.exception("document download failed")
        await update.message.reply_text("⚠️ تعذّر تنزيل الملف، ابعت الصورة تاني.")
        return
    chat_id = update.effective_chat.id
    USER_LAST_IMAGE[chat_id] = image_bytes
    AWAITING_TEXT.discard(chat_id)
    override = (update.message.caption or "").strip() or None
    status = await update.message.reply_text("🔬 استلمت ملف الصورة، جاري التحليل... ⏳")
    await _analyze_and_respond(update.effective_chat, status, image_bytes, "auto", override)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in AWAITING_TEXT and chat_id in USER_LAST_IMAGE:
        AWAITING_TEXT.discard(chat_id)
        word = update.message.text.strip()
        status = await update.message.reply_text(f"🔬 إعادة المطابقة بالنص: {word} ... ⏳")
        await _analyze_and_respond(update.effective_chat, status, USER_LAST_IMAGE[chat_id], "auto", word)
    else:
        await update.message.reply_text(
            "ابعت 📸 صورة فيها كلمة بخط إنجليزي وأنا أحدد لك الخط وأرسل بطاقة مقارنة بالاسم والمعاينة فقط."
        )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data or ""
    if data == "correct:text":
        if chat_id not in USER_LAST_IMAGE:
            await query.answer("ابعت صورة الأول.", show_alert=True)
            return
        await query.answer()
        AWAITING_TEXT.add(chat_id)
        await query.message.reply_text(
            "✏️ اكتب الكلمة/الجملة الصحيحة زي ما هي مكتوبة في الصورة بالظبط، وهعيد المطابقة بيها."
        )
        return
    await query.answer("جاري إعادة التحليل...")
    if data.startswith("reanalyze:"):
        image_bytes = USER_LAST_IMAGE.get(chat_id)
        if not image_bytes:
            await query.message.reply_text("⚠️ انتهت جلسة الصورة، ابعتها تاني.")
            return
        mode = data.split(":", 1)[1]
        status = await query.message.reply_text(f"🔄 إعادة التحليل بوضع ({mode})... ⏳")
        await _analyze_and_respond(update.effective_chat, status, image_bytes, mode, None)


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("لازم تحط TELEGRAM_BOT_TOKEN في ملف .env")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_callback))
    logger.info("Englishawy v3 running (local AI pipeline, no download links)...")
    app.run_polling()


if __name__ == "__main__":
    main()
