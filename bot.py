"""
بوت تليجرام: ابعتله صورة فيها خط انجليزي، وهو:
1. يبعتها لـ WhatFontIs API عشان يحدد أقرب الخطوط المطابقة (تحليل شكل الحروف).
2. يدور على كل خط منهم في dafont.com.
3. يرجعلك النتائج مع روابط تحميل/صفحة الخط.

تشغيل:
    python bot.py
"""

import io
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from whatfontis import identify_font
from dafont import search_dafont

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WHATFONTIS_API_KEY = os.getenv("WHATFONTIS_API_KEY")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بيك! 👋\n\n"
        "ابعتلي صورة واضحة فيها كلمة أو جملة بالخط اللي عايز تعرف اسمه، "
        "وهجيبلك أقرب الخطوط المطابقة مع روابط تحميلها من dafont.com.\n\n"
        "عشان تطلع نتيجة أدق:\n"
        "• الصورة واضحة وغير ضبابية وبدقة كويسة\n"
        "• خلفية بسيطة وتباين واضح بين لون الحروف والخلفية\n"
        "• الحروف مش ملزّقة في بعض (خصوصًا لو الخط كرسيف/يدوي)\n"
        "• قصّ الصورة عشان تركز بس على الكلمة/الجملة"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not WHATFONTIS_API_KEY:
        await update.message.reply_text(
            "لسه محتاج تحط WHATFONTIS_API_KEY في ملف .env عشان أقدر أحلل الصور.\n"
            "تقدر تجيب المفتاح ده من حسابك على whatfontis.com."
        )
        return

    status_msg = await update.message.reply_text("جاري تحليل الصورة... ⏳")

    try:
        photo = update.message.photo[-1]  # أعلى دقة متاحة
        tg_file = await photo.get_file()
        buf = io.BytesIO()
        await tg_file.download_to_memory(out=buf)
        image_bytes = buf.getvalue()
    except Exception:
        logger.exception("Failed to download photo from Telegram")
        await status_msg.edit_text("معرفتش أنزّل الصورة من تليجرام، جرب تبعتها تاني.")
        return

    try:
        matches = identify_font(image_bytes, WHATFONTIS_API_KEY, limit=MAX_RESULTS)
    except Exception as e:
        logger.exception("WhatFontIs API error")
        await status_msg.edit_text(f"⚠️ حصل خطأ أثناء تحليل الخط:\n{e}")
        return

    if not matches:
        await status_msg.edit_text(
            "معرفتش أوصل لنتيجة مضمونة. جرب صورة أوضح، أو قصّ الصورة على الكلمة بس، "
            "أو تأكد إن الحروف مش ملزّقة في بعض."
        )
        return

    lines = ["🔎 أقرب الخطوط المطابقة:\n"]

    for i, m in enumerate(matches[:MAX_RESULTS], start=1):
        title = m.get("title", "غير معروف")
        wfi_url = m.get("url", "")
        font_type = m.get("type", "")

        df = search_dafont(title)

        line = f"{i}. {title}"
        if font_type:
            line += f"  [{font_type}]"
        lines.append(line)

        if df["download_url"]:
            lines.append(f"   ⬇️ تحميل مباشر من dafont: {df['download_url']}")
        elif df["font_page"]:
            lines.append(f"   🔗 صفحة الخط على dafont: {df['font_page']}")
        else:
            lines.append(f"   🔍 غير موجود بنفس الاسم بالظبط، جرب البحث هنا: {df['search_url']}")

        if wfi_url:
            lines.append(f"   ℹ️ معاينة/تفاصيل إضافية: {wfi_url}")

        lines.append("")

    lines.append(
        "ملحوظة: بعض الخطوط في dafont للاستخدام الشخصي فقط وتحتاج ترخيص "
        "للاستخدام التجاري — راجع الترخيص المكتوب فوق زرار التحميل في صفحة كل خط."
    )

    await status_msg.edit_text("\n".join(lines), disable_web_page_preview=True)


async def handle_image_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """لو المستخدم بعت الصورة كملف (document) مش كصورة عادية."""
    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        return
    await update.message.reply_text(
        "من فضلك ابعت الصورة كـ 'صورة/Photo' عادية (مش كملف/Document) عشان أقدر أحللها بشكل صحيح."
    )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("لازم تحط TELEGRAM_BOT_TOKEN في ملف .env")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image_document))

    logger.info("البوت شغال دلوقتي (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
