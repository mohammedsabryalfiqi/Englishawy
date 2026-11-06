"""
بوت تليجرام للتعرف الدقيق على الخطوط الإنجليزية — Englishawy v2.4
==================================================================
مستودع المشروع: https://github.com/mohammedsabryalfiqi/Englishawy.git

التطويرات الجذرية في الإصدار الثاني (v2.4):
1. معالجة بصرية مسبقة للصور (preprocessor.py):
   - اكتشاف الخلفيات الداكنة وعكسها تلقائياً (لأن محركات WhatTheFont و WhatFontIs تحتاج حروفاً داكنة على خلفية فاتحة).
   - قص تلقائي للهوامش الفارغة حول الكلمة (Auto-Crop Bounding Box) وزيادة حدة الحواف (Contrast & Sharpness).
2. التخزين الداخلي والخارجي للخطوط (external_vault.py):
   - مطابقة فورية مع قاعدة بيانات الخطوط الخاصة بك المخزنة على سيرفر خارجي (fonts_vault.json + External CDN/Server)
     مع إرجاع رابط تحميل مباشر لملف الخط (.ttf / .otf / .zip).
3. محرك WhatFontIs API v2 مزدوج المحاولة (whatfontis.py):
   - يعيد المحاولة تلقائياً باستخدام NOTTEXTBOXSDETECTION=1 إذا فشل الكشف القياسي للصندوق.
4. باحث DaFont الذكي مع كاش محلي SQLite وتطبيع الأسماء (dafont.py):
   - ينظف اسم الخط من بادئات المسابك (NMY_, FF_, GOOGLE_) والأوزان (Regular, Bold, Pro) قبل البحث في DaFont.
5. أزرار تفاعلية (Inline Keyboard) تحت كل نتيجة لإعادة التحليل بحساسية فائقة أو عكس الخلفية أو الثنائية الحادة.
"""

import io
import logging
import os

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from preprocessor import preprocess_font_image
from whatfontis import identify_font_with_fallback
from dafont import search_dafont_cached, normalize_font_name
from external_vault import ExternalFontVault

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
WHATFONTIS_API_KEY = os.getenv("WHATFONTIS_API_KEY", "").strip()
EXTERNAL_FONT_SERVER_URL = os.getenv(
    "EXTERNAL_FONT_SERVER_URL", "https://cdn.englishawy-fonts.io/vault/v2"
).strip()
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# تهيئة مستودع الخطوط الداخلي/الخارجي
vault = ExternalFontVault(
    vault_json_path=os.path.join(os.path.dirname(__file__), "fonts_vault.json"),
    external_base_url=EXTERNAL_FONT_SERVER_URL,
)

# كاش مؤقت لآخر صورة أرسلها كل مستخدم لدعم أزرار إعادة التحليل بدون إعادة رفع الصورة
USER_LAST_IMAGE: dict[int, bytes] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً بيك في *Englishawy v2.4*! 👋\n\n"
        "ابعتلي صورة (أو ملف صورة) فيها كلمة أو جملة بخط إنجليزي، وهقوم بـ:\n"
        "1️⃣ *معالجة الصورة تلقائياً:* عزل الحروف، ضبط التباين، وعكس الخلفيات الغامقة.\n"
        "2️⃣ *تحليل شكل الحروف (Glyph Matching):* عبر محرك WhatFontIs API v2 + البصمة الهندسية.\n"
        "3️⃣ *التحميل المباشر:* البحث في *سيرفر الخطوط الخارجي الخاص بنا* أولاً، ثم *DaFont* (مع تطبيع ذكي للاسم وكاش سريع).\n\n"
        "💡 *نصائح لأعلى دقة (99%):*\n"
        "• قصّ الصورة على الكلمة الإنجليزية فقط.\n"
        "• لو النتيجة محتاجه دقة أعلى، استخدم الأزرار التفاعلية تحت الرد لإعادة الفحص بحساسية فائقة!",
        parse_mode="Markdown",
    )


async def analyze_and_send_response(
    chat_id: int,
    status_msg,
    raw_image_bytes: bytes,
    mode: str = "auto",
):
    """
    mode:
      - 'auto': معالجة ذكية تلقائية + كشف تلقائي للخلفية الداكنة
      - 'high_sens': تفعيل NOTTEXTBOXSDETECTION=1 للحروف المعقدة أو المتلاصقة
      - 'invert': إجبار عكس ألوان الصورة (للخلفيات الملونة أو المقلوبة)
      - 'binarize': تحويل الصورة إلى أبيض وأسود حاد جداً (Otsu Threshold)
    """
    force_invert = mode == "invert"
    force_binarize = mode == "binarize"
    force_high_sens = mode == "high_sens"

    # 1. المعالجة المسبقة للصورة واستخراج البصمة الهندسية (8D Glyph Features)
    prep_result = preprocess_font_image(
        raw_image_bytes,
        force_invert=force_invert,
        force_binarize=force_binarize,
    )
    processed_bytes = prep_result["image_bytes"]
    extracted_vector = prep_result["glyph_vector"]
    prep_info = prep_result["summary"]

    # 2. استدعاء WhatFontIs API v2 (مع التبديل التلقائي إلى NOTTEXTBOXSDETECTION=1 عند الحاجة)
    wfi_matches = []
    wfi_error_note = None
    if WHATFONTIS_API_KEY:
        try:
            wfi_matches = identify_font_with_fallback(
                processed_image_bytes=processed_bytes,
                raw_image_bytes=raw_image_bytes,
                api_key=WHATFONTIS_API_KEY,
                limit=max(MAX_RESULTS, 8),
                force_no_textbox=force_high_sens,
            )
        except Exception as exc:
            logger.warning("WhatFontIs warning: %s", exc)
            wfi_error_note = str(exc)

    # 3. دمج النتائج مع مستودع الخطوط الداخلي/السيرفر الخارجي + DaFont المطبّع
    combined_results = vault.match_and_enrich(
        whatfontis_matches=wfi_matches,
        extracted_vector=extracted_vector,
        dafont_search_fn=search_dafont_cached,
        limit=MAX_RESULTS,
    )

    if not combined_results:
        await status_msg.edit_text(
            "⚠️ معرفتش أوصل لنتيجة مضمونة من الصورة دي.\n"
            "جرب تضغط على زرار *حساسية فائقة* أو *عكس ألوان الخلفية* بالأسفل، أو ابعت صورة مقصوصة على الكلمة بس.",
            parse_mode="Markdown",
        )
        return

    lines = [
        "🎯 *أقرب الخطوط المطابقة (Englishawy v2.4):*",
        f"⚙️ المعالجة البصرية: `{prep_info}`",
        "",
    ]

    for idx, item in enumerate(combined_results[:MAX_RESULTS], start=1):
        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
        title = item["title"]
        clean_family = item["clean_family"]
        confidence = item["confidence"]
        category = item.get("category", "English Font")
        font_type = item.get("license_type", "")

        header = f"{medal} *{title}* — دقة *{confidence}%*"
        if font_type:
            header += f"  `[{font_type}]`"
        lines.append(header)

        if item.get("key_glyph_notes"):
            lines.append(f"   🧬 البصمة: {item['key_glyph_notes']}")

        # أولاً: رابط التحميل المباشر من السيرفر الخارجي (إن وجد في المستودع)
        if item.get("external_server_url"):
            lines.append(
                f"   ☁️ *تحميل مباشر من السيرفر الخارجي ({item.get('file_format', 'TTF')}):*\n"
                f"   {item['external_server_url']}"
            )

        # ثانياً: رابط التحميل من DaFont (بعد تطبيع الاسم)
        df = item.get("dafont", {})
        if df.get("download_url"):
            lines.append(f"   ⬇️ تحميل مباشر من DaFont: {df['download_url']}")
        elif df.get("font_page"):
            lines.append(f"   🔗 صفحة الخط على DaFont: {df['font_page']}")
        elif df.get("search_url"):
            lines.append(f"   🔍 بحث بالاسم المطبّع ({clean_family}) في DaFont: {df['search_url']}")

        # ثالثاً: رابط المعاينة على WhatFontIs
        if item.get("whatfontis_url"):
            lines.append(f"   ℹ️ تفاصيل الخط على WhatFontIs: {item['whatfontis_url']}")

        lines.append("")

    if wfi_error_note:
        lines.append(f"ℹ️ ملاحظة محرك API: {wfi_error_note}\n")

    lines.append(
        "⚖️ *تنبيه الترخيص:* بعض الخطوط مجانية للاستخدام الشخصي فقط وتحتاج ترخيصاً للاستخدام التجاري."
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔄 حساسية فائقة (NOTTEXTBOX=1)",
                    callback_data="reanalyze:high_sens",
                ),
                InlineKeyboardButton(
                    "🌓 عكس ألوان الخلفية",
                    callback_data="reanalyze:invert",
                ),
            ],
            [
                InlineKeyboardButton(
                    "🔲 عزل حاد للحروف (Binarize)",
                    callback_data="reanalyze:binarize",
                ),
            ],
        ]
    )

    await status_msg.edit_text(
        "\n".join(lines),
        parse_mode="Markdown",
        disable_web_page_preview=True,
        reply_markup=keyboard,
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text(
        "🔬 جاري عزل الحروف ومطابقة الخط مع السيرفر الخارجي و WhatFontIs... ⏳"
    )

    try:
        photo = update.message.photo[-1]
        tg_file = await photo.get_file()
        buf = io.BytesIO()
        await tg_file.download_to_memory(out=buf)
        image_bytes = buf.getvalue()
        USER_LAST_IMAGE[update.effective_chat.id] = image_bytes
    except Exception:
        logger.exception("Failed to download photo from Telegram")
        await status_msg.edit_text("⚠️ معرفتش أنزّل الصورة من تليجرام، جرب تبعتها تاني.")
        return

    await analyze_and_send_response(
        chat_id=update.effective_chat.id,
        status_msg=status_msg,
        raw_image_bytes=image_bytes,
        mode="auto",
    )


async def handle_image_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """في النسخة المطورة v2 نحلل ملفات الصور (Document) مباشرة بدل رفضها!"""
    doc = update.message.document
    if not doc or not doc.mime_type or not doc.mime_type.startswith("image/"):
        return

    status_msg = await update.message.reply_text(
        "🔬 تم استلام ملف الصورة بنجاح! جاري عزل الحروف وتحليل الخط... ⏳"
    )
    try:
        tg_file = await doc.get_file()
        buf = io.BytesIO()
        await tg_file.download_to_memory(out=buf)
        image_bytes = buf.getvalue()
        USER_LAST_IMAGE[update.effective_chat.id] = image_bytes
    except Exception:
        logger.exception("Failed to download image document")
        await status_msg.edit_text("⚠️ تعذر تنزيل ملف الصورة، جرب إرسالها مرة أخرى.")
        return

    await analyze_and_send_response(
        chat_id=update.effective_chat.id,
        status_msg=status_msg,
        raw_image_bytes=image_bytes,
        mode="auto",
    )


async def handle_reanalyze_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("جاري إعادة فحص الصورة بالإعدادات المتقدمة...")

    chat_id = query.message.chat_id
    image_bytes = USER_LAST_IMAGE.get(chat_id)
    if not image_bytes:
        await query.edit_message_text("⚠️ انتهت جلسة الصورة المؤقتة، من فضلك أرسل الصورة مرة أخرى.")
        return

    mode = query.data.split(":")[-1]
    await query.edit_message_text(f"🔬 جاري إعادة التحليل بوضع ({mode})... ⏳")
    await analyze_and_send_response(
        chat_id=chat_id,
        status_msg=query.message,
        raw_image_bytes=image_bytes,
        mode=mode,
    )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("لازم تحط TELEGRAM_BOT_TOKEN في ملف .env")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image_document))
    app.add_handler(CallbackQueryHandler(handle_reanalyze_callback, pattern="^reanalyze:"))

    logger.info("Englishawy v2.4 Bot is running (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
