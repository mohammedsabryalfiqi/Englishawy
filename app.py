"""Hugging Face Gradio/ZeroGPU entry point; Telegram defaults to long polling.

Local CPU preview: python app.py (one process serves Gradio and polls Telegram).
TELEGRAM_MODE=webhook explicitly enables the optional /telegram/webhook route.
Never run a second consumer with the same bot token. No application ZIP limit.
"""
from __future__ import annotations

import io
from contextlib import AsyncExitStack, asynccontextmanager
import json
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_SSR_MODE", "False")

import gradio as gr
from PIL import Image

import assets_runtime
import neural_reranker
import zero_gpu
from config import UPLOAD_SUFFIXES, owner_telegram_id, storage_status
from font_store import format_import_report
from pipeline import identify
from services import Services
from telegram_webhook import TelegramBridge, describe_telegram_error, mount_telegram
from app_github_addon import bind_services, github_job_status, register_github_tab, start_github_import

logger = logging.getLogger(__name__)
services = Services.create()
bind_services(services)  # share ONE Services instance with the GitHub tab (no re-import of app.py)
try:
    bridge = TelegramBridge.from_env(services)
    bridge_error = ""
except ValueError as exc:
    bridge, bridge_error = None, str(exc)
if bridge:
    logger.info("Telegram token detected; %s will start with Gradio.", bridge.mode)
else:
    logger.warning("Telegram bot disabled: %s", bridge_error or "TELEGRAM_BOT_TOKEN is not set in Space secrets")

SOURCE_OPTIONS = {"Google + مكتبتي": "all", "Google Fonts فقط": "google", "مكتبتي فقط": "owner"}
BACKEND_OPTIONS = {"تلقائي (ZeroGPU إن توفر)": "auto", "CPU / ONNX": "cpu", "ZeroGPU فقط": "gpu"}
CSS = """
.gradio-container {max-width: 1280px !important; margin: auto !important; direction: rtl;}
.english-hero {direction:rtl; padding:30px 32px; border-radius:22px; margin:12px 0 22px;
 background:linear-gradient(120deg,#0d1626 0%,#123b43 100%); color:#f4fffd; border:1px solid #2dd4bf55;}
.english-hero h1 {margin:0 0 8px; font-size:34px; font-weight:800; color:#fff;}
.english-hero p {margin:0; color:#c5e5e2; line-height:1.9; font-size:15px;}
.english-kicker {direction:ltr; text-align:right; letter-spacing:3px; color:#5eead4!important; font-size:11px!important; margin-bottom:12px!important;}
.english-note {padding:14px 18px; border:1px solid #2dd4bf55; border-radius:12px; line-height:1.9;}
button.primary {background:linear-gradient(110deg,#0f766e,#14b8a6)!important; border:0!important;}
footer {display:none!important;}
"""


def _library_table(query: str = "") -> list[list]:
    rows = []
    for entry in services.store.entries(query=query, limit=200):
        try:
            analysis = json.loads(entry["analysis_json"])
            coverage = f"{len(analysis.get('latin_coverage', ''))}/62"
        except (ValueError, TypeError):
            coverage = "—"
        rows.append([entry["id"], entry["family"], entry["style"], coverage, entry["origin"]])
    return rows


def _library_stats() -> str:
    stats = services.catalog.stats()
    owner = stats["owner"]
    return (f"### المكتبة\n**Google:** {stats['google']} خط  \n"
            f"**المالك:** {owner['faces']} وجه، {owner['families']} عائلة، {owner['bytes']/1048576:.1f} MB")


def runtime_status() -> str:
    asset = assets_runtime.snapshot()
    storage = storage_status()
    if bridge:
        bot = bridge.status()
        transport = "Polling" if bot.get("mode") == "polling" else "Webhook"
        bot_text = f"متصل عبر {transport}" if bot["connected"] else f"غير متصل: {bot['error'] or 'في انتظار البدء'}"
    else:
        bot_text = bridge_error or "غير مُعدّ: أضف TELEGRAM_BOT_TOKEN"
    progress = f"{asset['done']}/{asset['total']}" if asset["total"] else "—"
    return "\n\n".join([
        f"**المحرك العصبي:** {zero_gpu.status()}",
        f"**تجهيز الأصول:** {asset['phase']} ({progress})" + (f" — {asset['error']}" if asset["error"] else ""),
        f"**بوت تليجرام:** {bot_text}",
        f"**التخزين:** {storage['label']}  \n`{storage['path']}`",
        _library_stats(),
        "رفع ZIP لا يضع حدّ حجم في التطبيق. السعة الفعلية للقرص/المنصة والشبكة هي الحدّ. "
        "تليجرام Cloud Bot API لا يستطيع تنزيل ملفات أكبر من 20MB؛ استخدم هذه اللوحة للملفات الكبيرة.",
    ])


def analyze_image(image, text_override, source_label, top_n, backend_label, invert, binarize, progress=gr.Progress()):
    if image is None:
        raise gr.Error("ارفع صورة أو قصّ الكلمة أولاً.")
    source = SOURCE_OPTIONS.get(source_label, "all")
    fonts = services.catalog.fonts(source)
    if not fonts:
        raise gr.Error("المكتبة المختارة غير جاهزة بعد. انتظر تنزيل Google أو أضف خطوط المالك.")
    backend_choice = BACKEND_OPTIONS.get(backend_label, "auto")
    if backend_choice == "gpu" and not zero_gpu.ready():
        raise gr.Error("ZeroGPU غير مهيأ. اختر الوضع التلقائي أو CPU في المعاينة المحلية.")
    backend = zero_gpu.similarities if zero_gpu.ready() and backend_choice in {"auto", "gpu"} else None
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    progress(0.05, desc="قراءة الحروف: Tesseract + تصويت + قاموس")
    def shape_progress(done, total):
        progress(0.15 + 0.60 * done / max(total, 1), desc=f"مطابقة الأشكال والسُمك: {done}/{total}")
    try:
        result = identify(buffer.getvalue(), top_n=int(top_n), candidates=30,
                          force_invert=bool(invert), force_binarize=bool(binarize),
                          text_override=(text_override or "").strip() or None,
                          fonts=fonts, neural_backend=backend, progress=shape_progress)
    except Exception as exc:
        logger.error("Gradio identification failed: %s", type(exc).__name__)
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "تعذر التحليل؛ جرّب صورة أوضح أو تحقق من النماذج."
        if backend is not None:
            message += " إن انتهت حصة ZeroGPU، انتظر إعادة ضبطها أو اختر CPU."
        raise gr.Error(message[:500]) from exc
    if not result.matches:
        raise gr.Error("لم أجد مرشحين. صحّح النص يدوياً أو اختر مصدراً يحتوي خطوطاً جاهزة.")
    progress(0.95, desc="رسم بطاقة المقارنة")
    rows = []
    for index, match in enumerate(result.matches, 1):
        rows.append([index, match.family, ("WhatTheFont" if str(match.source).endswith("WhatTheFont") else ("WhatFontIs" if str(match.source).endswith("WhatFontIs") else ("مكتبة المالك" if match.source == "owner" else "Google Fonts"))),
                     round(getattr(match, "final_score", match.score) * 100, 1),
                     match.neural_sim if match.neural_sim is not None else "—"])
    note = (f"✅ قرأت: **{result.text}** · ثقة OCR: **{result.ocr.confidence:.0f}%** · "
            f"{result.font_count} خط · {result.elapsed_s:.1f} ثانية · {result.neural_backend} · المصدر الفائز: {getattr(result, 'winning_stage', 'owner_library')}")
    if not result.neural_used:
        note += "\n\n⚠️ الترتيب شكلي فقط؛ نموذج DINOv2 غير متاح في هذا الوضع."
    return Image.open(io.BytesIO(result.card_png)).copy(), rows, result.text, note


def owner_login(code, token, request: gr.Request):
    try:
        client = request.client.host if request and request.client else "unknown"
        new_token = services.auth.exchange(code or "", client)
    except PermissionError as exc:
        return token, f"🔒 {exc}", gr.update(), gr.update()
    return new_token, "🔓 تم التحقق من المالك. يمكنك رفع الخطوط وإدارتها.", gr.update(visible=False), gr.update(visible=True)


def owner_logout(token):
    services.auth.logout(token)
    return "", "🔒 تم تسجيل الخروج. اطلب كوداً جديداً بأمر /login.", gr.update(visible=True), gr.update(visible=False)


def _safe_upload_path(value) -> Path:
    raw = Path(str(value))
    if raw.is_symlink():
        raise ValueError("الروابط الرمزية غير مسموحة")
    path = raw.resolve(strict=True)
    if path.suffix.lower() not in UPLOAD_SUFFIXES or not path.is_file():
        raise ValueError("صيغة أو مسار ملف غير مسموح")
    roots = [Path(tempfile.gettempdir()).resolve()]
    if os.getenv("GRADIO_TEMP_DIR"):
        roots.append(Path(os.environ["GRADIO_TEMP_DIR"]).resolve())
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError("يجب رفع الملف عبر واجهة Gradio")
    return path


def import_owner_files(files, token, progress=gr.Progress()):
    try:
        services.auth.require(token)
    except PermissionError as exc:
        raise gr.Error(str(exc)) from exc
    if not files:
        raise gr.Error("اختر ZIP أو ملفات خطوط أولاً.")
    if isinstance(files, (str, Path)):
        files = [files]
    reports, errors = [], []
    for index, value in enumerate(files):
        try:
            path = _safe_upload_path(value)
            progress(index / len(files), desc=f"فحص {path.name[:60]}")
            if path.suffix.lower() == ".zip":
                def archive_progress(done, total, name):
                    progress((index + done / max(total, 1)) / len(files), desc=f"فحص الخطوط: {done}/{total}")
                reports.append(format_import_report(services.store.import_zip(path, progress=archive_progress)))
            else:
                result = services.store.install_font(path, origin=path.name)
                reports.append(f"✅ {path.name}: وجوه جديدة {result['added']}، مكررة {result['duplicates']}")
        except Exception as exc:
            logger.warning("Owner import rejected: %s", type(exc).__name__)
            errors.append(f"⚠️ {Path(str(value)).name}: {str(exc)[:220]}")
    progress(1.0, desc="تم تحديث فهرس البحث")
    message = "\n\n".join(reports + errors) or "لم يتم استيراد أي خط."
    message += "\n\n" + storage_status()["label"]
    return message, _library_table(), _library_stats(), None


def refresh_owner_library(token, query=""):
    services.auth.require(token)
    return _library_table(query), _library_stats()


def delete_owner_font(font_id, token, query=""):
    services.auth.require(token)
    try:
        removed = services.store.delete(int(font_id))
    except (TypeError, ValueError) as exc:
        raise gr.Error("أدخل رقم ID صحيحاً من الجدول.") from exc
    return ("✅ تم حذف الخط." if removed else "لم أجد هذا ID."), _library_table(query), _library_stats()


class EnglishawyBlocks(gr.Blocks):
    """Gradio replaces Blocks.app inside launch(); inject our lifespan into that new app.

    Registering routes on `demo.app` before launch is lost when Gradio creates the
    server app. The HF Gradio runner calls `demo.launch()`, so this override is the
    reliable place to attach Telegram/health and run startup on the *active* app.
    """

    def launch(self, *args, **kwargs):
        app_kwargs = dict(kwargs.pop("app_kwargs", None) or {})
        app_kwargs["lifespan"] = make_lifespan(app_kwargs.get("lifespan"))
        kwargs["app_kwargs"] = app_kwargs
        return super().launch(*args, **kwargs)


def build_ui() -> gr.Blocks:
    with EnglishawyBlocks(title="Englishawy — كاشف الخطوط", theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=CSS) as demo:
        gr.HTML("""<div class="english-hero"><p class="english-kicker">ENGLISHAWY · FONT INTELLIGENCE · ZEROGPU</p>
        <h1>اعرف الخط. وشوف أقرب البدائل.</h1><p>من صورة واحدة إلى بطاقة مقارنة حيّة: Tesseract، البحث أولاً في WhatTheFont ثم WhatFontIs، ثم مكتبة المالك تلقائياً عند الحاجة، ثم DINOv2. أسماء ومعاينات فقط — بدون روابط تحميل.</p></div>""")
        with gr.Tabs():
            with gr.Tab("🔎 تحديد الخط"):
                with gr.Row():
                    with gr.Column(scale=4):
                        image_input = gr.Image(type="pil", label="الصورة أو قصّ الكلمة", height=300, show_download_button=False)
                        text_input = gr.Textbox(label="تصحيح النص (اختياري)", placeholder="اكتب الكلمة كما تظهر إذا أخطأ OCR", max_lines=2)
                        source_input = gr.Dropdown(choices=list(SOURCE_OPTIONS), value="Google + مكتبتي", label="مصدر البحث")
                        with gr.Accordion("خيارات متقدمة", open=False):
                            top_input = gr.Slider(1, 10, value=5, step=1, label="عدد النتائج في البطاقة")
                            backend_input = gr.Dropdown(choices=list(BACKEND_OPTIONS), value="تلقائي (ZeroGPU إن توفر)", label="المحرك العصبي")
                            invert_input = gr.Checkbox(label="عكس ألوان الخلفية")
                            binary_input = gr.Checkbox(label="عزل ثنائي حاد")
                        analyze_button = gr.Button("حلّل الصورة واعرض المقارنة", variant="primary")
                    with gr.Column(scale=6):
                        result_note = gr.Markdown("ارفع صورة واضحة، واضغط زر التحليل. أفضل 30 مرشحاً تُعاد ترتيبها بـDINOv2.")
                        card_output = gr.Image(type="pil", label="بطاقة المقارنة البصرية", show_download_button=False, interactive=False)
                        table_output = gr.Dataframe(headers=["الترتيب", "الخط", "المصدر", "درجة الترتيب", "DINOv2 cosine"],
                                                    datatype=["number", "str", "str", "number", "number"],
                                                    label="النتائج — درجة ترتيب نسبية وليست احتمال صحة", interactive=False, wrap=True)
                        ocr_output = gr.Textbox(label="النص المستخدم في المطابقة", interactive=False)
                analyze_button.click(analyze_image, inputs=[image_input, text_input, source_input, top_input, backend_input, invert_input, binary_input],
                                     outputs=[card_output, table_output, ocr_output, result_note], concurrency_limit=1)
            with gr.Tab("🔐 مكتبة المالك"):
                gr.Markdown(f"المالك المعرّف برقم تليجرام `{owner_telegram_id()}` فقط. افتح محادثتك الخاصة مع البوت وأرسل **/login**، ثم أدخل الكود هنا. لا نثق برقم يدخله المتصفح.")
                with gr.Group() as login_group:
                    code_input = gr.Textbox(label="كود الدخول من تليجرام", type="password", max_lines=1)
                    login_button = gr.Button("تحقق وافتح المكتبة", variant="primary")
                auth_status = gr.Markdown("🔒 لوحة الإدارة مقفلة.")
                token_state = gr.State("")
                with gr.Group(visible=False) as owner_group:
                    gr.Markdown("### أضف مجموعة خطوط\nZIP أو TTF / OTF / TTC / WOFF / WOFF2. نفحص كل وجه، نحوّل WOFF عند الحاجة، ونمنع التكرار. **لا يوجد حدّ حجم ZIP في التطبيق**؛ 200MB مقبول إذا سمحت المنصة والتخزين.")
                    file_input = gr.File(label="ملفات الخطوط أو أرشيفات ZIP", file_count="multiple", type="filepath",
                                         file_types=sorted(UPLOAD_SUFFIXES), height=180)
                    import_button = gr.Button("فكّ الضغط، ثبّت الخطوط وحدّث البحث", variant="primary")
                    import_status = gr.Markdown()
                    library_summary = gr.Markdown(_library_stats())
                    with gr.Row():
                        search_input = gr.Textbox(label="ابحث في مكتبتك", scale=4)
                        refresh_button = gr.Button("تحديث القائمة", scale=1)
                        logout_button = gr.Button("خروج", scale=1)
                    library_table = gr.Dataframe(headers=["ID", "العائلة", "الوزن", "الحروف اللاتينية", "المصدر"],
                                                 label="أول 200 وجه — استخدم البحث لباقي المكتبة", interactive=False, wrap=True)
                    with gr.Row():
                        delete_input = gr.Textbox(label="ID الخط للحذف")
                        delete_button = gr.Button("احذف الخط", variant="stop")
                    delete_status = gr.Markdown()
                    github_url, github_branch, github_start, github_status, github_jobs = register_github_tab(owner_group)
                login_button.click(owner_login, inputs=[code_input, token_state], outputs=[token_state, auth_status, login_group, owner_group])
                import_button.click(import_owner_files, inputs=[file_input, token_state], outputs=[import_status, library_table, library_summary, file_input], concurrency_limit=1)
                refresh_button.click(refresh_owner_library, inputs=[token_state, search_input], outputs=[library_table, library_summary])
                logout_button.click(owner_logout, inputs=token_state, outputs=[token_state, auth_status, login_group, owner_group])
                delete_button.click(delete_owner_font, inputs=[delete_input, token_state, search_input], outputs=[delete_status, library_table, library_summary])
                github_start.click(start_github_import, inputs=[github_url, github_branch, token_state], outputs=[github_status, github_jobs], concurrency_limit=1)
            with gr.Tab("⚙️ الحالة والتشغيل"):
                status_output = gr.Markdown(runtime_status)
                status_button = gr.Button("تحديث الحالة")
                gr.Markdown("""<div class="english-note"><b>كيف يعمل؟</b><br>
                ① Tesseract بعدة معالجات: تصويت PSM 7/6/13/11 وتصحيح قاموسي.<br>
                ② البحث في WhatTheFont ثم WhatFontIs عند ضبط مفاتيحها؛ زر «خاطئة» في البوت يبحث في مكتبة المالك فقط.<br>
                ③ رسم النص ومقارنة القناع والحواف والنسبة والسُمك والبروفايل عبر Google + مكتبة المالك.<br>
                ④ DINOv2 يعيد ترتيب أفضل 30؛ على Space يستخدم ZeroGPU، ومحلياً CPU ONNX.<br><br>
                <b>مهم:</b> اربط Storage Bucket/Volume بمسار البيانات لكي تبقى الخطوط بعد إعادة التشغيل. وجود مجلد قابل للكتابة وحده لا يضمن الحفظ.</div>""")
                status_button.click(lambda: runtime_status(), outputs=status_output)
                timer = gr.Timer(15)
                timer.tick(lambda: runtime_status(), outputs=status_output)
                timer.tick(github_job_status, outputs=github_jobs)
        gr.Markdown("Englishawy · Gradio + ZeroGPU · Telegram owner library · لا روابط تحميل في النتائج")
    return demo


demo = build_ui()
demo.queue(max_size=32, default_concurrency_limit=1, api_open=False)


async def health():
    telegram = bridge.status() if bridge else {"configured": False, "error": bridge_error or "TELEGRAM_BOT_TOKEN is not set"}
    if bridge and hasattr(bridge, "webhook_info"):
        if getattr(bridge, "mode", "webhook") == "webhook":
            telegram["telegram_side"] = await bridge.webhook_info()
    return {"ok": True, "assets": assets_runtime.snapshot(), "storage": storage_status(),
            "gpu": zero_gpu.status(), "telegram": telegram}


def make_lifespan(inner=None):
    @asynccontextmanager
    async def lifespan(active_app):
        if not getattr(active_app.state, "englishawy_routes_mounted", False):
            mount_telegram(active_app, bridge)
            active_app.add_api_route("/health", health, methods=["GET"])
            active_app.state.englishawy_routes_mounted = True
        logger.info("Englishawy startup: preparing assets and Telegram transport")
        assets_runtime.start()
        if bridge and not bridge.started:
            try:
                await bridge.start()
                logger.info("Telegram %s started", bridge.mode)
            except Exception as exc:
                bridge.error = describe_telegram_error(exc)
                logger.error("Telegram startup failed: %s", bridge.error)
        elif not bridge:
            logger.warning("Telegram remains disabled: %s", bridge_error or "TELEGRAM_BOT_TOKEN is not set")
        async with AsyncExitStack() as stack:
            if inner is not None:
                await stack.enter_async_context(inner(active_app))
            try:
                yield
            finally:
                if bridge:
                    await bridge.stop()
    return lifespan


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
                ssr_mode=False, show_api=False, max_file_size=None)
