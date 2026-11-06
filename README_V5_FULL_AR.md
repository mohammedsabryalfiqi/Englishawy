# Englishawy v5 — النسخة الكاملة الجاهزة للتشغيل 🇪🇬

هذه نسخة **كاملة** من مشروع البوت — كل الملفات التي ستُشغّل البوت بها موجودة هنا،
مع تعديلات v5 مُركّبة وجاهزة (لا تحتاج أي خطوة يدوية في الكود).

## ما الجديد في v5؟

1. **البحث بالترتيب المطلوب لأي صورة:**
   - **أول خيار:** WhatTheFont — https://www.myfonts.com/pages/whatthefont
   - **ثاني خيار:** WhatFontIs — https://www.whatfontis.com/ (عبر API الرسمي `api2`)
   - **بعدين:** لو ما لقاش الخط المطابق (أقل من `CASCADE_THRESHOLD=0.78`) يبحث تلقائياً في **مكتبة المالك**.
2. **رابط GitHub فيه مستودع خطوط كبير:**
   - المالك يلصق الرابط في تبويب مكتبة المالك بالموقع **أو يبعته للبوت مباشرة** (أو `/import_github <رابط>`).
   - **البوت/الخادم يحمّله ويثبّته ويحلّله وحده في الخلفية (Daemon Thread)** — تخرج من صفحة الرفع أو تقفل المتصفح، والتحميل **ما يوقفش**.
   - التقدم محفوظ في جدول `github_jobs` — تتابعه من الموقع (يتحدث كل 15 ثانية) أو `/jobs` في تليجرام.

## التشغيل المحلي (تجربة سريعة)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# عدّل .env: ضع TELEGRAM_BOT_TOKEN و WHATFONTIS_API_KEY
python app.py
# افتح http://localhost:7860
# وللبوت polling محلي (بعد ضبط التوكن): python bot.py
```

## النشر على Hugging Face Space

1. أنشئ Space جديد (SDK: Gradio) واختر **ZeroGPU** من إعدادات العتاد.
2. ارفع **كل ملفات هذا المجلد** إلى الـ Space.
3. في **Settings → Variables and secrets** أضف:
   - `TELEGRAM_BOT_TOKEN` (Secret) — توكن BotFather
   - `TELEGRAM_WEBHOOK_SECRET` (Secret) — نص عشوائي 16 حرفاً على الأقل
   - `OWNER_TELEGRAM_ID` = `5231453921`
   - `WHATFONTIS_API_KEY` (Secret) — مفتاح مجاني بعد تسجيل الدخول في https://www.whatfontis.com/API-identify-fonts-from-image.html
   - `CASCADE_THRESHOLD` = `0.78` (Variable — اختياري)
   - `MYFONTS_API_KEY` + `WHATTHEFONT_API_URL` (اختياري — لعملاء MyFonts enterprise فقط)
   - `GH_TOKEN` (Secret اختياري — يرفع حد GitHub عند سحب المستودعات الضخمة)
   - `ENGLISHAWY_DATA=/data/englishawy` + `ENGLISHAWY_PERSISTENT_STORAGE=1` (بعد ربط Volume)
4. اربط **Storage Bucket كـ Volume** على `/data` ثم أعد تشغيل الـ Space.

## أوامر البوت (للمالك في محادثة خاصة)

- `/start` `/help` — شرح الترتيب الجديد
- `/status` — حالة المكتبة + الترتيب
- `/login` — كود لوحة الويب (5 دقائق، مرة واحدة)
- `/owner` `/fonts` `/removefont` — إدارة المكتبة
- `/import_github https://github.com/google/fonts` — سحب مستودع ضخم في الخلفية (البوت يحمّل بنفسه)
- لصق رابط `github.com/...` مباشرة — يُلتقط تلقائياً ويبدأ السحب الخلفي
- `/jobs` — عرض مهام السحب الخلفية وحالتها

## الملفات الجديدة/المعدلة في v5

| الملف | الحالة |
|---|---|
| `external_font_search.py` | جديد — المرحلة 1+2 |
| `github_ingestor.py` | جديد — السحب الخلفي |
| `app_github_addon.py` | جديد — واجهة السحب في Gradio (مربوطة تلقائياً في `app.py`) |
| `pipeline.py` | معدّل — التسلسل الكامل |
| `bot.py` | معدّل — الأوامر الجديدة |
| `services.py` | معدّل — `ingestor` مشترك |
| `config.py` | معدّل — العتبة والمفاتيح |
| `app.py` | معدّل — مربوط تلقائياً (لا تلمس شيئاً) |
| `.env.example` | معدّل — المتغيرات الجديدة |
| `tests/test_cascade_github.py` | جديد — اختبارات بدون إنترنت |

## ملاحظات

- WhatTheFont ليس له API عام مجاني — بدون مفتاح enterprise يظهر كزر تحقق يدوي أولاً ثم يكمل التسلسل تلقائياً. الترتيب محفوظ دائماً.
- النتائج **أسماء ومعاينات فقط — بدون روابط تحميل خطوط**.
- حد تليجرام 20MB يخص الملفات المرفوعة للمحادثة فقط — **سحب GitHub يتم من الخادم مباشرة ولا يتأثر به**.
