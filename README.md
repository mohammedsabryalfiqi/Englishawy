---
title: Englishawy — كاشف الخطوط
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.12.12"
short_description: Tesseract + Google/owner fonts + DINOv2 on ZeroGPU
models:
  - facebook/dinov2-small
preload_from_hub:
  - facebook/dinov2-small
---

# Englishawy v4 — Gradio / Hugging Face ZeroGPU

كاشف خطوط إنجليزي بواجهة عربية، وبوت تليجرام، ومكتبة خطوط خاصة بالمالك تُستخدم في البحث إلى جانب Google Fonts.

**النتيجة:** بطاقة مقارنة مثل MUQLA: اسم الخط + معاينة حيّة للنص. **لا توجد روابط تحميل خطوط في النتائج.**

## ماذا يفعل؟

1. **Tesseract OCR:** ثماني معالجات مسبقة، تصويت بين PSM 7/6/13/11، وتصحيح قاموسي بـwordfreq/rapidfuzz.
2. **مطابقة الشكل:** يرسم النص بكل خط متاح ويقارن القناع soft-IoU، الحواف chamfer، النسبة، السُمك، وبروفايل الحروف. يبحث في Google Fonts (1818 عائلة في المنيفست) **ومكتبة المالك**.
3. **DINOv2:** يعيد ترتيب أفضل **30** مرشحاً. على Space يستخدم `facebook/dinov2-small` عبر PyTorch وZeroGPU؛ محلياً أو عند اختيار CPU يستخدم ONNX.
4. **البطاقة:** أعلى 5 افتراضياً (حتى 10)، بالأسماء والمعاينات فقط.

## لماذا يوجد رفع ZIP من المتصفح ومن تليجرام؟

- **Gradio:** المالك يرفع ZIP أو ملفات خطوط مباشرة. **لا يوجد حدّ حجم مفروض في التطبيق**؛ أرشيف 200MB مقبول إذا سمحت المنصة والشبكة والقرص. لا نمرر `max_file_size` إلى Gradio.
- **Telegram Cloud Bot API:** تنزيل الملفات عبر `getFile` محدود بـ20MB. لذلك الأرشيفات الأكبر من 20MB تُحوَّل إلى لوحة المالك بكود `/login`. هذا حدّ من تليجرام، لا من Englishawy.
- **Local Bot API server:** إذا شغّلته بنفسك، يستطيع البوت المحلي تنزيل ملفات أكبر؛ فعّل `TELEGRAM_LOCAL_API=1` واضبط عنوان الخادم. لا تفعل ذلك إلا إذا كان الخادم يعمل فعلاً.

## النشر على Hugging Face Spaces

1. أنشئ Space جديداً واختر **SDK: Gradio**.
2. من إعدادات العتاد اختر **ZeroGPU** يدوياً. ملف README يضبط Gradio/Python وتجهيز النموذج، لكنه لا يغيّر عتاد Space موجوداً تلقائياً.
3. ارفع ملفات المستودع إلى الـSpace. الأصول الثقيلة غير موجودة في Git وتُجهّز عند التشغيل/البناء.
4. أضف هذه القيم في **Settings → Variables and secrets**:

| النوع | الاسم | القيمة/الوظيفة |
|---|---|---|
| Secret | `TELEGRAM_BOT_TOKEN` | توكن BotFather |
| Secret | `TELEGRAM_WEBHOOK_SECRET` | نص عشوائي 16 حرفاً على الأقل، مثلاً `openssl rand -hex 32` |
| Variable | `OWNER_TELEGRAM_ID` | `5231453921` (وهو الافتراضي في الكود) |
| Variable | `ENGLISHAWY_DATA` | `/data/englishawy` |
| Variable | `ENGLISHAWY_PERSISTENT_STORAGE` | `1` **بعد** ربط Volume فعلياً |
| Secret اختياري | `GH_TOKEN` | لتقليل مشاكل rate limit أثناء تنزيل خطوط Google فقط |

5. اربط **Storage Bucket كـVolume** على `/data` بوضع قراءة/كتابة. التخزين العادي في Space مؤقت؛ مجرد وجود `/data` لا يثبت أنه دائم. التطبيق يعرض تحذيراً إذا لم تؤكد ربط التخزين الدائم.
6. شغّل/أعد تشغيل Space. يجهّز Tesseract وخطوط Google ونموذج ONNX الاحتياطي في الخلفية، ويظهر التقدم في تبويب الحالة. يُجهّز النموذج العصبي من HF أثناء البناء ويُحمّل إلى CUDA على مستوى الوحدة (module scope) حسب إرشادات ZeroGPU.
7. يفتح التطبيق Webhook على نفس منفذ Gradio: `https://SPACE_HOST/telegram/webhook`. `SPACE_HOST` توفره HF تلقائياً. عند استخدام نطاق مخصص اضبط `PUBLIC_BASE_URL` كـHTTPS.

> لا تشغّل `bot.py` polling بنفس التوكن أثناء تشغيل Webhook على Space؛ سيحدث تعارض في استلام التحديثات.

### ZeroGPU

- الدالة decorated بـ`@spaces.GPU(duration=30)` تستقبل مصفوفات CPU وتعيد درجات cosine كـPython floats؛ لا تُمرّر كائنات CUDA عبر حدود العملية.
- OCR/fك الضغط/فحص الخطوط/المطابقة الشكلية تبقى على CPU.
- البوت يستخدم CPU ONNX حتى لا يستهلك حصة GPU خارج طلب Gradio المصرّح به.
- إذا انتهت حصة ZeroGPU، يظهر خطأ واضح ويمكن اختيار CPU من الخيارات المتقدمة.
- الإصدارات المثبتة: Gradio 5.50.0، Python 3.12.12، Torch 2.8.0، Transformers 4.57.1. راجع دليل ZeroGPU الرسمي لأن المنصة تتحدث بمرور الوقت.

## مكتبة المالك

1. افتح محادثة **خاصة** مع البوت من حساب تليجرام رقم `5231453921`.
2. أرسل `/login`.
3. في Gradio افتح «مكتبة المالك» وأدخل الكود خلال 5 دقائق. الكود يعمل مرة واحدة، والجلسة تنتهي بعد 8 ساعات أو إعادة التشغيل.
4. ارفع ZIP أو TTF/OTF/TTC/WOFF/WOFF2 واضغط التثبيت.
5. تظهر الخطوط فوراً في البحث كمصدر «مكتبة المالك». يمكن البحث والحذف بالـID.

أوامر المالك في تليجرام: `/login`, `/owner`, `/fonts [page]`, `/removefont ID`.

### ما يحدث داخل ZIP؟

- قراءة الأعضاء بالتدفق إلى ملفات مؤقتة، بدون `extractall` وبدون تحميل الأرشيف كله في RAM.
- رفض path traversal والروابط الرمزية والأعضاء المشفرة.
- دعم ZIP متداخل حتى عمق 3.
- فحص جداول `glyf`/`CFF` و`cmap` ووجود حروف لاتينية، ثم رسم معاينة حقيقية قبل القبول.
- دعم وجوه TTC، وتحويل WOFF/WOFF2 إلى sfnt لكي يرسمها Pillow بثبات.
- إزالة التكرار ببصمة SHA-256 للوجه، وفهرسة SQLite بجانب الملفات.
- الخط التالف لا يوقف باقي المجموعة؛ يظهر تقرير بالملفات المرفوضة.
- لا يوجد حدّ حجم ZIP في الكود. تبقى حدود القرص/الشبكة/المنصة، ونحتفظ بهامش أمان 64MB على القرص.

**المسؤولية:** ارفع فقط خطوطاً تملك حق استخدامها/تخزينها. Englishawy لا يوزع روابط تحميلها للزوار.

## التشغيل المحلي (CPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install gradio==5.50.0 -r requirements.txt -r requirements-dev.txt
cp .env.example .env
# عدّل .env: للتجربة المحلية اجعل ENGLISHAWY_PERSISTENT_STORAGE=0
# ويمكن ترك ENGLISHAWY_DATA فارغاً لاستخدام .cache/englishawy/data
python setup_assets.py --limit 940 --workers 16
python app.py
```

افتح `http://localhost:7860`. بدون توكن تليجرام تعمل واجهة تحديد الخطوط، لكن دخول المالك يحتاج البوت لإصدار الكود.

للبوت المحلي polling (بعد ضبط التوكن في `.env`):

```bash
python bot.py
```

لتجربة المحرك بدون واجهة:

```bash
python identify_cli.py image.png --text "Question" --source all --out card.png
```

> حزمة `tesserocr` توفر عجلة manylinux تتضمن مكتبة Tesseract؛ لا تحتاج apt في البيئة القياسية. إن لم توجد عجلة لمنصتك، ثبت Tesseract/Libleptonica development headers أولاً.

## الاختبارات

```bash
python -m pytest -q
```

تغطي: هوية المالك، الكود أحادي الاستخدام وانتهاء الجلسة، فك ZIP الآمن، archive >200MiB بلا حدّ تطبيقي، الخطوط التالفة/المتداخلة، WOFF، الحفظ وإعادة الفتح والحذف، بحث خطوط المالك، Webhook secret، ورفض رفع غير المالك قبل التنزيل، بطاقة المقارنة، وواجهة ZeroGPU بدرجات CPU باستخدام نموذج mock.

**لم يتم اختبار عتاد ZeroGPU الفعلي أو نشر Space حي من هذه البيئة.** المسار المحلي CPU والاختبارات الآلية منفذة؛ تحقق من Space بعد النشر لأن الحصة/الشبكة/التخزين تعتمد على حسابك وإعداداته.

## البنية

| الملف | الدور |
|---|---|
| `app.py` | Gradio UI + FastAPI health/webhook على منفذ واحد |
| `zero_gpu.py` | تهيئة DINOv2 على مستوى module وحد GPU ضيق |
| `neural_reranker.py` | تجهيز مدخلات CPU ومزج الدرجات واحتياطي ONNX |
| `font_store.py` / `font_analysis.py` | فك/فحص/تثبيت وفهرسة خطوط المالك |
| `owner_auth.py` | أكواد تليجرام لمرة واحدة وجلسات متصفح مبهمة |
| `bot.py` / `telegram_webhook.py` | أوامر تليجرام، polling محلي أو webhook على Space |
| `font_catalog.py` | Google + owner كمصدر بحث واحد |
| `preprocessor.py` / `ocr_engine.py` | Tesseract voting/dictionary |
| `font_matcher.py` | shape/stroke/aspect matching |
| `comparison_card.py` | بطاقة أسماء ومعاينات بلا روابط |

## مراجع المنصة

- [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Spaces storage / attached volumes](https://huggingface.co/docs/hub/spaces-storage)
- [Spaces configuration](https://huggingface.co/docs/hub/spaces-config-reference)
- [Telegram Bot API — getFile](https://core.telegram.org/bots/api#getfile)
