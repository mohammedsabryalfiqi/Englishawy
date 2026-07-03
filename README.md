# بوت تحديد الخطوط من صورة (Telegram Font Identifier Bot)

بوت تليجرام: تبعتله صورة فيها كلمة/جملة بخط انجليزي، فيحللها باستخدام
[WhatFontIs API](https://www.whatfontis.com/API-identify-fonts-from-image.html)
(نفس المحرك اللي بيحلل شكل الحروف ويطابقها مع قاعدة بيانات ضخمة من الخطوط)،
وبعدين يدور تلقائيًا على كل خط في dafont.com ويجيبلك رابط تحميله المباشر
لو موجود، أو رابط صفحته/رابط بحث لو مش موجود بنفس الاسم بالظبط.

## ليه مش هتكون النتيجة 100% مضمونة دايمًا؟

تحديد الخط من صورة عملية احتمالية مش يقينية — حتى أقوى الأدوات المتخصصة
(زي WhatFontIs نفسه) بتديك "أقرب تطابقات" مش إجابة مؤكدة 100%، لأن آلاف
الخطوط بتتشابه في شكل الحروف. البوت بيرجعلك أفضل ٥ نتائج (تقدر تزودها) عشان
تختار الأقرب لعينك.

كمان: مش كل خط ظاهر في نتائج WhatFontIs هيكون موجود في dafont.com بالظبط
بنفس الاسم (ممكن يكون خط تجاري من موقع تاني زي MyFonts أو Creative Fabrica).
في الحالة دي البوت بيديك رابط بحث في dafont عشان تدور بنفسك على بديل قريب،
أو رابط صفحته على WhatFontIs نفسه.

## المتطلبات

- Python 3.10+
- توكن بوت تليجرام (من [@BotFather](https://t.me/BotFather))
- حساب ومفتاح API من [whatfontis.com](https://www.whatfontis.com/API-identify-fonts-from-image.html)
  (مجاني للاستخدام الشخصي حتى 200 طلب/يوم — سجّل حساب ثم روح لصفحة الـ API
  هتلاقي المفتاح بتاعك بعد تسجيل الدخول)

## التركيب والتشغيل

```bash
# 1. فك الضغط وادخل المجلد
cd font_bot

# 2. (اختياري بس مستحسن) اعمل بيئة افتراضية
python3 -m venv venv
source venv/bin/activate

# 3. ثبّت المكتبات
pip install -r requirements.txt

# 4. جهّز ملف الإعدادات
cp .env.example .env
nano .env   # أو أي محرر تاني، حط التوكن ومفتاح الـ API بتوعك

# 5. شغّل البوت
python bot.py
```

لو شغال على VPS وعايز يفضل شغال حتى بعد ما تقفل الترمينال، استخدم `tmux`
أو `screen`، أو الأفضل تعمله service بـ `systemd`:

```ini
# /etc/systemd/system/fontbot.service
[Unit]
Description=Telegram Font Identifier Bot
After=network.target

[Service]
WorkingDirectory=/path/to/font_bot
ExecStart=/path/to/font_bot/venv/bin/python bot.py
Restart=always
User=your_linux_user

[Install]
WantedBy=multi-user.target
```

ثم:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fontbot
sudo systemctl status fontbot
```

## بنية المشروع

```
font_bot/
├── bot.py           # منطق بوت تليجرام (استقبال الصور، الردود)
├── whatfontis.py     # الاتصال بـ WhatFontIs API
├── dafont.py         # البحث عن الخط في dafont.com واستخراج رابط التحميل
├── requirements.txt
├── .env.example
└── README.md
```

## ملاحظات مهمة

- **ملف `dafont.py`** بيعتمد على تحليل صفحات HTML الحالية لموقع dafont.com.
  لو الموقع غيّر شكل صفحاته مستقبلًا، ممكن تحتاج تعدّل الـ regex الموجودة
  فيه (فيه تعليقات توضح كل جزء). في كل الحالات البوت مش هيقف عن الشغل —
  أسوأ سيناريو إنه هيديك رابط بحث بدل رابط تحميل مباشر.
- **حدود الاستخدام:** الخطة المجانية من WhatFontIs بتسمح بـ 200 طلب/يوم لكل
  مفتاح API. لو البوت هيستخدمه ناس كتير، هتحتاج تكلمهم لترقية الخطة
  ([رابط التواصل](https://www.whatfontis.com/contact-api.html)).
- **حقوق الخطوط:** بعض الخطوط في dafont للاستخدام الشخصي فقط وبتحتاج ترخيص
  للاستخدام التجاري. البوت بيذكّر المستخدم بكده في كل رد، لكن لازم هو
  نفسه يتأكد من الترخيص المكتوب في صفحة كل خط قبل ما يستخدمه تجاريًا.
- **الصور المقصوصة كويس بتدي نتائج أدق** — لو الصورة فيها عناصر كتير غير
  النص (خلفية معقدة، لوجو، إلخ) ممكن الدقة تقل.

## تطوير مستقبلي مقترح

- إضافة زرار "أعد التحليل بحساسية أعلى" باستخدام `NOTTEXTBOXSDETECTION=1`
  لو الصورة معقدة.
- كاش محلي (قاعدة بيانات صغيرة) لتخزين نتائج dafont لأسماء خطوط اتكررت
  قبل كده، عشان تقلل عدد الطلبات لموقع dafont.
- دعم إرسال أكتر من صورة/زاوية لنفس الخط لتحسين الدقة.
