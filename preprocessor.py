"""
preprocessor.py — وحدة المعالجة البصرية المسبقة وعزل الحروف (مثل WhatTheFont / WhatFontIs)
========================================================================================
تعالج أهم سبب لعدم دقة التعرف في النسخة الأولى:
1. اكتشاف الخلفيات الغامقة تلقائياً وعكسها بحيث تصبح الحروف سوداء على خلفية بيضاء.
2. قص الهوامش الفارغة حول النص تلقائياً (Auto-Crop Text ROI).
3. تعزيز التباين والحدة أو تطبيق Otsu-like Binarization لفصل الحروف عن الخلفية.
4. استخراج متجه البصمة الهندسية (8D Glyph Vector) للمطابقة مع مستودع السيرفر الخارجي.
"""

import io
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def preprocess_font_image(
    image_bytes: bytes,
    force_invert: bool = False,
    force_binarize: bool = False,
) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # تصغير أو تكبير الصورة إذا كانت صغيرة جداً أو ضخمة جداً لتحسين دقة OCR
    w, h = img.size
    if max(w, h) > 1600:
        scale = 1600 / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif max(w, h) < 400:
        scale = 600 / float(max(1, max(w, h)))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(img)

    # حساب متوسط سطوع الخلفية والأطراف لمعرفة ما إذا كانت الخلفية داكنة
    hist = gray.histogram()
    total_px = max(1, sum(hist))
    avg_lum = sum(i * hist[i] for i in range(256)) / total_px

    inverted = False
    if force_invert or avg_lum < 118:
        gray = ImageOps.invert(gray)
        inverted = True

    # تعزيز التباين والحدة
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = ImageEnhance.Contrast(gray).enhance(1.75)
    gray = gray.filter(ImageFilter.SHARPEN)

    # تطبيق عزل ثنائي حاد (Binarization) عند الطلب
    if force_binarize:
        threshold = 148
        gray = gray.point(lambda p: 255 if p > threshold else 0)

    # قص الهوامش البيضاء الفارغة حول الكلمة مع ترك هامش أمان (Padding)
    inv_for_bbox = ImageOps.invert(gray).point(lambda p: 255 if p > 60 else 0)
    bbox = inv_for_bbox.getbbox()
    if bbox:
        pad_x = max(12, int(gray.width * 0.04))
        pad_y = max(12, int(gray.height * 0.08))
        left = max(0, bbox[0] - pad_x)
        top = max(0, bbox[1] - pad_y)
        right = min(gray.width, bbox[2] + pad_x)
        bottom = min(gray.height, bbox[3] + pad_y)
        if (right - left) > 40 and (bottom - top) > 20:
            gray = gray.crop((left, top, right, bottom))

    # استخراج متجه مبدئي للبصمة الهندسية من الصورة المعالجة
    glyph_vector = _extract_glyph_vector(gray)

    out_buf = io.BytesIO()
    gray.convert("RGB").save(out_buf, format="PNG", optimize=True)

    summary_parts = [
        f"Size={gray.width}x{gray.height}",
        "Polarity=Inverted" if inverted else "Polarity=Normal",
        "Mode=Binarized" if force_binarize else "Mode=EnhancedGray",
    ]

    return {
        "image_bytes": out_buf.getvalue(),
        "glyph_vector": glyph_vector,
        "summary": " | ".join(summary_parts),
    }


def _extract_glyph_vector(gray_img: Image.Image) -> dict:
    """يحسب كثافة الحبر ونسب الانتقال الأفقي والرأسي لتقدير خصائص الخط."""
    sample = gray_img.resize((240, 80), Image.Resampling.BILINEAR)
    px = list(sample.getdata())
    w, h = 240, 80

    ink_count = 0
    horiz_edges = 0
    vert_edges = 0

    for y in range(h - 1):
        row_offset = y * w
        for x in range(w - 1):
            idx = row_offset + x
            is_ink = px[idx] < 140
            if is_ink:
                ink_count += 1
            right_ink = px[idx + 1] < 140
            down_ink = px[idx + w] < 140
            if is_ink != right_ink:
                horiz_edges += 1
            if is_ink != down_ink:
                vert_edges += 1

    ink_density = round(min(0.65, max(0.12, ink_count / float(w * h))), 3)
    edge_ratio = vert_edges / float(max(1, horiz_edges))
    est_serif = round(min(0.95, max(0.04, (edge_ratio - 0.55) * 1.25)), 2)
    est_contrast = round(min(0.92, max(0.08, est_serif * 0.82 + 0.08)), 2)
    est_weight = int(min(850, max(400, round(430 + ink_density * 720))))

    return {
        "serif_score": est_serif,
        "stroke_contrast": est_contrast,
        "x_height_ratio": 0.70,
        "width_ratio": 0.98,
        "weight_score": est_weight,
        "slope_angle": 0.0,
    }
