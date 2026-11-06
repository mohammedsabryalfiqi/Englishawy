"""
whatfontis.py — واجهة مطورة للتعامل مع WhatFontIs API v2 مع آلية Fallback الذكية
===============================================================================
التوثيق الرسمي: https://www.whatfontis.com/API-identify-fonts-from-image.html
"""

import base64
import logging
import requests

WHATFONTIS_ENDPOINT = "https://www.whatfontis.com/api2/"
logger = logging.getLogger(__name__)


def call_whatfontis_single(
    image_bytes: bytes,
    api_key: str,
    limit: int = 8,
    not_textbox_detection: bool = False,
    free_only: bool = False,
) -> list:
    if not api_key:
        raise ValueError("WHATFONTIS_API_KEY غير موجود في ملف .env")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    data = {
        "API_KEY": api_key,
        "IMAGEBASE64": "1",
        "urlimagebase64": b64,
        "NOTTEXTBOXSDETECTION": "1" if not_textbox_detection else "0",
        "FREEFONTS": "1" if free_only else "0",
        "limit": str(max(1, min(limit, 20))),
    }

    resp = requests.post(WHATFONTIS_ENDPOINT, data=data, timeout=30)

    if resp.status_code == 429:
        raise RuntimeError(
            "تجاوزت الحد اليومي المسموح لطلبات WhatFontIs API (200 طلب/يوم). سيتم الاعتماد على مستودع الخطوط الداخلي/الخارجي."
        )
    if resp.status_code == 409:
        raise RuntimeError("مفتاح WHATFONTIS_API_KEY غير صحيح.")
    if resp.status_code >= 400:
        return []

    try:
        payload = resp.json()
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def identify_font_with_fallback(
    processed_image_bytes: bytes,
    raw_image_bytes: bytes,
    api_key: str,
    limit: int = 8,
    force_no_textbox: bool = False,
) -> list:
    """
    استراتيجية متعددة المراحل لضمان أعلى دقة ممكنة من WhatFontIs:
    1. فحص الصورة المعالجة مسبقاً (Binarized/Enhanced).
    2. إذا لم يعثر على صندوق نص، يعيد المحاولة تلقائياً بـ NOTTEXTBOXSDETECTION=1.
    3. إذا ظلت النتيجة فارغة، يجرب الصورة الأصلية الخام مع NOTTEXTBOXSDETECTION=1.
    """
    # المحاولة 1
    matches = call_whatfontis_single(
        processed_image_bytes,
        api_key=api_key,
        limit=limit,
        not_textbox_detection=force_no_textbox,
    )
    if matches:
        return matches

    # المحاولة 2: عكس إعداد NOTTEXTBOXSDETECTION
    logger.info("Retrying WhatFontIs with NOTTEXTBOXSDETECTION=%s", not force_no_textbox)
    matches = call_whatfontis_single(
        processed_image_bytes,
        api_key=api_key,
        limit=limit,
        not_textbox_detection=not force_no_textbox,
    )
    if matches:
        return matches

    # المحاولة 3: استخدام الصورة الأصلية الخام مع NOTTEXTBOXSDETECTION=1
    return call_whatfontis_single(
        raw_image_bytes,
        api_key=api_key,
        limit=limit,
        not_textbox_detection=True,
    )
