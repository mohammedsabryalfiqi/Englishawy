"""
واجهة بسيطة للتعامل مع WhatFontIs API v2
التوثيق: https://www.whatfontis.com/API-identify-fonts-from-image.html

لازم تعمل حساب على whatfontis.com وتاخد الـ API_KEY بتاعك من صفحة حسابك.
الخطة المجانية: حتى 200 طلب/يوم.
"""

import base64
import requests

WHATFONTIS_ENDPOINT = "https://www.whatfontis.com/api2/"


def identify_font(image_bytes: bytes, api_key: str, limit: int = 8, free_only: bool = False):
    """
    يبعت صورة لـ WhatFontIs ويرجع قائمة بأقرب الخطوط المطابقة.

    كل عنصر في القائمة شكله تقريبًا:
        {
            "title": "Abril Fatface",
            "url": "https://www.whatfontis.com/NMY_Abril-Fatface.font",
            "image": "...",
            "type": "Commercial" أو "Free",
            "site": "..."
        }
    """
    if not api_key:
        raise ValueError("WHATFONTIS_API_KEY غير موجود. حطه في ملف .env")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    data = {
        "API_KEY": api_key,
        "IMAGEBASE64": "1",          # 1 = هنبعت الصورة كـ base64 مش رابط
        "urlimagebase64": b64,
        "NOTTEXTBOXSDETECTION": "0",  # 0 = حاول تلاقي صندوق النص تلقائيًا
        "FREEFONTS": "1" if free_only else "0",
        "limit": str(max(1, min(limit, 20))),
    }

    resp = requests.post(WHATFONTIS_ENDPOINT, data=data, timeout=30)

    if resp.status_code == 429:
        raise RuntimeError("تجاوزت الحد اليومي المسموح لطلبات WhatFontIs API (200 طلب/يوم في الخطة المجانية).")
    if resp.status_code == 409:
        raise RuntimeError("مفتاح الـ API غير صحيح أو مفقود.")
    if resp.status_code >= 400:
        # زي حالات: مفيش صندوق نص، مفيش حروف، الصورة كبيرة، نوع صورة غير مدعوم...
        raise RuntimeError(f"خطأ من WhatFontIs (كود {resp.status_code}): {resp.text[:300]}")

    return resp.json()
