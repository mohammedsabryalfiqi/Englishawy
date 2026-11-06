"""Cascade stage 1+2: search external font-identification sites FIRST.

Order (as requested by the owner):
  1) MyFonts WhatTheFont  (https://www.myfonts.com/pages/whatthefont)
  2) WhatFontIs           (https://www.whatfontis.com/)
  3) Owner library fallback (handled by pipeline.py when confidence < threshold)

WhatFontIs has an official public API (api2) and is fully implemented here:
  POST https://www.whatfontis.com/api2/
  fields: API_KEY, IMAGEBASE64=1, urlimagebase64=<b64>, NOTTEXTBOXSDETECTION, FREEFONTS, limit
  See: https://www.whatfontis.com/API-identify-fonts-from-image.html

MyFonts WhatTheFont has NO free public JSON API (enterprise contract only).
This module therefore:
  - calls your enterprise endpoint when MYFONTS_API_KEY + WHATTHEFONT_API_URL are set,
  - otherwise returns [] gracefully so the cascade continues to WhatFontIs
    and then to the owner library (never crashes the bot).
The Gradio/Telegram UI always shows the WhatTheFont deep-link first so the
owner/user can verify manually in one click.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from config import cascade_threshold, whatfontis_api_key

logger = logging.getLogger(__name__)

WHATTHEFONT_URL = "https://www.myfonts.com/pages/whatthefont"
WHATFONTIS_URL = "https://www.whatfontis.com/"
WHATFONTIS_API_ENDPOINT = "https://www.whatfontis.com/api2/"
CASCADE_THRESHOLD = cascade_threshold()


@dataclass
class ExternalMatch:
    family: str
    style: str = "Regular"
    source: str = "WhatTheFont"          # "WhatTheFont" | "WhatFontIs"
    source_url: str = WHATTHEFONT_URL
    score: float = 0.0                   # 0..1
    final_score: float = 0.0             # 0..1 (same scale, kept for pipeline compat)
    preview_images: tuple[str, ...] = ()  # provider specimens; not the user's word


@dataclass
class CascadeReport:
    whatthefont_tried: bool = False
    whatthefont_best: float = 0.0
    whatfontis_tried: bool = False
    whatfontis_best: float = 0.0
    fallback_to_owner: bool = True
    winning_stage: str = "owner_library"  # whatthefont | whatfontis | owner_library
    matches: list[ExternalMatch] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)


def _post_form(url: str, fields: dict, timeout: int) -> bytes:
    data = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; EnglishawyBot/5.0)",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": WHATFONTIS_URL,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def query_whatthefont(image_bytes: bytes, ocr_text: str = "", timeout: int = 12) -> list[ExternalMatch]:
    """Step 1 — MyFonts WhatTheFont (first option).

    Enterprise path: set WHATTHEFONT_API_URL + MYFONTS_API_KEY in Space secrets.
    Expected JSON: {"results": [{"family_name":..., "style_name":..., "confidence": 0..1|0..100}]}
    Without keys: returns [] so the cascade continues (bot never crashes).
    """
    api_url = os.getenv("WHATTHEFONT_API_URL", "").strip()
    api_key = os.getenv("MYFONTS_API_KEY", "").strip()
    if not api_url or not api_key:
        logger.info("WhatTheFont: no enterprise key configured, skipping API call (manual deep-link shown in UI).")
        return []
    try:
        payload = json.dumps({
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "text_hint": (ocr_text or "")[:160],
            "limit": 5,
        }).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; EnglishawyBot/5.0)",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Referer": WHATTHEFONT_URL,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        items = data.get("results") or data.get("fonts") or []
        out: list[ExternalMatch] = []
        for it in items[:5]:
            family = str(it.get("family_name") or it.get("title") or "").strip()
            if not family:
                continue
            style = str(it.get("style_name") or "Regular").strip()
            try:
                conf = float(it.get("confidence", it.get("score", 0.8)))
            except (TypeError, ValueError):
                conf = 0.8
            if conf > 1.0:  # accept 0..100 scale too
                conf = conf / 100.0
            conf = max(0.0, min(1.0, conf))
            out.append(ExternalMatch(family=family, style=style, source="WhatTheFont",
                                     source_url=WHATTHEFONT_URL, score=conf, final_score=conf))
        return out
    except Exception as exc:  # never break identification
        logger.info("WhatTheFont query fallback: %s", type(exc).__name__)
        return []


def query_whatfontis(image_bytes: bytes, ocr_text: str = "", timeout: int = 15,
                     report: CascadeReport | None = None) -> list[ExternalMatch]:
    """Step 2 — WhatFontIs official api2 (second option). Needs WHATFONTIS_API_KEY.

    Get a free key after login at:
      https://www.whatfontis.com/API-identify-fonts-from-image.html
    """
    _ = ocr_text
    api_key = whatfontis_api_key()
    if not api_key:
        logger.info("WhatFontIs: API key not set, skipping.")
        if report is not None:
            report.notices.append("مفتاح البحث الخارجي غير مضبوط")
        return []
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        raw = _post_form(WHATFONTIS_API_ENDPOINT, {
            "API_KEY": api_key,
            "IMAGEBASE64": "1",
            "NOTTEXTBOXSDETECTION": "1",  # input is already a tight word crop
            "FREEFONTS": "0",
            "urlimage": "",
            "urlimagebase64": b64,
            "limit": "5",
        }, timeout=timeout)
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        if isinstance(data, dict) and data.get("error"):
            # Provider errors may echo request data. Log only a safe category.
            logger.warning("WhatFontIs API returned an error response")
            if report is not None:
                report.notices.append("خدمة البحث الخارجي رفضت الطلب؛ راجع المفتاح/الرصيد")
            return []
        if not isinstance(data, list):
            if report is not None:
                report.notices.append("رد غير متوقع من خدمة البحث الخارجي")
            return []
        out: list[ExternalMatch] = []
        for idx, item in enumerate(data[:5]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            page = str(item.get("url") or WHATFONTIS_URL)
            score = max(0.60, 0.90 - idx * 0.04)  # api2 returns best-first, no scores
            specimens = tuple(str(item[k]) for k in ("image", "image1", "image2")
                              if isinstance(item.get(k), str) and item[k].startswith("https://"))
            out.append(ExternalMatch(family=title, style="Regular", source="WhatFontIs",
                                     source_url=page, score=score, final_score=score,
                                     preview_images=specimens))
        return out
    except urllib.error.HTTPError as exc:
        logger.warning("WhatFontIs HTTP status: %s", exc.code)
        if report is not None:
            hint = "انتهى رصيد/حد طلبات البحث الخارجي" if exc.code == 429 else "تعذر الاتصال بالبحث الخارجي؛ راجع المفتاح أو الشبكة"
            report.notices.append(hint)
        return []
    except Exception as exc:
        logger.info("WhatFontIs query fallback: %s", type(exc).__name__)
        if report is not None:
            report.notices.append("تعذر الاتصال بخدمة البحث الخارجي")
        return []


def search_external_first(image_bytes: bytes, ocr_text: str = "",
                          threshold: float = CASCADE_THRESHOLD) -> CascadeReport:
    """Run WhatTheFont FIRST, then WhatFontIs, decide owner-library fallback."""
    report = CascadeReport()
    t0 = time.time()
    wtf = query_whatthefont(image_bytes, ocr_text=ocr_text)
    report.timings["whatthefont"] = round(time.time() - t0, 2)
    report.whatthefont_tried = True
    report.whatthefont_best = max((m.final_score for m in wtf), default=0.0)

    t1 = time.time()
    wfi = query_whatfontis(image_bytes, ocr_text=ocr_text, report=report)
    report.timings["whatfontis"] = round(time.time() - t1, 2)
    report.whatfontis_tried = True
    report.whatfontis_best = max((m.final_score for m in wfi), default=0.0)

    combined = sorted(wtf + wfi, key=lambda m: m.final_score, reverse=True)
    seen: set[str] = set()
    deduped: list[ExternalMatch] = []
    for m in combined:
        key = m.family.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(m)
    report.matches = deduped

    best = max(report.whatthefont_best, report.whatfontis_best)
    if deduped and best >= threshold:
        report.fallback_to_owner = False
        report.winning_stage = "whatthefont" if report.whatthefont_best >= report.whatfontis_best else "whatfontis"
    else:
        report.fallback_to_owner = True
        report.winning_stage = "owner_library"
    return report
