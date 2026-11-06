"""
pipeline.py — خط الأنابيب الكامل لـ Englishawy v3 (محلي بالكامل)
================================================================
يربط المراحل الثلاث التي وصفها سير العمل:
  1) قراءة الحروف: Tesseract عصبي + تصويت على معالجات متعددة + تصحيح قاموسي.
  2) مطابقة الأشكال: رسم النص بمكتبة GoogleFonts ومقارنة القناع/الحواف/النسبة/السُمك.
  3) إعادة الترتيب العصبي: DINOv2 يعيد ترتيب أفضل 30 مرشحاً.
والنتيجة: بطاقة مقارنة بصرية (اسم + معاينة) بدون أي روابط تحميل.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

import neural_reranker
from comparison_card import render_card
from font_matcher import FontMatch, load_index, match_fonts, render_text
from ocr_engine import OCRResult, read_text
from preprocessor import PreprocessResult, preprocess


@dataclass
class Identification:
    text: str
    ocr: OCRResult
    prep: PreprocessResult
    matches: list[FontMatch]
    card_png: bytes
    font_count: int
    elapsed_s: float
    neural_used: bool
    timings: dict = field(default_factory=dict)


def identify(
    image_bytes: bytes,
    top_n: int = 5,
    candidates: int = 30,
    force_invert: bool = False,
    force_binarize: bool = False,
    text_override: str | None = None,
    font_limit: int | None = None,
    progress=None,
) -> Identification:
    t0 = time.time()
    timings = {}

    t = time.time()
    prep = preprocess(image_bytes, force_invert=force_invert, force_binarize=force_binarize)
    timings["preprocess"] = round(time.time() - t, 2)

    t = time.time()
    if text_override:
        ocr = OCRResult(text=text_override.strip(), confidence=100.0, notes="نص يدوي من المستخدم")
    else:
        ocr = read_text(prep.variants)
    timings["ocr"] = round(time.time() - t, 2)

    matches: list[FontMatch] = []
    if ocr.text:
        t = time.time()
        matches = match_fonts(
            prep.reference_crop,
            ocr.text,
            top_k=candidates,
            limit=font_limit,
            progress=progress,
        )
        timings["shape_match"] = round(time.time() - t, 2)

        t = time.time()
        matches = neural_reranker.rerank(
            prep.reference_crop,
            matches,
            render_fn=lambda p: render_text(p, ocr.text, size=160),
        )
        timings["neural_rerank"] = round(time.time() - t, 2)

    matches = matches[:top_n]
    card = render_card(
        prep.original_crop,
        ocr.text or "—",
        matches,
        top_n=top_n,
        ocr_confidence=ocr.confidence if ocr.text else None,
    )
    return Identification(
        text=ocr.text,
        ocr=ocr,
        prep=prep,
        matches=matches,
        card_png=card,
        font_count=len(load_index()),
        elapsed_s=round(time.time() - t0, 2),
        neural_used=neural_reranker.available(),
        timings=timings,
    )
