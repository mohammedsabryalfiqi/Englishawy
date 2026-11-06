"""Validate font tables and render a real probe before accepting an uploaded font."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

PROBE_TEXT = "Hamburgefonstiv Question 0123456789"
SAMPLE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def inspect_face(path: str | Path, face_index: int = 0) -> dict:
    path = str(path)
    with TTFont(path, fontNumber=face_index, lazy=True, recalcBBoxes=False) as font:
        if not any(tag in font for tag in ("glyf", "CFF ", "CFF2")) or "cmap" not in font:
            raise ValueError("الخط لا يحتوي على outlines وخريطة Unicode صالحة")
        cmap = font.getBestCmap() or {}
        if not any(ord(c) in cmap for c in SAMPLE_CHARS):
            raise ValueError("الخط لا يحتوي على حروف لاتينية قابلة للمطابقة")
        family = font["name"].getDebugName(16) or font["name"].getDebugName(1) or Path(path).stem
        style = font["name"].getDebugName(17) or font["name"].getDebugName(2) or "Regular"
        os2 = font.get("OS/2")
        weight = int(getattr(os2, "usWeightClass", 400)) if os2 else 400
        italic = bool(os2 and (os2.fsSelection & 1))
        coverage = "".join(c for c in SAMPLE_CHARS if ord(c) in cmap)
        glyph_count = int(font["maxp"].numGlyphs)
        sfnt = font.sfntVersion
        font.close()

    pil = ImageFont.truetype(path, 72, index=face_index)
    bbox = pil.getbbox(PROBE_TEXT)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if width <= 0 or height <= 0 or width > 16000 or height > 2000:
        raise ValueError("أبعاد معاينة الخط غير صالحة")
    image = Image.new("L", (width + 24, height + 24), 255)
    ImageDraw.Draw(image).text((12 - bbox[0], 12 - bbox[1]), PROBE_TEXT, font=pil, fill=0)
    ink = np.asarray(image) < 160
    if not ink.any():
        raise ValueError("الخط لا يرسم معاينة حقيقية")
    metrics = {
        "probe_text": PROBE_TEXT,
        "width": width,
        "height": height,
        "aspect": round(width / height, 5),
        "ink_density": round(float(ink.mean()), 6),
        "weight": weight,
        "italic": italic,
        "glyph_count": glyph_count,
        "latin_coverage": coverage,
    }
    return {
        "family": " ".join(str(family).split())[:180],
        "style": " ".join(str(style).split())[:100],
        "category": "owner",
        "face_index": face_index,
        "extension": ".otf" if sfnt == "OTTO" else ".ttf",
        "analysis_json": json.dumps(metrics, ensure_ascii=False),
    }
