"""
comparison_card.py — بطاقة المقارنة البصرية (مثل مثال MUQLA) بدون أي روابط
=========================================================================
ترسم صورة PNG واحدة تحتوي:
  - ترويسة بالعلامة (Englishawy / FONT COMPARISON) وشارة TOP N.
  - خانة ORIGINAL CROP تعرض قصّ المستخدم الأصلي.
  - صف لكل خط مرشح: رقم الترتيب + اسم الخط + معاينة حيّة للنص المقروء مرسوماً بالخط نفسه.
    إذا تعذّر الرسم يكتب "Preview unavailable" تماماً كما في المثال.

تُرسَل هذه البطاقة للمستخدم كصورة في تليجرام مع تعليق نصّي بالأسماء فقط — لا روابط تحميل.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("ENGLISHAWY_CACHE", ROOT / ".cache" / "englishawy"))
FONTS_DIR = CACHE_DIR / "fonts"

# لوحة الألوان (مستوحاة من المثال)
BG = (13, 22, 38)
PANEL = (241, 244, 249)
CARD = (255, 255, 255)
TEAL = (45, 212, 191)
TEAL_SOFT = (219, 244, 240)
INK = (30, 51, 75)
MUTED = (120, 132, 150)
LABEL = (90, 105, 125)
BORDER = (45, 212, 191)

W = 760
MARGIN = 26
INNER = W - 2 * MARGIN


def _ui_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        FONTS_DIR / ("manrope.ttf" if not bold else "manrope.ttf"),
        FONTS_DIR / "inter.ttf",
        FONTS_DIR / "montserrat.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


def _font_for(path: str, size: int, face_index: int = 0) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(path, size, index=face_index)
    except Exception:
        return None


def _fit_preview(text: str, font_path: str, max_w: int, max_h: int, face_index: int = 0) -> Image.Image | None:
    """يرسم النص بالخط ويحجّمه ليلائم الإطار مع الحفاظ على النسبة."""
    size = 150
    f = _font_for(font_path, size, face_index)
    if f is None:
        return None
    try:
        bbox = f.getbbox(text)
    except Exception:
        return None
    if not bbox:
        return None
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return None
    pad = 12
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((pad - bbox[0], pad - bbox[1]), text, font=f, fill=(17, 24, 33, 255))
    a = np.asarray(img)
    ys, xs = np.where(a[:, :, 3] > 8)
    if len(xs) == 0:
        return None
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    return img


def _round_rect(draw: ImageDraw.ImageDraw, xy, r, fill):
    draw.rounded_rectangle(xy, radius=r, fill=fill)


def render_card(
    original_crop: Image.Image,
    text: str,
    matches: list,
    top_n: int = 5,
    ocr_confidence: float | None = None,
) -> bytes:
    matches = matches[:top_n]
    row_h = 104
    header_h = 84
    crop_panel_h = 150
    gap = 14
    footer_h = 46
    total_h = header_h + crop_panel_h + gap + len(matches) * (row_h + gap) + footer_h + MARGIN

    img = Image.new("RGB", (W, total_h), BG)
    d = ImageDraw.Draw(img)

    # ---- الترويسة ----
    logo = 44
    lx, ly = MARGIN, 24
    _round_rect(d, (lx, ly, lx + logo, ly + logo), 12, TEAL)
    lf = _ui_font(26, bold=True)
    d.text((lx + 12, ly + 7), "E", font=lf, fill=BG)
    d.text((lx + logo + 14, ly + 2), "ENGLISHAWY", font=_ui_font(25, True), fill=(255, 255, 255))
    d.text((lx + logo + 14, ly + 30), "FONT COMPARISON", font=_ui_font(12, True), fill=(170, 185, 205))
    # شارة TOP N
    badge = f"TOP {len(matches)}"
    bf = _ui_font(13, True)
    bw = int(d.textlength(badge, font=bf)) + 28
    bx = W - MARGIN - bw
    d.rounded_rectangle((bx, ly + 6, bx + bw, ly + 34), radius=14, outline=TEAL, width=1)
    d.text((bx + 14, ly + 11), badge, font=bf, fill=TEAL)

    # ---- خانة القص الأصلي ----
    cy = header_h
    _round_rect(d, (MARGIN, cy, W - MARGIN, cy + crop_panel_h), 18, PANEL)
    d.text((MARGIN + 20, cy + 14), "ORIGINAL CROP", font=_ui_font(13, True), fill=LABEL)
    if ocr_confidence is not None:
        ctxt = f'OCR: "{text}"  ·  {ocr_confidence:.0f}%'
        d.text((MARGIN + 20, cy + crop_panel_h - 26), ctxt, font=_ui_font(12), fill=MUTED)
    # ملاءمة القص داخل الخانة
    box_w, box_h = INNER - 60, crop_panel_h - 70
    oc = original_crop.convert("RGB")
    s = min(box_w / oc.width, box_h / oc.height, 1.6)
    oc = oc.resize((max(1, int(oc.width * s)), max(1, int(oc.height * s))), Image.Resampling.LANCZOS)
    ox = MARGIN + (INNER - oc.width) // 2
    oy = cy + 38 + (box_h - oc.height) // 2
    img.paste(oc, (ox, oy))

    # ---- صفوف المرشحين ----
    y = cy + crop_panel_h + gap
    name_f = _ui_font(19, True)
    prev_missing_f = _ui_font(16)
    for idx, m in enumerate(matches, 1):
        _round_rect(d, (MARGIN, y, W - MARGIN, y + row_h), 18, CARD)
        # شارة الرقم
        nb = 36
        nx, ny = MARGIN + 18, y + 16
        _round_rect(d, (nx, ny, nx + nb, ny + nb), 10, TEAL_SOFT)
        nf = _ui_font(15, True)
        num = f"{idx:02d}"
        nw = d.textlength(num, font=nf)
        d.text((nx + (nb - nw) / 2, ny + 9), num, font=nf, fill=(15, 118, 110))
        # اسم الخط
        family = getattr(m, "family", str(m))
        family_text = family if len(family) <= 43 else family[:40] + "…"
        d.text((nx + nb + 16, y + 16), family_text, font=name_f, fill=INK)
        source = getattr(m, "source", "google")
        badge_text = "OWNER LIBRARY" if source == "owner" else "GOOGLE FONTS"
        badge_font = _ui_font(10, True)
        badge_w = d.textlength(badge_text, font=badge_font) + 16
        d.rounded_rectangle((W - MARGIN - badge_w - 16, y + 15, W - MARGIN - 16, y + 35), radius=8, fill=TEAL_SOFT)
        d.text((W - MARGIN - badge_w - 8, y + 19), badge_text, font=badge_font, fill=(15, 118, 110))
        # المعاينة الحيّة
        preview = _fit_preview(text, getattr(m, "font_path", ""), INNER - 110, row_h - 52, getattr(m, "face_index", 0))
        if preview is not None:
            px = nx + nb + 16
            py = y + 46 + (row_h - 52 - preview.height) // 2
            img.paste(preview, (px, py), preview)
        else:
            d.text((nx + nb + 16, y + 58), "Preview unavailable", font=prev_missing_f, fill=MUTED)
        y += row_h + gap

    # ---- تذييل ----
    fy = y + 2
    note = "Tesseract OCR  ·  Google + Owner fonts  ·  DINOv₂ rerank"
    d.text((MARGIN, fy), note, font=_ui_font(12), fill=(120, 135, 158))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
