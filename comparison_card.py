"""
comparison_card.py — بطاقة المقارنة البصرية (مثل مثال MUQLA) بدون أي روابط
=========================================================================
ترسم صورة PNG واحدة تحتوي:
  - شريطًا علويًا بلون الشوكولاتة يحمل العلامة (Englishawy / FONT COMPARISON) وشارة TOP N.
  - بطاقة كريمية عائمة بحدود متقطعة، بداخلها خانة ORIGINAL CROP.
  - صف لكل خط مرشح على خلفية «سمني»: رقم الترتيب + اسم الخط + معاينة حيّة للنص
    المقروء مرسوماً بالخط نفسه، أو شريط نموذج الحروف من المزوّد معلَّماً sample
    عندما لا يتوفر ملف الخط محلياً.

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

# Custom mark, drop-in via assets/logo.png or $ENGLISHAWY_LOGO (PNG with alpha).
# Without one the card draws a built-in mark; LOGO_VARIANT picks it.
LOGO_PATH = Path(os.getenv("ENGLISHAWY_LOGO", str(ROOT / "assets" / "logo.png")))
LOGO_VARIANT = os.getenv("ENGLISHAWY_LOGO_VARIANT", "aa").strip().lower()

# لوحة الألوان — بني شوكولاتة، كريمي، وسمني
BG = (239, 233, 224)            # #EFE9E0 خلفية الصفحة (رمادي دافئ)
HEADER = (62, 39, 35)           # #3E2723 شريط علوي بني
CARD = (253, 252, 249)          # #FDFCF9 بطاقة كريمية
ROW = (251, 243, 227)           # #FBF3E3 السمني — خلفية المعاينات
PANEL = (246, 235, 215)         # #F6EBD7 سمني أعمق — خانة القص
DASH = (215, 184, 156)          # #D7B89C حد متقطع رملي
GOLD = (169, 116, 46)           # #A9742E ذهبي بني (التدرج في الشعار)
GOLD_SOFT = (241, 224, 198)     # #F1E0C6 تعبئة الشارات
INK = (62, 39, 35)              # #3E2723 نص أساسي
MUTED = (156, 138, 118)         # #9C8A76 نص ثانوي
LABEL = (138, 118, 96)          # #8A7660 عناوين صغيرة
ON_HEADER = (255, 255, 255)
ON_HEADER_SOFT = (215, 184, 156)

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
    if not path or not Path(path).is_file():
        return None
    try:
        return ImageFont.truetype(path, size, index=face_index)
    except (OSError, ValueError):
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
    d.text((pad - bbox[0], pad - bbox[1]), text, font=f, fill=INK + (255,))
    a = np.asarray(img)
    ys, xs = np.where(a[:, :, 3] > 8)
    if len(xs) == 0:
        return None
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    return img


def _fit_provider_preview(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Fit a validated provider rendering on a transparent paper.

    The specimen arrives on white paper; keying that white out lets it sit
    cleanly on the butter row instead of leaving a white rectangle behind.
    """
    image = image.convert("RGB")
    ink = np.min(np.asarray(image), axis=2) < 235
    ys, xs = np.where(ink)
    if len(xs):
        image = image.crop((max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
                            min(image.width, int(xs.max()) + 5), min(image.height, int(ys.max()) + 5)))
    arr = np.asarray(image).astype(np.int16)
    alpha = np.clip((255 - arr.min(axis=2) - 8) * 3, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.dstack([np.asarray(image).astype(np.uint8), alpha]), "RGBA")
    scale = min(max_w / max(1, image.width), max_h / max(1, image.height))
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                        Image.Resampling.LANCZOS)


def _round_rect(draw: ImageDraw.ImageDraw, xy, r, fill):
    draw.rounded_rectangle(xy, radius=r, fill=fill)


def _dash_line(draw: ImageDraw.ImageDraw, p0, p1, color, width=2, dash=9, gap=7) -> None:
    """Draw one straight dashed run between p0 and p1."""
    x0, y0 = p0
    x1, y1 = p1
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line((x0 + ux * pos, y0 + uy * pos, x0 + ux * end, y0 + uy * end),
                  fill=color, width=width)
        pos += dash + gap


def _dashed_round_rect(draw: ImageDraw.ImageDraw, xy, r, color, width=2) -> None:
    """Rounded rectangle with a dotted straight edge and solid corner arcs."""
    x0, y0, x1, y1 = xy
    _dash_line(draw, (x0 + r, y0), (x1 - r, y0), color, width)
    _dash_line(draw, (x0 + r, y1), (x1 - r, y1), color, width)
    _dash_line(draw, (x0, y0 + r), (x0, y1 - r), color, width)
    _dash_line(draw, (x1, y0 + r), (x1, y1 - r), color, width)
    draw.arc((x0, y0, x0 + 2 * r, y0 + 2 * r), 180, 270, fill=color, width=width)
    draw.arc((x1 - 2 * r, y0, x1, y0 + 2 * r), 270, 360, fill=color, width=width)
    draw.arc((x0, y1 - 2 * r, x0 + 2 * r, y1), 90, 180, fill=color, width=width)
    draw.arc((x1 - 2 * r, y1 - 2 * r, x1, y1), 0, 90, fill=color, width=width)


def _centered_text(draw: ImageDraw.ImageDraw, text: str, font, cx: float, cy: float, fill):
    """Draw TEXT centred on (cx, cy) without relying on anchor= (bitmap fallback)."""
    try:
        box = draw.textbbox((0, 0), text, font=font)
    except Exception:
        box = (0, 0, 0, 0)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text((cx - w / 2 - box[0], cy - h / 2 - box[1]), text, font=font, fill=fill)


def _logo_asset(size: int) -> Image.Image | None:
    """Owner-supplied mark: assets/logo.png or $ENGLISHAWY_LOGO."""
    if not LOGO_PATH.is_file():
        return None
    try:
        with Image.open(LOGO_PATH) as probe:
            logo = probe.convert("RGBA")
    except (OSError, ValueError):
        return None
    logo.thumbnail((size, size), Image.Resampling.LANCZOS)
    return logo


def _draw_aa_mark(d: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    """Aa — the universal "this is about type" mark: two faces side by side."""
    _centered_text(d, "E", _ui_font(int(size * 0.55), True), x + size * 0.34, y + size * 0.50, HEADER)
    _centered_text(d, "a", _ui_font(int(size * 0.34), True), x + size * 0.70, y + size * 0.66, HEADER)


def _draw_lens_mark(d: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    """Magnifier over a glyph — "identify this letterform"."""
    _centered_text(d, "E", _ui_font(int(size * 0.52), True), x + size * 0.44, y + size * 0.46, HEADER)
    r = size * 0.20
    cx, cy = x + size * 0.68, y + size * 0.64
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=HEADER, width=max(2, round(size * 0.045)))
    d.line((cx + r * 0.7, cy + r * 0.7, x + size * 0.92, y + size * 0.92),
           fill=HEADER, width=max(2, round(size * 0.055)))


def _draw_slash_mark(d: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    """Italic slant behind the E — a type specimen tile."""
    d.line((x + size * 0.60, y + size * 0.12, x + size * 0.40, y + size * 0.88),
           fill=HEADER, width=max(3, round(size * 0.09)))
    _centered_text(d, "E", _ui_font(int(size * 0.52), True), x + size * 0.50, y + size * 0.50, HEADER)


def _draw_logo(d: ImageDraw.ImageDraw, img: Image.Image, x: int, y: int, size: int) -> None:
    custom = _logo_asset(size)
    if custom is not None:
        img.paste(custom, (x + (size - custom.width) // 2, y + (size - custom.height) // 2), custom)
        return
    _round_rect(d, (x, y, x + size, y + size), int(size * 0.27), GOLD)
    if LOGO_VARIANT == "lens":
        _draw_lens_mark(d, x, y, size)
    elif LOGO_VARIANT == "slash":
        _draw_slash_mark(d, x, y, size)
    else:
        _draw_aa_mark(d, x, y, size)


def _truncate(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> str:
    """Ellipsize by measured width so the ellipsis itself cannot overlap a badge."""
    if max_w <= 0:
        return ""
    if draw.textlength(text, font=font) <= max_w:
        return text
    ellipsis = "…"
    limit = max_w - draw.textlength(ellipsis, font=font)
    out = text
    while out and draw.textlength(out, font=font) > limit:
        out = out[:-1]
    return out.rstrip() + ellipsis


def render_card(
    original_crop: Image.Image,
    text: str,
    matches: list,
    top_n: int = 5,
    ocr_confidence: float | None = None,
) -> bytes:
    matches = matches[:top_n]
    row_h = 120
    header_h = 92
    crop_panel_h = 150
    gap = 14
    footer_h = 34
    pad = 24
    card_top = header_h + 18
    content_h = crop_panel_h + gap + len(matches) * (row_h + gap) + footer_h
    card_h = pad + content_h + pad
    total_h = card_top + card_h + MARGIN

    inner_x0 = MARGIN + 18
    inner_x1 = W - MARGIN - 18
    inner_w = inner_x1 - inner_x0

    img = Image.new("RGB", (W, total_h), BG)
    d = ImageDraw.Draw(img)

    # ---- الشريط العلوي البني ----
    d.rectangle((0, 0, W, header_h), fill=HEADER)
    logo = 56
    lx, ly = MARGIN, (header_h - logo) // 2
    _draw_logo(d, img, lx, ly, logo)
    d.text((lx + logo + 16, ly + 4), "ENGLISHAWY", font=_ui_font(25, True), fill=ON_HEADER)
    d.text((lx + logo + 16, ly + 34), "FONT COMPARISON", font=_ui_font(12, True), fill=ON_HEADER_SOFT)
    # شارة TOP N
    badge = f"TOP {len(matches)}"
    bf = _ui_font(13, True)
    bw = int(d.textlength(badge, font=bf)) + 28
    bx = W - MARGIN - bw
    d.rounded_rectangle((bx, ly + 8, bx + bw, ly + 36), radius=14, outline=ON_HEADER_SOFT, width=1)
    d.text((bx + 14, ly + 13), badge, font=bf, fill=ON_HEADER_SOFT)

    # ---- البطاقة الكريمية ----
    _round_rect(d, (MARGIN, card_top, W - MARGIN, card_top + card_h), 22, CARD)
    _dashed_round_rect(d, (MARGIN + 11, card_top + 11, W - MARGIN - 11, card_top + card_h - 11),
                       14, DASH, width=2)

    # ---- خانة القص الأصلي ----
    cy = card_top + pad
    _round_rect(d, (inner_x0, cy, inner_x1, cy + crop_panel_h), 16, PANEL)
    d.text((inner_x0 + 20, cy + 14), "ORIGINAL CROP", font=_ui_font(13, True), fill=LABEL)
    if ocr_confidence is not None:
        ctxt = f'OCR: "{text}"  ·  {ocr_confidence:.0f}%'
        d.text((inner_x0 + 20, cy + crop_panel_h - 26), ctxt, font=_ui_font(12), fill=MUTED)
    # ملاءمة القص داخل الخانة
    box_w, box_h = inner_w - 60, crop_panel_h - 70
    oc = original_crop.convert("RGB")
    s = min(box_w / oc.width, box_h / oc.height, 1.6)
    oc = oc.resize((max(1, int(oc.width * s)), max(1, int(oc.height * s))), Image.Resampling.LANCZOS)
    ox = inner_x0 + (inner_w - oc.width) // 2
    oy = cy + 38 + (box_h - oc.height) // 2
    # Rounded corners so the crop reads as a sample tile, not a raw white rectangle.
    mask = Image.new("L", oc.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, oc.width - 1, oc.height - 1), radius=10, fill=255)
    img.paste(oc, (ox, oy), mask)

    # ---- صفوف المرشحين ----
    y = cy + crop_panel_h + gap
    name_f = _ui_font(19, True)
    prev_missing_f = _ui_font(16)
    for idx, m in enumerate(matches, 1):
        _round_rect(d, (inner_x0, y, inner_x1, y + row_h), 16, ROW)
        # شارة الرقم
        nb = 36
        nx, ny = inner_x0 + 18, y + 16
        _round_rect(d, (nx, ny, nx + nb, ny + nb), 10, GOLD_SOFT)
        nf = _ui_font(15, True)
        num = f"{idx:02d}"
        nw = d.textlength(num, font=nf)
        d.text((nx + (nb - nw) / 2, ny + 9), num, font=nf, fill=GOLD)
        # اسم الخط
        family = str(getattr(m, "family", str(m)))
        source = str(getattr(m, "source", "google"))
        # Never name the upstream service on the card: the source of a match is
        # implementation detail, the family name is what the user came for.
        badge_text = ("FOUND BOTH" if source == "merged" else
                      "OWNER LIBRARY" if source == "owner" else
                      "FOUND ONLINE" if source.startswith("external:") else "GOOGLE FONTS")
        badge_font = _ui_font(10, True)
        badge_w = d.textlength(badge_text, font=badge_font) + 16
        d.rounded_rectangle((inner_x1 - badge_w - 16, y + 15, inner_x1 - 16, y + 35), radius=8, fill=GOLD_SOFT)
        d.text((inner_x1 - badge_w - 8, y + 19), badge_text, font=badge_font, fill=GOLD)
        name_x = nx + nb + 16
        name_limit = inner_x1 - badge_w - 32 - name_x
        family_text = _truncate(d, family, name_f, name_limit)
        d.text((name_x, y + 16), family_text, font=name_f, fill=INK)
        # معاينة: الكلمة نفسها من ملف محلي مطابق، أو شريط نموذج الحروف من المزوّد.
        area_w, area_h = inner_w - 110, row_h - 52
        preview = _fit_preview(text, getattr(m, "font_path", ""), area_w, area_h, getattr(m, "face_index", 0))
        paste_mask = preview
        is_sample = False
        provider_image = getattr(m, "preview_image", None)
        if preview is None and provider_image is not None:
            preview = _fit_provider_preview(provider_image, area_w - 64, area_h)
            is_sample = True
        if preview is not None:
            px = nx + nb + 16
            py = y + 46 + (area_h - preview.height) // 2
            img.paste(preview, (px, py), paste_mask if paste_mask is not None else preview)
            if is_sample:
                # Honest label: this is the family's own specimen, not the user's word.
                d.text((px + preview.width + 10, py + (preview.height - 11) // 2), "sample",
                       font=_ui_font(10), fill=MUTED)
        else:
            d.text((nx + nb + 16, y + 64), "Preview unavailable", font=prev_missing_f, fill=MUTED)
        y += row_h + gap

    # ---- تذييل ----
    fy = y + 2
    note = "Font names + live previews  ·  Send any font file on request"
    d.text((inner_x0 + 6, fy), note, font=_ui_font(12), fill=MUTED)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
