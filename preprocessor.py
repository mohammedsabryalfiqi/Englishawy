"""
preprocessor.py — المعالجة البصرية المسبقة لعزل النص وتجهيزه للـ OCR والمطابقة
==============================================================================
ينتج:
  *.reference_crop*: نسخة ثنائية نظيفة ومقصوصة بإحكام من الكلمة (تُستخدم في مطابقة
   الأشكال وفي خانة "ORIGINAL CROP" ببطاقة المقارنة).
  *variants*: عدة معالجات مختلفة للصورة تُمرَّر كلها إلى Tesseract ثم يُصوّت على
   نتائجها (هذا ما يرفع دقة قراءة الحروف بشكل كبير مقارنة بمعالجة واحدة).

لا يعتمد على أي خدمة خارجية؛ OpenCV + Pillow فقط.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class PreprocessResult:
    reference_crop: Image.Image          # ثنائي (أسود على أبيض) مقصوص بإحكام
    original_crop: Image.Image           # قص أصلي بالألوان للعرض في البطاقة
    variants: list[tuple[str, Image.Image]] = field(default_factory=list)
    inverted: bool = False
    summary: str = ""


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(img))


def _ensure_dark_on_light(gray: np.ndarray, force_invert: bool) -> tuple[np.ndarray, bool]:
    """يقلب الألوان إذا كانت الخلفية داكنة (الحروف يجب أن تكون داكنة على فاتح)."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    inv = force_invert or bg < 128
    if inv:
        gray = 255 - gray
    return gray, inv


def _tight_crop(binary: np.ndarray, pad: int = 6) -> np.ndarray:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return binary
    x0, x1 = max(0, xs.min() - pad), min(binary.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(binary.shape[0], ys.max() + pad + 1)
    return binary[y0:y1, x0:x1]


def _otsu(gray: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th  # حبر=0 خلفية=255


def preprocess(
    image_bytes: bytes,
    force_invert: bool = False,
    force_binarize: bool = False,
) -> PreprocessResult:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    # تكبير الصور الصغيرة يساعد Tesseract كثيراً
    scale = 1.0
    if max(w, h) < 700:
        scale = 700 / max(w, h)
    elif max(w, h) > 2200:
        scale = 2200 / max(w, h)
    if scale != 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    gray = _to_gray(img)
    gray, inverted = _ensure_dark_on_light(gray, force_invert)

    # قص منطقة النص من الصورة الأصلية للعرض
    bin0 = _otsu(gray)
    ink0 = (bin0 == 0).astype(np.uint8)
    ink0 = cv2.morphologyEx(ink0, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ys, xs = np.where(ink0 > 0)
    if len(xs):
        pad = 10
        x0, x1 = max(0, xs.min() - pad), min(gray.shape[1], xs.max() + pad + 1)
        y0, y1 = max(0, ys.min() - pad), min(gray.shape[0], ys.max() + pad + 1)
        original_crop = img.crop((x0, y0, x1, y1))
        gray_c = gray[y0:y1, x0:x1]
    else:
        original_crop = img
        gray_c = gray

    # المرجع الثنائي المقصوص بإحكام (للمطابقة الشكلية)
    ref_bin = _otsu(gray_c)
    if force_binarize:
        ref_bin = cv2.threshold(gray_c, 140, 255, cv2.THRESH_BINARY)[1]
    ref_ink = _tight_crop((ref_bin == 0).astype(np.uint8) * 255, pad=4)
    ref_img = Image.fromarray(255 - ref_ink).convert("L")  # أسود على أبيض

    # بناء عدة معالجات للتصويت في OCR
    g = Image.fromarray(gray_c)
    big = g.resize((g.width * 2, g.height * 2), Image.Resampling.LANCZOS)
    auto = ImageOps.autocontrast(g, cutoff=1)
    auto_big = auto.resize((auto.width * 2, auto.height * 2), Image.Resampling.LANCZOS)
    sharp = auto_big.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=2))
    otsu_big = Image.fromarray(_otsu(np.asarray(auto_big)))
    adapt = cv2.adaptiveThreshold(
        np.asarray(auto_big), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )
    adapt_img = Image.fromarray(adapt)
    mild = ImageEnhance.Contrast(auto).enhance(1.4)

    variants = [
        ("gray", g),
        ("auto", auto),
        ("auto2x", auto_big),
        ("sharp2x", sharp),
        ("otsu2x", otsu_big),
        ("adaptive", adapt_img),
        ("contrast", mild),
        ("big2x", big),
    ]

    summary = (
        f"{ref_img.width}x{ref_img.height}px | "
        f"{'معكوس' if inverted else 'طبيعي'} | "
        f"{'ثنائي' if force_binarize else 'رمادي'} | {len(variants)} معالجات"
    )
    return PreprocessResult(
        reference_crop=ref_img,
        original_crop=original_crop.convert("RGB"),
        variants=variants,
        inverted=inverted,
        summary=summary,
    )


def to_ink_array(img: Image.Image) -> np.ndarray:
    """يحول صورة (حبر داكن على فاتح) إلى مصفوفة حبر 0/1 بإحكام."""
    a = np.asarray(ImageOps.grayscale(img))
    ink = (a < 160).astype(np.uint8)
    return _tight_crop(ink, pad=2)
