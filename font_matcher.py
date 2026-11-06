"""
font_matcher.py — مطابقة أشكال الحروف عبر مكتبة خطوط Google المحلية
===================================================================
سير العمل (المرحلة الثانية):
  1. رسم النص الذي قرأه الـ OCR بكل خط من المكتبة (~1800 خط، قابلة للتقليص).
  2. مقارنة كل رسمة بالصورة المرجعية بعدة مقاييس شكلية:
     - تطابق القناع (IoU) بعد تحجيم/محاذاة موحّدة.
     - تشابه الحواف (Chamfer) لالتقاط تفاصيل الحروف.
     - نسبة العرض إلى الارتفاع (Aspect Ratio).
     - سُمك الخط / كثافة الحبر (Stroke Weight) عبر distance transform.
     - بروفايل الإسقاط الأفقي/Rأسي (projection profiles) لإيقاع الحروف.
  3. إخراج أفضل المرشحين (افتراضياً 30) لتعيد المرحلة العصبية ترتيبها.

كل الرسم والمقارنة محلياً بـ Pillow + OpenCV؛ لا روابط ولا خدمات خارجية.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("ENGLISHAWY_CACHE", ROOT / ".cache" / "englishawy"))
FONTS_DIR = CACHE_DIR / "fonts"

NORM_H = 96          # ارتفاع التوحيد للمقارنة
NORM_W = 512         # عرض اللوحة الموحّدة


@dataclass
class FontMatch:
    slug: str
    family: str
    category: str
    font_path: str
    score: float                 # درجة مركّبة 0..1
    iou: float
    aspect_diff: float
    weight_diff: float
    edge_sim: float


@lru_cache(maxsize=1)
def load_index() -> list[dict]:
    idx = FONTS_DIR / "index.json"
    if idx.exists():
        return json.loads(idx.read_text(encoding="utf-8"))
    # بناء فهرس ارتجالي من الملفات الموجودة + المنيفست
    manifest = ROOT / "font_manifest.json"
    meta = {}
    if manifest.exists():
        meta = {e["slug"]: e for e in json.loads(manifest.read_text(encoding="utf-8"))}
    out = []
    for p in sorted(FONTS_DIR.glob("*.ttf")):
        e = meta.get(p.stem, {"slug": p.stem, "family": p.stem, "category": "sans-serif"})
        out.append({**e, "file": f"fonts/{p.name}"})
    return out


@lru_cache(maxsize=4096)
def _font_obj(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def render_text(font_path: str, text: str, size: int = 160) -> Image.Image | None:
    """يرسم النص بالخط ويعيد صورة رمادية مقصوصة بإحكام (أسود على أبيض)."""
    try:
        font = _font_obj(font_path, size)
        bbox = font.getbbox(text)
        if not bbox:
            return None
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return None
        pad = max(8, size // 8)
        img = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
        d = ImageDraw.Draw(img)
        d.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=0)
        a = np.asarray(img)
        ys, xs = np.where(a < 200)
        if len(xs) == 0:
            return None
        img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        return img
    except Exception:
        return None


def _ink(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return 1.0 - a


def _resize_norm(ink: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA)


def _stroke_weight(ink: np.ndarray) -> float:
    """متوسط سُمك الضربة مقدّراً عبر distance transform (نسبي للارتفاع)."""
    binary = (ink > 0.4).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return 0.0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    # خطوط الحجب (مثل Flow Block) تملأ اللوحة بالكامل فيعيد distanceTransform قيمة
    # FLT_MAX؛ نقصّ القيم عند ارتفاع اللوحة حتى لا يحدث overflow في التجميع.
    dist = np.minimum(dist, float(binary.shape[0]))
    vals = dist[binary > 0]
    return float(np.mean(vals, dtype=np.float64)) / max(1, binary.shape[0])


def _profile(ink: np.ndarray, axis: int) -> np.ndarray:
    p = ink.sum(axis=axis)
    s = p.sum()
    return p / s if s > 0 else p


def _tight(ink: np.ndarray) -> np.ndarray:
    ys, xs = np.where(ink > 0.05)
    if len(xs) == 0:
        return ink
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def compare_to_reference(ref_img: Image.Image, cand_img: Image.Image) -> dict:
    # قصّ بإحكام حول الحبر في الطرفين حتى لا يفسد التبطينَ نسبةَ العرض/المحاذاة
    ref = _tight(_ink(ref_img))
    cand = _tight(_ink(cand_img))
    rh, rw = ref.shape
    ch, cw = cand.shape
    aspect_diff = abs(np.log((cw / max(1, ch) + 1e-6) / (rw / max(1, rh) + 1e-6)))

    # نمدّ الحبر إلى إطار موحّد (تجاهل النسبة هنا؛ النسبة تُعاقَب منفصلة أدناه)
    # فتتم مقارنة topology الحروف نفسها بدون إزاحة أفقية بسبب اختلاف العرض.
    a = cv2.resize(ref, (NORM_W, NORM_H), interpolation=cv2.INTER_AREA)
    b = cv2.resize(cand, (NORM_W, NORM_H), interpolation=cv2.INTER_AREA)

    # IoU ناعم على شدة الحبر (يتسامح مع ضبابية الصور الحقيقية أكثر من العتبة الصلبة)
    inter = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum())
    iou = inter / union if union > 0 else 0.0

    # تشابه الحواف via chamfer على خرائط المسافة
    ea = cv2.Canny((a * 255).astype(np.uint8), 50, 150)
    eb = cv2.Canny((b * 255).astype(np.uint8), 50, 150)
    if np.count_nonzero(ea) > 0 and np.count_nonzero(eb) > 0:
        da = cv2.distanceTransform((ea == 0).astype(np.uint8), cv2.DIST_L2, 3)
        db = cv2.distanceTransform((eb == 0).astype(np.uint8), cv2.DIST_L2, 3)
        da = np.minimum(da, NORM_H)
        db = np.minimum(db, NORM_H)
        cham = (da[eb > 0].mean() + db[ea > 0].mean()) / 2.0
        edge_sim = float(np.exp(-cham / 6.0))
    else:
        edge_sim = 0.0

    # سُمك الضربة يُقاس على نسخة موحّدة الارتفاع (بنسبة محفوظة) لا على الممدودة
    def hnorm(ink: np.ndarray) -> np.ndarray:
        h, w = ink.shape
        nw = max(1, min(NORM_W, int(round(w * (NORM_H / h)))))
        return cv2.resize(ink, (nw, NORM_H), interpolation=cv2.INTER_AREA)

    wdiff = abs(_stroke_weight(hnorm(ref)) - _stroke_weight(hnorm(cand)))
    pa, pb = _profile(a, 0), _profile(b, 0)
    prof_sim = float(np.minimum(pa, pb).sum())  # تقاطع الهيستوجرام = 0..1

    return {
        "iou": float(iou),
        "aspect_diff": float(aspect_diff),
        "weight_diff": float(wdiff),
        "edge_sim": float(edge_sim),
        "profile_sim": prof_sim,
    }


def _score(m: dict) -> float:
    aspect_pen = np.exp(-m["aspect_diff"] * 1.4)
    weight_pen = np.exp(-m["weight_diff"] * 18.0)
    return float(
        0.40 * m["iou"]
        + 0.24 * m["edge_sim"]
        + 0.16 * aspect_pen
        + 0.10 * weight_pen
        + 0.10 * m["profile_sim"]
    )


def match_fonts(
    ref_img: Image.Image,
    text: str,
    top_k: int = 30,
    limit: int | None = None,
    progress=None,
) -> list[FontMatch]:
    index = load_index()
    if limit:
        index = index[:limit]
    results: list[FontMatch] = []
    for i, entry in enumerate(index):
        path = str(CACHE_DIR / entry["file"])
        cand = render_text(path, text)
        if cand is None:
            continue
        m = compare_to_reference(ref_img, cand)
        results.append(
            FontMatch(
                slug=entry["slug"],
                family=entry.get("family", entry["slug"]),
                category=entry.get("category", ""),
                font_path=path,
                score=_score(m),
                iou=m["iou"],
                aspect_diff=m["aspect_diff"],
                weight_diff=m["weight_diff"],
                edge_sim=m["edge_sim"],
            )
        )
        if progress and i % 300 == 0:
            progress(i, len(index))
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]
