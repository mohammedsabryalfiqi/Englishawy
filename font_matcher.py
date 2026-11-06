"""
font_matcher.py — مطابقة أشكال الحروف عبر مكتبة خطوط Google والمالك
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
from dataclasses import dataclass, field
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
    source: str = "google"
    face_index: int = 0
    style: str = "Regular"
    preview_image: Image.Image | None = field(default=None, repr=False)
    source_url: str = ""
    # Filled by neural_reranker.blend_scores and by the pipeline merge. Kept as
    # optional fields so every consumer can read them through getattr without
    # caring whether a given match ever went through the neural pass.
    neural_sim: float | None = None
    final_score: float | None = None


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


@lru_cache(maxsize=8192)
def _font_obj(path: str, size: int, face_index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=face_index)


def render_text(font_path: str, text: str, size: int = 160, face_index: int = 0) -> Image.Image | None:
    """يرسم النص بالخط ويعيد صورة رمادية مقصوصة بإحكام (أسود على أبيض)."""
    try:
        font = _font_obj(font_path, size, face_index)
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


def score_pair(ref_img: Image.Image, cand_img: Image.Image) -> tuple[float, dict]:
    """Full metric pass for one candidate: composite score plus its components."""
    metrics = compare_to_reference(ref_img, cand_img)
    return _score(metrics), metrics


# ---- المرحلة الخفيفة قبل المقارنة الكاملة ---------------------------------
# The library can hold thousands of faces (owner uploads + Google). Rendering
# and fully comparing every one of them at 160 px is what made the library
# stage feel slow. A small render plus a cheap overlap score cuts the list to
# a shortlist, and only the survivors pay for the full metric pass.

COARSE_SIZE = 56
COARSE_W, COARSE_H = 160, 48


def _coarse_signature(ref_img: Image.Image) -> tuple[np.ndarray, float]:
    ref = _tight(_ink(ref_img))
    height, width = ref.shape
    aspect = width / max(1, height)
    thumb = cv2.resize(ref, (COARSE_W, COARSE_H), interpolation=cv2.INTER_AREA)
    return thumb, aspect


def _coarse_score(signature: tuple[np.ndarray, float], cand: Image.Image) -> float:
    ref_thumb, ref_aspect = signature
    ink = _tight(_ink(cand))
    height, width = ink.shape
    aspect_diff = abs(np.log((width / max(1, height) + 1e-6) / (ref_aspect + 1e-6)))
    thumb = cv2.resize(ink, (COARSE_W, COARSE_H), interpolation=cv2.INTER_AREA)
    inter = float(np.minimum(ref_thumb, thumb).sum())
    union = float(np.maximum(ref_thumb, thumb).sum())
    iou = inter / union if union > 0 else 0.0
    return 0.72 * iou + 0.28 * float(np.exp(-aspect_diff * 1.4))


def _report(progress, done: int, total: int) -> None:
    if progress is None:
        return
    try:
        progress(done, total)
    except Exception:  # a UI progress bar must never break a search
        pass


def match_fonts(
    ref_img: Image.Image,
    text: str,
    top_k: int = 30,
    limit: int | None = None,
    progress=None,
    fonts: list[dict] | None = None,
) -> tuple[list[FontMatch], int]:
    index = fonts if fonts is not None else load_index()
    if limit and fonts is None:
        index = index[:limit]
    total = len(index)
    if total == 0:
        return [], 0

    # المرحلة الأولى: رسم صغير + درجة تقاطع رخيصة لكل الخطوط. هذا يحوّل
    # المسح من O(N × المقارنة الكاملة) إلى O(N × رسم صغير) + مقارنة كاملة
    # لقائمة قصيرة فقط.
    signature = _coarse_signature(ref_img)
    shortlisted: list[tuple[float, dict, str, int]] = []
    for i, entry in enumerate(index):
        path = entry.get("path") or str(CACHE_DIR / entry["file"])
        face_index = int(entry.get("face_index", 0))
        small = render_text(path, text, size=COARSE_SIZE, face_index=face_index)
        if small is not None:
            shortlisted.append((_coarse_score(signature, small), entry, path, face_index))
        if i % 50 == 0:
            _report(progress, i + 1, total)
    _report(progress, total, total)

    shortlisted.sort(key=lambda item: item[0], reverse=True)
    # سقف القائمة القصيرة: كافٍ لالتقاط أفضل المرشحين لكل نمط، وقليل بما
    # يكفي لجعل المقارنة الكاملة سريعة. لا يقلّ عن 120 حتى لمكتبة صغيرة.
    keep = min(len(shortlisted), max(4 * max(1, top_k), 120))

    results: list[FontMatch] = []
    for coarse, entry, path, face_index in shortlisted[:keep]:
        cand = render_text(path, text, face_index=face_index)
        if cand is None:
            continue
        score, m = score_pair(ref_img, cand)
        results.append(
            FontMatch(
                slug=entry["slug"],
                family=entry.get("family", entry["slug"]),
                category=entry.get("category", ""),
                font_path=path,
                score=score,
                iou=m["iou"],
                aspect_diff=m["aspect_diff"],
                weight_diff=m["weight_diff"],
                edge_sim=m["edge_sim"],
                source=entry.get("source", "google"),
                face_index=face_index,
                style=entry.get("style", "Regular"),
            )
        )
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k], total
