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

import io
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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


# كان الكاش 8192 كائن خط مفتوح (كل واحد يُبقي ملف الخط في الذاكرة): مع مكتبة كبيرة
# كانت الرام تنفد ويُعاد تشغيل الـ Space. المسح الخفيف لا يستخدم الكاش أصلاً (cache=False)،
# وهذا الكاش يخدم فقط القائمة القصيرة وإعادة الترتيب العصبي.
@lru_cache(maxsize=int(os.getenv("FONT_OBJ_CACHE", "256")))
def _font_obj(path: str, size: int, face_index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=face_index)


def render_text(font_path: str, text: str, size: int = 160, face_index: int = 0,
                cache: bool = True) -> Image.Image | None:
    """يرسم النص بالخط ويعيد صورة رمادية مقصوصة بإحكام (أسود على أبيض)."""
    try:
        font = (_font_obj(font_path, size, face_index) if cache
                else ImageFont.truetype(font_path, size, index=face_index))
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


SCAN_WORKERS = max(1, int(os.getenv("MATCH_WORKERS", str(min(3, os.cpu_count() or 2)))))
SCAN_CHUNK = max(20, int(os.getenv("MATCH_CHUNK", "300")))          # خطوط لكل عملية فرعية
MAX_FONT_MB = float(os.getenv("MATCH_MAX_FONT_MB", "60"))          # تخطّي الملفات العملاقة فقط
CHUNK_TIMEOUT_S = float(os.getenv("MATCH_CHUNK_TIMEOUT_S", "180"))
USE_SUBPROCESS = os.getenv("MATCH_SUBPROCESS", "1").lower() not in {"0", "false", "no"}
_WORKER = Path(__file__).resolve().with_name("font_scan_worker.py")

# ---- وضع البحث العميق (زر «❌ خاطئة») -------------------------------------
# مرحلة إضافية تُشغَّل على أفضل المرشحين فقط: نفس الخط يُرسم بحجمين وبحالات
# الأحرف المحتملة (OCR كثيراً ما يخطئ في حالة الأحرف)، وتُختار أفضل درجة.
# تكلّف ثوانٍ معدودة وترفع دقة الترتيب دون المساس بزمن البحث العادي.
DEEP_KEEP_MIN = int(os.getenv("MATCH_DEEP_KEEP_MIN", "360"))
DEEP_REFINE_TOP = int(os.getenv("MATCH_DEEP_REFINE_TOP", "40"))
DEEP_SIZES = (160, 112)


def coarse_entry(entry: dict, text: str, signature, max_mb: float = MAX_FONT_MB) -> float | None:
    """Cheap score for ONE font (small render, never cached). None = cannot render / too big."""
    path = entry.get("path") or str(CACHE_DIR / entry["file"])
    try:
        if os.path.getsize(path) > max_mb * 1048576:
            return None
    except OSError:
        return None
    small = render_text(path, text, size=COARSE_SIZE, face_index=int(entry.get("face_index", 0)), cache=False)
    return None if small is None else _coarse_score(signature, small)


def _coarse_chunk_subprocess(batch: list[dict], text: str, ref_hex: str) -> list | None:
    """One chunk in a short-lived child process: memory is returned to the OS afterwards,
    the server's event loop never shares the GIL with the scan, and a crash stays contained."""
    job = {"entries": [{k: e.get(k) for k in ("path", "file", "face_index") if e.get(k) is not None}
                       for e in batch],
           "text": text, "ref_png_hex": ref_hex, "max_mb": MAX_FONT_MB}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        proc = subprocess.run([sys.executable, str(_WORKER)], input=json.dumps(job).encode("utf-8"),
                              capture_output=True, timeout=CHUNK_TIMEOUT_S, env=env, cwd=str(_WORKER.parent))
        return json.loads(proc.stdout.decode("utf-8")) if proc.returncode == 0 else None
    except Exception:
        return None


def _best_full(ref_img: Image.Image, font_path: str, text: str, face_index: int,
               deep: bool = False) -> tuple[float, dict] | None:
    """Full metric pass for one face, over one render size (deep: two sizes)."""
    best: tuple[float, dict] | None = None
    for size in (DEEP_SIZES if deep else (160,)):
        cand = render_text(font_path, text, size=size, face_index=face_index)
        if cand is None:
            continue
        score, metrics = score_pair(ref_img, cand)
        if best is None or score > best[0]:
            best = (score, metrics)
    return best


def _case_variants(text: str) -> list[str]:
    """Letter cases OCR may have got wrong; two at most to bound the cost."""
    out: list[str] = []
    for variant in (text.upper(), text.lower(), text.title()):
        if variant and variant != text and variant not in out:
            out.append(variant)
    return out[:2]


def _deep_refine(ref_img: Image.Image, text: str, results: list[FontMatch],
                 cancel: threading.Event | None = None) -> None:
    """Re-score the leading candidates under alternate letter cases and sizes.

    The reference image is a real screenshot; the OCR text behind it may differ
    in case only (``Baby`` read as ``baby``). One fixed case can sink the right
    font below a worse one that happens to match the typo, so each of the top
    candidates is retried and keeps its best score. Mutates ``results`` in place.
    """
    variants = _case_variants(text)
    if not variants:
        return
    for match in results[:DEEP_REFINE_TOP]:
        if cancel is not None and cancel.is_set():
            return
        for variant in variants:
            for size in DEEP_SIZES:
                cand = render_text(match.font_path, variant, size=size, face_index=match.face_index)
                if cand is None:
                    continue
                score, m = score_pair(ref_img, cand)
                if score > match.score:
                    match.score = score
                    match.iou = m["iou"]
                    match.aspect_diff = m["aspect_diff"]
                    match.weight_diff = m["weight_diff"]
                    match.edge_sim = m["edge_sim"]


def match_fonts(
    ref_img: Image.Image,
    text: str,
    top_k: int = 30,
    limit: int | None = None,
    progress=None,
    fonts: list[dict] | None = None,
    cancel: threading.Event | None = None,
    stats: dict | None = None,
    deep: bool = False,
) -> tuple[list[FontMatch], int]:
    index = fonts if fonts is not None else load_index()
    if limit and fonts is None:
        index = index[:limit]
    total = len(index)
    if total == 0:
        return [], 0

    # المرحلة الأولى: رسم صغير + درجة تقاطع رخيصة لكل الخطوط (المكتبة كلها، بلا حدّ زمني).
    # تُنفَّذ على دفعات في عمليات فرعية متوازية فلا يتجمّد السيرفر ولا تتراكم الذاكرة.
    signature = _coarse_signature(ref_img)
    ref_buf = io.BytesIO()
    ref_img.convert("L").save(ref_buf, format="PNG")
    ref_hex = ref_buf.getvalue().hex()
    scores: list[float | None] = [None] * total
    state = {"done": 0, "failed": 0}
    lock = threading.Lock()

    def run_chunk(start: int) -> None:
        if cancel is not None and cancel.is_set():
            return
        batch = index[start:start + SCAN_CHUNK]
        rows = _coarse_chunk_subprocess(batch, text, ref_hex) if USE_SUBPROCESS else None
        if rows is not None:
            for i, value in rows:
                scores[start + i] = value
        else:  # العملية الفرعية فشلت (أو معطّلة): مسح داخلي لهذه الدفعة فقط
            if USE_SUBPROCESS:
                with lock:
                    state["failed"] += 1
            for i, entry in enumerate(batch):
                if cancel is not None and cancel.is_set():
                    break
                scores[start + i] = coarse_entry(entry, text, signature)
        with lock:
            state["done"] += len(batch)
            _report(progress, min(total, state["done"]), total)

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        list(pool.map(run_chunk, range(0, total, SCAN_CHUNK)))
    if cancel is not None and cancel.is_set():
        raise RuntimeError("أُلغي التحليل")
    _report(progress, total, total)
    shortlisted: list[tuple[float, dict, str, int]] = []
    for value, entry in zip(scores, index):
        if value is not None:
            shortlisted.append((value, entry, entry.get("path") or str(CACHE_DIR / entry["file"]),
                                int(entry.get("face_index", 0))))
    if stats is not None:
        stats.update({"scanned": state["done"], "total": total, "failed_chunks": state["failed"],
                      "skipped": total - len(shortlisted)})

    shortlisted.sort(key=lambda item: item[0], reverse=True)
    # سقف القائمة القصيرة: كافٍ لالتقاط أفضل المرشحين لكل نمط، وقليل بما
    # يكفي لجعل المقارنة الكاملة سريعة. البحث العميق (زر «❌ خاطئة») يوسّعها
    # إلى 360 ويُرسم بحجمين، فيصعد الخط الصحيح حتى لو كان ترتيبه الخشن متوسطاً.
    keep = min(len(shortlisted), max((10 if deep else 6) * max(1, top_k),
                                     max(DEEP_KEEP_MIN, 200) if deep else 200))

    results: list[FontMatch] = []
    for coarse, entry, path, face_index in shortlisted[:keep]:
        if cancel is not None and cancel.is_set():
            break
        best = _best_full(ref_img, path, text, face_index, deep=deep)
        if best is None:
            continue
        score, m = best
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
    if deep:
        _deep_refine(ref_img, text, results, cancel)
        results.sort(key=lambda x: x.score, reverse=True)
        if stats is not None:
            stats["deep"] = {"keep": keep, "refined": min(len(results), DEEP_REFINE_TOP)}
    return results[:top_k], total
