"""
neural_reranker.py — إعادة الترتيب العصبي بنموذج الرؤية DINOv2 (ViT-S/14)
========================================================================
المرحلة الثالثة من سير العمل: نأخذ أفضل 30 مرشحاً من المطابقة الشكلية، ونحسب
بصمة DINOv2 (384 بُعد) لكلٍّ من:
  - الصورة المرجعية (قصّ المستخدم).
  - معاينة كل خط مرشح (نفس النص مرسوماً بالخط).
ثم نعيد الترتيب بمزج التشابه الدلالي العصبي (cosine) مع الدرجة الشكلية، فتطفو
الخطوط الأقرب بصرياًจริงๆ إلى القمة حتى لو خدعت المقاييس الهندسية البسيطة.

يعمل عبر onnxruntime على CPU بدون torch. النموذج يُنزّله setup_assets.py مرة واحدة.
إذا غاب النموذج لأي سبب، يعود الترتيب Shape-only بسلاسة (لا ينهار البوت).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("ENGLISHAWY_CACHE", ROOT / ".cache" / "englishawy"))
MODEL_PATH = Path(os.getenv("DINOV2_ONNX_PATH", CACHE_DIR / "models" / "dinov2_vits14.onnx"))

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

_session = None
_lock = threading.Lock()
_input_name = "image"


def available() -> bool:
    return MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000


def _get_session():
    global _session, _input_name
    if _session is None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.getenv("DINOV2_THREADS", "4"))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            str(MODEL_PATH), so, providers=["CPUExecutionProvider"]
        )
        _input_name = _session.get_inputs()[0].name
    return _session


def _prep(img: Image.Image) -> np.ndarray:
    """يحوّل قصعة نصية إلى إدخال DINOv2: خلفية بيضاء، تبطين مربّع، 224×224، تطبيع ImageNet."""
    g = ImageOps.grayscale(img)
    # قصّ بإحكام حول الحبر ثم تبطين مربّع حتى لا يتشوّه شكل الحروف
    a = np.asarray(g)
    ys, xs = np.where(a < 220)
    if len(xs):
        g = g.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = max(g.size)
    pad = int(side * 0.18) + 8
    canvas = Image.new("L", (g.width + 2 * pad, g.height + 2 * pad), 255)
    canvas.paste(g, (pad, pad))
    side = max(canvas.size)
    sq = Image.new("L", (side, side), 255)
    sq.paste(canvas, ((side - canvas.width) // 2, (side - canvas.height) // 2))
    rgb = ImageOps.colorize(sq, black="#101010", white="#ffffff").convert("RGB")
    rgb = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32)
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[None, ...]


def embed(img: Image.Image) -> np.ndarray:
    with _lock:
        sess = _get_session()
        out = sess.run(None, {_input_name: _prep(img)})[0]
    v = np.asarray(out, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def rerank(ref_img: Image.Image, candidates: list, render_fn) -> list:
    """
    يعيد ترتيب المرشحين (كائنات لها .font_path و .score و .family).
    render_fn(font_path)->PIL.Image يرسم نفس النص بالخط.
    يضيف حقولاً ديناميكية: neural_sim, final_score.
    """
    if not available() or not candidates:
        for c in candidates:
            c.neural_sim = 0.0
            c.final_score = c.score
        return candidates

    ref_vec = embed(ref_img)
    shape_scores = np.array([c.score for c in candidates], dtype=np.float32)
    s_min, s_max = shape_scores.min(), shape_scores.max()
    shape_norm = (shape_scores - s_min) / (s_max - s_min + 1e-6)

    sims = []
    for c in candidates:
        img = render_fn(c.font_path)
        if img is None:
            sims.append(0.0)
            continue
        v = embed(img)
        sims.append(float(np.dot(ref_vec, v)))
    sims = np.array(sims, dtype=np.float32)
    n_min, n_max = sims.min(), sims.max()
    neural_norm = (sims - n_min) / (n_max - n_min + 1e-6)

    for c, sim, sh, nn in zip(candidates, sims, shape_norm, neural_norm):
        c.neural_sim = round(float(sim), 4)
        c.final_score = round(float(0.45 * sh + 0.55 * nn), 4)
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates
