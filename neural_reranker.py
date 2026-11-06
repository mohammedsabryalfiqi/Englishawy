"""DINOv2 top-30 reranking with interchangeable ZeroGPU and CPU ONNX backends.

Preprocessing/rendering stays on CPU. Backends receive normalized float32 arrays
and return plain cosine scores, which is safe across ZeroGPU's worker boundary.
The final blend remains 45% normalized shape + 55% normalized neural similarity.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from config import CACHE_DIR

MODEL_PATH = Path(os.getenv("DINOV2_ONNX_PATH", CACHE_DIR / "models" / "dinov2_vits14.onnx"))
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
_session = None
_lock = threading.Lock()
_input_name = "image"


def available() -> bool:
    return MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000


def _get_session():
    global _session, _input_name
    if _session is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.getenv("DINOV2_THREADS", "4"))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(str(MODEL_PATH), so, providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
    return _session


def _prep(img: Image.Image) -> np.ndarray:
    g = ImageOps.grayscale(img)
    a = np.asarray(g)
    ys, xs = np.where(a < 220)
    if len(xs):
        g = g.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    side = max(g.size)
    pad = int(side * 0.18) + 8
    canvas = Image.new("L", (g.width + 2 * pad, g.height + 2 * pad), 255)
    canvas.paste(g, (pad, pad))
    side = max(canvas.size)
    square = Image.new("L", (side, side), 255)
    square.paste(canvas, ((side - canvas.width) // 2, (side - canvas.height) // 2))
    rgb = ImageOps.colorize(square, black="#101010", white="#ffffff").convert("RGB")
    rgb = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    arr = (np.asarray(rgb, dtype=np.float32) - _MEAN) / _STD
    return arr.transpose(2, 0, 1)


def embed(img: Image.Image) -> np.ndarray:
    """Compatibility helper for single-image CPU inference."""
    with _lock:
        session = _get_session()
        out = session.run(None, {_input_name: _prep(img)[None, ...].copy()})[0]
    vector = np.asarray(out, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def cpu_similarities(reference: np.ndarray, candidates: np.ndarray) -> list[float]:
    """The mirrored ONNX graph has a fixed batch of one; loop without changing it."""
    with _lock:
        session = _get_session()
        vectors = [session.run(None, {_input_name: np.ascontiguousarray(x[None, ...])})[0].reshape(-1)
                   for x in np.concatenate([reference, candidates], axis=0)]
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return (vectors[1:] @ vectors[0]).astype(float).tolist()


def prepare_inputs(ref_img: Image.Image, candidates: list, render_fn: Callable):
    arrays, valid = [], []
    for candidate in candidates:
        image = render_fn(candidate.font_path, getattr(candidate, "face_index", 0))
        valid.append(image is not None)
        arrays.append(_prep(image) if image is not None else np.zeros((3, 224, 224), dtype=np.float32))
    reference = _prep(ref_img)[None, ...]
    candidate_array = np.stack(arrays) if arrays else np.empty((0, 3, 224, 224), dtype=np.float32)
    return reference, candidate_array, valid


def blend_scores(candidates: list, scores: list[float], valid: list[bool]) -> list:
    if len(scores) != len(candidates) or len(valid) != len(candidates):
        raise ValueError("Neural backend returned an invalid score count")
    sims = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(sims).all():
        raise RuntimeError("Neural backend returned non-finite scores")
    sims = np.where(valid, sims, 0.0)
    shape = np.asarray([c.score for c in candidates], dtype=np.float32)
    shape_norm = (shape - shape.min()) / (shape.max() - shape.min() + 1e-6)
    neural_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-6)
    for candidate, sim, sh, nn, is_valid in zip(candidates, sims, shape_norm, neural_norm, valid):
        candidate.neural_sim = round(float(sim), 4) if is_valid else None
        candidate.final_score = round(float(0.45 * sh + 0.55 * nn), 4)
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates


def rerank(ref_img: Image.Image, candidates: list, render_fn: Callable,
           backend: Callable[[np.ndarray, np.ndarray], list[float]] | None = None) -> list:
    if not candidates:
        return candidates
    backend = backend or cpu_similarities
    if backend is cpu_similarities and not available():
        for candidate in candidates:
            candidate.neural_sim = None
            candidate.final_score = candidate.score
        return candidates
    reference, arrays, valid = prepare_inputs(ref_img, candidates, render_fn)
    if not any(valid):
        raise RuntimeError("تعذر رسم أي مرشح لإعادة الترتيب العصبي")
    return blend_scores(candidates, backend(reference, arrays), valid)
