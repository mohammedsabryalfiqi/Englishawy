"""Tesseract voting → Google/owner glyph matching → DINOv2 top-30 reranking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

import neural_reranker
from comparison_card import render_card
from font_matcher import FontMatch, match_fonts, render_text
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
    neural_backend: str = "shape-only"


def identify(
    image_bytes: bytes,
    top_n: int = 5,
    candidates: int = 30,
    force_invert: bool = False,
    force_binarize: bool = False,
    text_override: str | None = None,
    font_limit: int | None = None,
    progress=None,
    fonts: list[dict] | None = None,
    neural_backend: Callable[[np.ndarray, np.ndarray], list[float]] | None = None,
) -> Identification:
    if not 1 <= int(top_n) <= 10 or not 1 <= int(candidates) <= 64:
        raise ValueError("Invalid result/candidate count")
    if text_override is not None and len(text_override) > 160:
        raise ValueError("النص اليدوي يجب ألا يتجاوز 160 حرفاً")
    start = time.time()
    timings = {}
    t = time.time()
    prep = preprocess(image_bytes, force_invert=force_invert, force_binarize=force_binarize)
    timings["preprocess"] = round(time.time() - t, 2)

    t = time.time()
    if text_override and text_override.strip():
        ocr = OCRResult(text=text_override.strip(), confidence=100.0, notes="نص يدوي من المستخدم")
    else:
        ocr = read_text(prep.variants)
    timings["ocr"] = round(time.time() - t, 2)

    matches: list[FontMatch] = []
    count = 0
    backend_name = "shape-only"
    if ocr.text:
        t = time.time()
        matches, count = match_fonts(prep.reference_crop, ocr.text, top_k=candidates,
                                     limit=font_limit, progress=progress, fonts=fonts)
        timings["shape_match"] = round(time.time() - t, 2)
        if matches:
            t = time.time()
            matches = neural_reranker.rerank(
                prep.reference_crop, matches,
                render_fn=lambda path, face=0: render_text(path, ocr.text, size=160, face_index=face),
                backend=neural_backend,
            )
            timings["neural_rerank"] = round(time.time() - t, 2)
            if any(getattr(m, "neural_sim", None) is not None for m in matches):
                backend_name = "DINOv2 ZeroGPU" if neural_backend is not None else "DINOv2 CPU ONNX"

    matches = matches[:top_n]
    card = render_card(prep.original_crop, ocr.text or "—", matches, top_n=top_n,
                       ocr_confidence=ocr.confidence if ocr.text else None)
    return Identification(
        text=ocr.text, ocr=ocr, prep=prep, matches=matches, card_png=card,
        font_count=count, elapsed_s=round(time.time() - start, 2),
        neural_used=backend_name != "shape-only", timings=timings, neural_backend=backend_name,
    )
