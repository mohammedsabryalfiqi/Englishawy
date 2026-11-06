"""Cascade pipeline (v5):
1) MyFonts WhatTheFont   (https://www.myfonts.com/pages/whatthefont)  -- FIRST
2) WhatFontIs            (https://www.whatfontis.com/)                 -- SECOND
3) Owner/Google library + DINOv2 rerank                                 -- FALLBACK
Only when no strong external match (>= CASCADE_THRESHOLD) do we run the
heavy local shape matching. Drop-in replacement for the original pipeline.py.
"""
from __future__ import annotations

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import neural_reranker
from comparison_card import render_card
from config import cascade_threshold
from external_font_search import search_external_first
from font_matcher import FontMatch, match_fonts, render_text
from font_preview import whatfontis_preview
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
    winning_stage: str = "owner_library"  # whatthefont | whatfontis | owner_library
    fallback_used: bool = True
    preview_count: int = 0
    notices: list[str] = field(default_factory=list)


def _name_key(value: str) -> str:
    value = re.sub(r"\.(ttf|otf|woff2?)\b", " ", value, flags=re.I)
    value = re.sub(r"\b(regular|normal|otf|ttf|400)\b", " ", value, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _local_path_for(family: str, fonts: list[dict] | None, style: str = "Regular") -> tuple[str, int, str]:
    """Reuse only an exact family/style match, never a misleading substring.

    For example, 'Nexa Rust Script L 0' must not render in 'Nexa Script' or a
    different shadow/trial face merely because the names overlap.
    """
    if not fonts:
        return "", 0, "sans-serif"
    family_key = _name_key(family)
    full_key = _name_key(f"{family} {style}")
    # Prefer an exact full face name. A Bold/Italic request must never silently
    # render in the family's Regular face just because the family name matches.
    for f in fonts:
        local_family = str(f.get("family", ""))
        local_style = str(f.get("style", "Regular"))
        keys = {_name_key(f"{local_family} {local_style}"),
                _name_key(f"{local_style} {local_family}")} - {""}
        path = str(f.get("path", ""))
        if path and full_key and full_key in keys:
            return path, int(f.get("face_index", 0)), str(f.get("category", "sans-serif"))
    # Family-only fallback is allowed only for an explicitly Regular/normal face.
    if family_key and _name_key(style) == "":
        for f in fonts:
            local_family = str(f.get("family", ""))
            local_style = str(f.get("style", "Regular"))
            path = str(f.get("path", ""))
            if path and _name_key(local_style) == "" and _name_key(local_family) == family_key:
                return path, int(f.get("face_index", 0)), str(f.get("category", "sans-serif"))
    return "", 0, "sans-serif"


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
    external_first: bool = True,
) -> Identification:
    if not 1 <= int(top_n) <= 10 or not 1 <= int(candidates) <= 64:
        raise ValueError("Invalid result/candidate count")
    if text_override is not None and len(text_override) > 160:
        raise ValueError("النص اليدوي يجب ألا يتجاوز 160 حرفاً")
    start = time.time()
    timings: dict = {}

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
    count = len(fonts) if fonts is not None else 0
    backend_name = "shape-only"
    winning_stage = "owner_library"
    fallback_used = True
    notices: list[str] = []

    # ---- Stage 1+2: external sites FIRST ---------------------------------
    if external_first and ocr.text:
        # The API is told this is a tight crop. Send the actual crop, not a full
        # Telegram screenshot with unrelated UI/background around the word.
        crop_buffer = io.BytesIO()
        prep.original_crop.save(crop_buffer, format="PNG")
        cascade = search_external_first(crop_buffer.getvalue(), ocr_text=ocr.text, threshold=cascade_threshold())
        notices.extend(cascade.notices)
        timings.update(cascade.timings)
        if not cascade.fallback_to_owner and cascade.matches:
            winning_stage = cascade.winning_stage
            fallback_used = False
            selected = cascade.matches[:top_n]
            # Fetch the actual-word renderings concurrently. Each worker has a
            # bounded deadline; a failed provider preview never blocks the card.
            with ThreadPoolExecutor(max_workers=min(5, max(1, len(selected)))) as pool:
                futures = [pool.submit(whatfontis_preview, ext.source_url, ocr.text)
                           if ext.source == "WhatFontIs" else None for ext in selected]
                provider_previews = [future.result() if future is not None else None for future in futures]
            for ext, provider_preview in zip(selected, provider_previews):
                local_path, face, cat = _local_path_for(ext.family, fonts, ext.style)
                matches.append(FontMatch(
                    slug=f"ext-{ext.source}-{ext.family}".lower().replace(" ", "-")[:80],
                    family=ext.family, category=cat, font_path=local_path,
                    score=ext.score, iou=ext.score, aspect_diff=0.0,
                    weight_diff=0.0, edge_sim=ext.score,
                    source=f"external:{ext.source}", face_index=face, style=ext.style,
                    preview_image=provider_preview, source_url=ext.source_url,
                ))

    # ---- Stage 3: owner/Google library fallback ---------------------------
    if fallback_used and ocr.text:
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
        winning_stage=winning_stage, fallback_used=fallback_used,
        preview_count=sum(bool(m.font_path) or m.preview_image is not None for m in matches),
        notices=notices,
    )
