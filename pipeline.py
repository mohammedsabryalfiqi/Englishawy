"""Dual-source pipeline (v6):
1) External provider search  (network)   --┐
2) Owner/Google library + DINOv2 rerank   --┤ run at the same time
                                           └→ merged into one ranked top-N
The external lookup and the local shape sweep used to run one after the
other, and whenever the provider answered above threshold the library was
skipped entirely — so the two never appeared together. They now start
together and Reciprocal Rank Fusion turns both ranked lists into one.
"""
from __future__ import annotations

import io
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

import numpy as np

import neural_reranker
from comparison_card import render_card
from config import cascade_threshold
from external_font_search import search_external_first
from font_matcher import FontMatch, match_fonts, render_text
from font_preview import specimen_preview
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


# Reciprocal Rank Fusion: both sources answer with a ranked list, but their
# numbers are not comparable — the provider returns synthetic decay, the local
# matcher returns absolute shape scores. Ranks are the common language.
RRF_K = 10
QUALITY_FLOOR = 0.55      # below this a match is filler, not a candidate
LOW_CONFIDENCE = 0.62     # tell the user when even the best hit is shaky

_STYLE_WORDS = {
    "regular", "normal", "book", "roman", "medium", "semibold", "demibold", "bold",
    "extrabold", "ultrabold", "black", "heavy", "light", "extralight", "ultralight",
    "thin", "hairline", "italic", "oblique", "condensed", "extended", "narrow", "wide",
}


def _family_key(value: str) -> str:
    """'Nexa Rust Script L Bold Italic' and 'Nexa Rust Script L' are one family."""
    words = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split()
    kept = [w for w in words if w not in _STYLE_WORDS]
    return "".join(kept) or _name_key(value)


def _rank_strength(rank: int) -> float:
    return max(0.35, 1.0 - 0.08 * max(0, rank))


def _merge_sources(external: list[FontMatch], local: list[FontMatch]) -> list[FontMatch]:
    """One ranked list out of both sources, one entry per family.

    A family both sources name collects both rank contributions and rises
    above anything a single source alone likes — which is the care the top-5
    needs: agreement between the provider and the shape matcher is far safer
    than a lone provider hit. The local entry is preferred as the carrier
    because it is the one that has a real font file to render and to send.
    """
    rrf: dict[str, float] = {}
    ext_hit: dict[str, tuple[int, FontMatch]] = {}
    loc_hit: dict[str, tuple[int, FontMatch]] = {}
    for bucket, items in ((ext_hit, external), (loc_hit, local)):
        for rank, match in enumerate(items):
            key = _family_key(match.family)
            if not key:
                continue
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            bucket.setdefault(key, (rank, match))

    merged: list[FontMatch] = []
    for key, _ in sorted(rrf.items(), key=lambda item: item[1], reverse=True):
        local_hit = loc_hit.get(key)
        ext = ext_hit.get(key)
        if local_hit and ext:
            match = local_hit[1]
            match.source = "merged"
            match.source_url = ext[1].source_url or match.source_url
            match.final_score = round(min(0.98, 0.62 * match.score
                                          + 0.38 * _rank_strength(ext[0])), 4)
        elif local_hit:
            match = local_hit[1]
            match.final_score = round(match.score, 4)
        else:
            match = ext[1]
            # Unverified by shape: the provider's own ordering is all we have,
            # so it never claims the confidence a verified family gets.
            match.final_score = round(min(0.90, 0.45 + 0.35 * match.score), 4)
        merged.append(match)
    return merged


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
    cancel: threading.Event | None = None,
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
    logger.info("stage preprocess done %.1fs", timings["preprocess"])

    t = time.time()
    if text_override and text_override.strip():
        ocr = OCRResult(text=text_override.strip(), confidence=100.0, notes="نص يدوي من المستخدم")
    else:
        ocr = read_text(prep.variants)
    timings["ocr"] = round(time.time() - t, 2)
    logger.info("stage ocr done %.1fs text=%r", timings["ocr"], (ocr.text or "")[:40])

    matches: list[FontMatch] = []
    count = len(fonts) if fonts is not None else 0
    backend_name = "shape-only"
    winning_stage = "owner_library"
    fallback_used = True
    notices: list[str] = []

    external_list: list[FontMatch] = []
    local_list: list[FontMatch] = []
    specimens: dict[str, tuple[str, ...]] = {}
    cascade = None

    def run_local() -> list[FontMatch]:
        nonlocal backend_name, count
        t0 = time.time()
        found, total = match_fonts(prep.reference_crop, ocr.text, top_k=candidates,
                                   limit=font_limit, progress=progress, fonts=fonts,
                                   cancel=cancel)
        count = total
        timings["shape_match"] = round(time.time() - t0, 2)
        logger.info("stage library-scan done %.1fs fonts=%d", timings["shape_match"], total)
        if not found:
            return found
        t1 = time.time()
        try:
            found = neural_reranker.rerank(
                prep.reference_crop, found,
                render_fn=lambda path, face=0: render_text(path, ocr.text, size=160, face_index=face),
                backend=neural_backend,
            )
            if any(getattr(m, "neural_sim", None) is not None for m in found):
                backend_name = "DINOv2 ZeroGPU" if neural_backend is not None else "DINOv2 CPU ONNX"
        except Exception:
            # The neural pass is a bonus; shape metrics alone must still answer.
            notices.append("تعذر إعادة الترتيب العصبي؛ النتيجة مبنية على تحليل الشكل فقط.")
        timings["neural_rerank"] = round(time.time() - t1, 2)
        logger.info("stage neural-rerank done %.1fs", timings["neural_rerank"])
        return found

    # ---- Both sources, at the same time -----------------------------------
    # The external lookup is network-bound and the library sweep is CPU-bound,
    # so running them one after the other doubled the wait for nothing. They
    # now start together and the merge happens once both have answered.
    want_external = bool(external_first and ocr.text)
    want_local = bool(ocr.text)
    if want_external:
        # The API is told this is a tight crop. Send the actual crop, not a full
        # Telegram screenshot with unrelated UI/background around the word.
        crop_buffer = io.BytesIO()
        prep.original_crop.save(crop_buffer, format="PNG")
        crop_bytes = crop_buffer.getvalue()

    if want_external and want_local:
        t_par = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            ext_future = pool.submit(search_external_first, crop_bytes,
                                     ocr_text=ocr.text, threshold=cascade_threshold())
            loc_future = pool.submit(run_local)
            cascade = ext_future.result()
            local_list = loc_future.result()
        timings["both_sources"] = round(time.time() - t_par, 2)
    elif want_external:
        cascade = search_external_first(crop_bytes, ocr_text=ocr.text, threshold=cascade_threshold())
    elif want_local:
        local_list = run_local()

    logger.info("stage sources done; external=%s", "yes" if cascade is not None else "no")
    if cascade is not None:
        notices.extend(cascade.notices)
        timings.update(cascade.timings)
        for ext in cascade.matches:
            path, face, cat = _local_path_for(ext.family, fonts, ext.style)
            external_list.append(FontMatch(
                slug=f"ext-{ext.source}-{ext.family}".lower().replace(" ", "-")[:80],
                family=ext.family, category=cat, font_path=path,
                score=ext.score, iou=ext.score, aspect_diff=0.0,
                weight_diff=0.0, edge_sim=ext.score,
                source=f"external:{ext.source}", face_index=face, style=ext.style,
                source_url=ext.source_url,
            ))
            if ext.preview_images:
                specimens.setdefault(_family_key(ext.family), ext.preview_images)

    # ---- Merge ------------------------------------------------------------
    merged = _merge_sources(external_list, local_list)
    strong = [m for m in merged if (m.final_score or 0.0) >= QUALITY_FLOOR]
    if len(strong) >= min(top_n, 3):
        merged = strong
    matches = merged[:top_n]

    if external_list and local_list:
        winning_stage = "merged"
        fallback_used = False
    elif external_list:
        winning_stage = cascade.winning_stage if cascade else "whatfontis"
        fallback_used = False
    else:
        winning_stage = "owner_library"
        fallback_used = True

    best_confidence = max((m.final_score or 0.0) for m in matches) if matches else 0.0
    if matches and best_confidence < LOW_CONFIDENCE:
        notices.append("الثقة في النتيجة منخفضة — راجع المرشحين قبل الاعتماد عليهم.")

    # The provider's own specimen strip is the only real rendering still
    # available for a commercial family; fetch it only for the entries that
    # made the final list and have no local file to draw from.
    pending = [m for m in matches
               if not m.font_path and m.preview_image is None
               and _family_key(m.family) in specimens]
    if pending:
        with ThreadPoolExecutor(max_workers=min(4, len(pending))) as pool:
            previews = list(pool.map(lambda m: specimen_preview(specimens[_family_key(m.family)]), pending))
        for match, preview in zip(pending, previews):
            match.preview_image = preview
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
