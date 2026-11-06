"""
ocr_engine.py — قراءة الحروف بنموذج Tesseract العصبي + تصويت + تصحيح قاموسي
===========================================================================
الخطوات (كما في وصف سير العمل):
  1. تشغيل Tesseract على الصورة بعدة معالجات مسبقة (من preprocessor) وبعدة أوضاع
     تقسيم صفحة (PSM)، وجمع كل النصوص المرشحة مع درجة ثقة كل منها.
  2. دمج النتائج بالتصويت: كل نص يحصل على وزن = مجموع (ثقة القراءات التي أنتجته) +
     مكافأة تكرار + مكافأة لغوية (متوسط تردد كلماته في القاموس).
  3. تصحيح الكلمات بقاموس إنجليزي ضخم (wordfreq) عبر أقرب مسافة تحرير (rapidfuzz)،
     مع الحفاظ على حالة الأحرف الأصلية (Capital / UPPER / lower).

لا يرسل أي بيانات لأي خدمة خارجية؛ كل المعالجة محلية.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import tesserocr
from PIL import Image
from rapidfuzz import fuzz, process
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("ENGLISHAWY_CACHE", ROOT / ".cache" / "englishawy"))
TESSDATA_DIR = Path(os.getenv("TESSDATA_PREFIX", CACHE_DIR / "tessdata"))

# أوضاع تقسيم الصفحة التي نجربها (سطر واحد / كتلة / سطر خام / متناثر)
PSM_MODES = (
    tesserocr.PSM.SINGLE_LINE,
    tesserocr.PSM.SINGLE_BLOCK,
    tesserocr.PSM.RAW_LINE,
    tesserocr.PSM.SPARSE_TEXT,
)

_CLEAN_RE = re.compile(r"[^A-Za-z0-9'’\- .,!?&]")
_WS_RE = re.compile(r"\s+")


@dataclass
class OCRResult:
    text: str
    confidence: float                      # 0..100
    raw_candidates: list[tuple[str, float]] = field(default_factory=list)
    corrected: bool = False
    notes: str = ""


@lru_cache(maxsize=1)
def _dictionary() -> tuple[list[str], dict[str, float]]:
    words = top_n_list("en", 60000)
    freq = {w: zipf_frequency(w, "en") for w in words}
    return words, freq


@lru_cache(maxsize=1)
def _api() -> tesserocr.PyTessBaseAPI:
    path = str(TESSDATA_DIR)
    if not (TESSDATA_DIR / "eng.traineddata").exists():
        # محاولة استخدام بيانات النظام إن وُجدت
        for alt in ("/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tessdata"):
            if (Path(alt) / "eng.traineddata").exists():
                path = alt
                break
    api = tesserocr.PyTessBaseAPI(path=path, lang="eng", oem=tesserocr.OEM.LSTM_ONLY)
    api.SetVariable("preserve_interword_spaces", "1")
    api.SetVariable("load_system_dawg", "0")
    api.SetVariable("load_freq_dawg", "0")
    return api


_API_LOCK = threading.Lock()


def _clean(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = _CLEAN_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _match_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected[:1].upper() + corrected[1:]
    return corrected


def _correct_word(word: str, words: list[str], freq: dict[str, float]) -> tuple[str, bool]:
    core = word.strip(".,!?&'’-")
    if not core or not core.isalpha():
        return word, False
    low = core.lower()
    raw_f = freq.get(low, 0.0)
    if raw_f >= 2.0:                      # كلمة صحيحة وشائعة بما يكفي
        return word, False
    if len(low) <= 2:
        return word, False
    min_sim = 78 if len(low) <= 5 else 70
    hits = process.extract(low, words, scorer=fuzz.ratio, limit=8)
    best, best_score = None, -1.0
    for cand, sim, _ in hits:
        if sim < min_sim:
            continue
        cf = freq.get(cand, 0.0)
        # رجّح التردد لكن اشترط تشابهاً شكلياً قوياً حتى لا نغيّر كلمات صحيحة
        score = 0.55 * sim + 0.45 * (cf / 7.0 * 100)
        if cf > raw_f + 0.4 and score > best_score:
            best, best_score = cand, score
    if best is None:
        return word, False
    fixed = _match_case(core, best)
    # أعد إلحاق علامات الترقيم التي أزلناها
    lead = word[: len(word) - len(word.lstrip(".,!?&'’-"))]
    trail = word[len(word.rstrip(".,!?&'’-")) :]
    return lead + fixed + trail, True


def _language_score(text: str, freq: dict[str, float]) -> float:
    toks = [t for t in re.findall(r"[A-Za-z]+", text.lower())]
    if not toks:
        return 0.0
    return sum(freq.get(t, 0.0) for t in toks) / len(toks)


def read_text(variants: list[tuple[str, Image.Image]]) -> OCRResult:
    words, freq = _dictionary()
    runs: list[tuple[str, float]] = []
    with _API_LOCK:
        api = _api()
        for _name, img in variants:
            for psm in PSM_MODES:
                try:
                    api.SetImage(img.convert("L"))
                    api.SetVariable("tessedit_pageseg_mode", str(int(psm)))
                    txt = _clean(api.GetUTF8Text())
                    conf = float(api.MeanTextConf())
                except Exception:
                    continue
                if txt and any(c.isalnum() for c in txt):
                    runs.append((txt, conf))

    if not runs:
        return OCRResult(text="", confidence=0.0, notes="لم يقرأ Tesseract أي نص")

    # تجميع بالتصويت على النص المنظّف (بدون حساسية حالة لتجميع المرشحين)
    buckets: dict[str, dict] = {}
    for txt, conf in runs:
        key = txt.lower()
        b = buckets.setdefault(key, {"text": txt, "weight": 0.0, "conf_sum": 0.0, "n": 0, "best": 0.0})
        b["weight"] += max(conf, 1.0)
        b["conf_sum"] += conf
        b["n"] += 1
        b["best"] = max(b["best"], conf)
        # احتفظ بالتمثيل الأكثر شيوعاً من حيث حالة الأحرف
        if conf > b.get("rep_conf", -1):
            b["text"] = txt
            b["rep_conf"] = conf

    scored = []
    for key, b in buckets.items():
        lang = _language_score(b["text"], freq)
        score = b["weight"] + 12.0 * b["n"] + 9.0 * lang
        scored.append((score, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    winner = scored[0][1]
    text = winner["text"]

    # تصحيح قاموسي كلمة-كلمة
    changed = False
    out_tokens = []
    for tok in text.split(" "):
        fixed, ch = _correct_word(tok, words, freq)
        out_tokens.append(fixed)
        changed = changed or ch
    text = " ".join(out_tokens)

    avg_conf = winner["conf_sum"] / max(1, winner["n"])
    # ثقة نهائية تمزج ثقة Tesseract مع agreement ومعقولية لغوية
    lang = _language_score(text, freq)
    agreement = winner["n"] / max(1, len(runs))
    confidence = max(0.0, min(99.0, 0.55 * avg_conf + 25.0 * agreement + 4.0 * lang))

    cands = [(b["text"], round(b["conf_sum"] / max(1, b["n"]), 1)) for _, b in scored[:6]]
    return OCRResult(
        text=text,
        confidence=round(confidence, 1),
        raw_candidates=cands,
        corrected=changed,
        notes=f"{len(runs)} قراءة / {len(buckets)} مرشح",
    )
