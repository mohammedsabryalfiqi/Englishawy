import io
from pathlib import Path

import pytest
from PIL import Image

from font_matcher import render_text
from font_store import FontStore
from pipeline import identify


def test_owner_only_end_to_end_card_and_no_links(tmp_path, sample_font):
    store = FontStore(tmp_path)
    store.install_font(sample_font)
    entry = store.entries()[0]
    image = render_text(entry["path"], "Question", size=160)
    buffer = io.BytesIO()
    Image.new("RGB", image.size, "white").save(buffer, format="PNG")
    canvas = Image.new("RGB", image.size, "white")
    canvas.paste(image, (0, 0))
    buffer = io.BytesIO(); canvas.save(buffer, format="PNG")
    result = identify(buffer.getvalue(), text_override="Question", fonts=store.entries(), top_n=1)
    assert result.text == "Question"
    assert result.matches[0].source == "owner"
    assert result.card_png.startswith(b"\x89PNG")
    assert b"http://" not in result.card_png and b"https://" not in result.card_png


def test_real_tesseract_reads_clean_owner_render(tmp_path):
    from ocr_engine import TESSDATA_DIR
    from config import CACHE_DIR
    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), CACHE_DIR / "fonts" / "roboto.ttf"]
    real_font = next((p for p in candidates if p.is_file()), None)
    if not (TESSDATA_DIR / "eng.traineddata").exists() or real_font is None:
        pytest.skip("tessdata or a real test font not prepared")
    store = FontStore(tmp_path)
    store.install_font(real_font)
    entry = store.entries()[0]
    glyph = render_text(entry["path"], "Question", size=160)
    canvas = Image.new("RGB", (glyph.width + 60, glyph.height + 60), "white")
    canvas.paste(glyph, (30, 30))
    buffer = io.BytesIO(); canvas.save(buffer, format="PNG")
    result = identify(buffer.getvalue(), fonts=store.entries(), top_n=1)
    assert result.text.lower() == "question"


def _match(slug, family, score, source, style="Regular"):
    from font_matcher import FontMatch
    return FontMatch(slug, family, "sans-serif", "", score, score, 0.0, 0.0, score,
                     source=source, style=style)


def test_family_key_collapses_style_words():
    from pipeline import _family_key
    assert _family_key("Nexa Rust Script L Bold Italic") == _family_key("Nexa Rust Script L")
    assert _family_key("Cairo SemiBold") == _family_key("Cairo")
    assert _family_key("Cairo") != _family_key("Cairo Pro")


def test_merge_prefers_a_family_both_sources_name():
    from pipeline import _merge_sources
    external = [
        _match("ext-other", "Other Font", 0.95, "external:WhatFontIs"),
        _match("ext-nexa", "Nexa Script Bold", 0.80, "external:WhatFontIs", style="Bold"),
    ]
    local = [
        _match("owner-nexa", "Nexa Script", 0.72, "owner"),
        _match("owner-other", "Other Family", 0.60, "owner"),
    ]
    merged = _merge_sources(external, local)
    # The family both sources agree on has to lead — that agreement is what
    # makes the top-5 safe to show in a group, even against a lone rank-0 hit.
    assert merged[0].family == "Nexa Script"
    assert merged[0].source == "merged"
    by_family = {m.family: m for m in merged}
    assert by_family["Nexa Script"].final_score > by_family["Other Font"].final_score
    assert merged.index(by_family["Nexa Script"]) < merged.index(by_family["Other Font"])


def test_merge_agreement_lifts_a_family_above_its_external_only_score():
    from pipeline import _merge_sources
    lonely = _merge_sources([_match("ext-a", "Lonely Online", 1.0, "external:WhatFontIs")], [])
    assert lonely[0].source.startswith("external:")
    assert lonely[0].final_score <= 0.90
    verified = _merge_sources([_match("ext-b", "Both", 0.80, "external:WhatFontIs")],
                              [_match("owner-b", "Both", 0.80, "owner")])
    assert verified[0].source == "merged"
    assert verified[0].final_score > 0.80

