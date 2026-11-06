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
