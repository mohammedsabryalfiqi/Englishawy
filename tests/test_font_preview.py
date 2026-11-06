import io
import urllib.parse

import pytest
from PIL import Image, ImageDraw, ImageFont

import font_preview
from comparison_card import _truncate, render_card
from font_matcher import FontMatch
from pipeline import _local_path_for


PAGE = '''<a name="ct"></a><form id="sample"></form>
<img src="//d1ly52g9wjvbd2.cloudfront.net/images/spacer.gif"
 data-src="https://imgall.whatfontis.com/index.php?p=refresh&amp;id=638817&amp;text=WhatFontis.com"
 id="result638817" class="lazy">'''


def word_png(text="welcome baby"):
    image = Image.new("RGB", (520, 100), "white")
    ImageDraw.Draw(image).text((15, 30), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_discovers_real_custom_text_endpoint_and_replaces_text():
    endpoint = font_preview.custom_preview_endpoint(PAGE, "welcome baby & more")
    assert endpoint is not None
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(endpoint).query)
    assert query == {"p": ["refresh"], "id": ["638817"], "text": ["welcome baby & more"]}
    assert urllib.parse.urlsplit(endpoint).hostname == "imgall.whatfontis.com"


def test_preview_rejects_untrusted_endpoint_and_blank_image():
    evil = PAGE.replace("imgall.whatfontis.com", "127.0.0.1")
    assert font_preview.custom_preview_endpoint(evil, "hello") is None
    assert font_preview._validated_image(io.BytesIO(Image.new("RGB", (100, 50), "white").tobytes()).getvalue()) is None
    blank = io.BytesIO()
    Image.new("RGB", (100, 50), "white").save(blank, format="PNG")
    assert font_preview._validated_image(blank.getvalue()) is None
    assert font_preview._validated_image(word_png()) is not None


def test_preview_fetches_word_and_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(font_preview, "PREVIEW_DIR", tmp_path)
    calls = []

    def fake_get(url, hosts, limit, timeout, accept, deadline):
        calls.append(url)
        return PAGE.encode() if accept == "text/html" else word_png()

    monkeypatch.setattr(font_preview, "_get", fake_get)
    url = "https://www.whatfontis.com/NMY_Example.font"
    first = font_preview.whatfontis_preview(url, "welcome baby")
    second = font_preview.whatfontis_preview(url, "welcome baby")
    assert first is not None and second is not None
    assert len(calls) == 2
    assert "text=welcome+baby" in calls[0]
    assert "text=welcome+baby" in calls[1]
    assert list(tmp_path.glob("*.png"))


def test_exact_style_lookup_does_not_substitute_different_nexa_face(sample_font):
    fonts = [{"family": "Nexa Script", "style": "Regular", "path": str(sample_font), "face_index": 0},
             {"family": "Nexa Rust Script L", "style": "0", "path": "correct.otf", "face_index": 0}]
    assert _local_path_for("Nexa Rust Script L 0", fonts)[0] == "correct.otf"
    assert _local_path_for("Nexa Rust Script R-Trial 4", fonts)[0] == ""


def test_bold_or_italic_never_falls_back_to_regular_family():
    fonts = [{"family": "Roboto", "style": "Regular", "path": "regular.ttf"},
             {"family": "Roboto", "style": "Bold", "path": "bold.ttf"}]
    assert _local_path_for("Roboto", fonts, "Bold")[0] == "bold.ttf"
    assert _local_path_for("Roboto", fonts, "Regular")[0] == "regular.ttf"
    assert _local_path_for("Roboto", fonts, "Italic")[0] == ""
    assert _local_path_for("Roboto Bold", fonts)[0] == "bold.ttf"


def test_provider_word_preview_appears_in_card_without_font_file():
    match = FontMatch("ext", "Nexa Script", "script", "", 0.9, 0, 0, 0, 0,
                      source="external:WhatFontIs", preview_image=Image.open(io.BytesIO(word_png())))
    card = render_card(Image.new("RGB", (250, 70), "white"), "welcome baby", [match])
    assert card.startswith(b"\x89PNG")
    image = Image.open(io.BytesIO(card))
    assert image.height > 300
    assert match.preview_image is not None


def test_custom_section_is_preferred_and_fragment_is_dropped():
    earlier = '<img data-src="https://imgall.whatfontis.com/index.php?p=refresh&id=1&text=x">'
    page = earlier + PAGE
    endpoint = font_preview.custom_preview_endpoint(page, "hello")
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(endpoint).query)["id"] == ["638817"]
    assert font_preview._safe_url("https://www.whatfontis.com/NMY_X.font#ct", font_preview.PAGE_HOSTS) == \
        "https://www.whatfontis.com/NMY_X.font"


def test_family_name_truncation_includes_ellipsis_width():
    image = Image.new("RGB", (400, 40), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = "A very long font family name that must not overlap the source badge"
    out = _truncate(draw, text, font, 120)
    assert out.endswith("…")
    assert draw.textlength(out, font=font) <= 120
    assert _truncate(draw, "short", font, 120) == "short"


def test_redirects_cannot_leave_https_allowlist(monkeypatch):
    class Response:
        status_code = 302
        headers = {"Location": "http://127.0.0.1/private"}
        def __enter__(self): return self
        def __exit__(self, *args): pass

    calls = []
    def fake_get(url, **kwargs):
        calls.append(url)
        assert kwargs["allow_redirects"] is False
        return Response()

    monkeypatch.setattr(font_preview.requests, "get", fake_get)
    import time
    assert font_preview._get("https://www.whatfontis.com/test", font_preview.PAGE_HOSTS,
                             1000, (1, 1), "text/html", time.monotonic() + 5) is None
    assert len(calls) == 1
