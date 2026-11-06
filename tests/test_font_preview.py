import io
import time

import pytest
from PIL import Image, ImageDraw, ImageFont

import font_preview
from comparison_card import _truncate, render_card
from font_matcher import FontMatch
from pipeline import _local_path_for


def word_png(text="welcome baby"):
    image = Image.new("RGB", (520, 100), "white")
    ImageDraw.Draw(image).text((15, 30), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def blank_png():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 50), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_validated_image_rejects_blank_and_accepts_ink():
    assert font_preview._validated_image(b"") is None
    assert font_preview._validated_image(blank_png()) is None
    assert font_preview._validated_image(word_png()) is not None


def test_specimen_preview_fetches_first_usable_specimen(tmp_path, monkeypatch):
    monkeypatch.setattr(font_preview, "PREVIEW_DIR", tmp_path)
    calls = []

    def fake_get(url, hosts, limit, timeout, accept, deadline):
        calls.append(url)
        return word_png()

    monkeypatch.setattr(font_preview, "_get", fake_get)
    url = "https://d1ly52g9wjvbd2.cloudfront.net/img16/A/L/ExampleA.png"
    image = font_preview.specimen_preview((url,))
    assert image is not None
    assert calls == [url]
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_specimen_preview_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(font_preview, "PREVIEW_DIR", tmp_path)
    calls = []

    def fake_get(url, hosts, limit, timeout, accept, deadline):
        calls.append(url)
        return word_png()

    monkeypatch.setattr(font_preview, "_get", fake_get)
    url = "https://d1ly52g9wjvbd2.cloudfront.net/img16/A/L/ExampleA.png"
    assert font_preview.specimen_preview((url,)) is not None
    assert font_preview.specimen_preview((url,)) is not None
    assert len(calls) == 1


def test_specimen_preview_skips_blank_then_uses_next(tmp_path, monkeypatch):
    monkeypatch.setattr(font_preview, "PREVIEW_DIR", tmp_path)
    seen = []

    def fake_get(url, hosts, limit, timeout, accept, deadline):
        seen.append(url)
        return blank_png() if url.endswith("blank.png") else word_png()

    monkeypatch.setattr(font_preview, "_get", fake_get)
    urls = ("https://d1ly52g9wjvbd2.cloudfront.net/blank.png",
            "https://d1ly52g9wjvbd2.cloudfront.net/ok.png")
    assert font_preview.specimen_preview(urls) is not None
    assert seen == list(urls)


def test_specimen_preview_rejects_untrusted_hosts(monkeypatch):
    def fake_get(*args, **kwargs):
        raise AssertionError("nothing may be fetched from a non-provider host")

    monkeypatch.setattr(font_preview, "_get", fake_get)
    assert font_preview.specimen_preview(("http://d1ly52g9wjvbd2.cloudfront.net/a.png",)) is None
    assert font_preview.specimen_preview(("https://127.0.0.1/a.png",)) is None
    assert font_preview.specimen_preview(()) is None


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


def test_card_never_names_the_upstream_service():
    match = FontMatch("ext", "Nexa Script", "script", "", 0.9, 0, 0, 0, 0,
                      source="external:WhatFontIs")
    card = render_card(Image.new("RGB", (250, 70), "white"), "welcome baby", [match])
    assert card.startswith(b"\x89PNG")


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
    assert font_preview._get("https://d1ly52g9wjvbd2.cloudfront.net/test.png",
                             font_preview.PREVIEW_HOSTS, 1000, (1, 1), "image/*",
                             time.monotonic() + 5) is None
    assert len(calls) == 1
