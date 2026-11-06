import io
import stat
import zipfile
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from font_catalog import FontCatalog
from font_matcher import match_fonts, render_text
from font_store import FontStore


def make_zip(path: Path, entries: list[tuple[str, bytes, int]]):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            archive.writestr(info, data)


def test_install_dedupe_persist_and_delete(tmp_path, sample_font):
    store = FontStore(tmp_path / "data")
    first = store.install_font(sample_font, "collection.zip")
    assert first["added"] == 1
    assert store.install_font(sample_font)["duplicates"] == 1
    entry = store.entries()[0]
    assert entry["family"] == "Test Owner Sans"
    assert Path(entry["path"]).is_file()
    reopened = FontStore(tmp_path / "data")
    assert reopened.stats()["faces"] == 1
    assert reopened.delete(entry["id"])
    assert reopened.stats()["faces"] == 0
    assert not Path(entry["path"]).exists()


def test_zip_never_extracts_traversal_or_symlinks(tmp_path, sample_font):
    archive_path = tmp_path / "safe.zip"
    raw = sample_font.read_bytes()
    make_zip(archive_path, [("../escape.ttf", raw, 0o100644),
                            ("link.ttf", raw, stat.S_IFLNK | 0o777),
                            ("nested/fonts/good.ttf", raw, 0o100644),
                            ("readme.txt", b"ignore", 0o100644)])
    store = FontStore(tmp_path / "data")
    report = store.import_zip(archive_path)
    assert report["added"] == 1
    assert not (tmp_path / "escape.ttf").exists()
    assert not (store.root / "link.ttf").exists()
    assert report["ignored"] >= 2


def test_nested_zip_and_corrupt_font_do_not_abort_collection(tmp_path, sample_font):
    inner = io.BytesIO()
    make_zip(inner, [("inner.ttf", sample_font.read_bytes(), 0o100644)])
    archive_path = tmp_path / "outer.zip"
    make_zip(archive_path, [("inner.zip", inner.getvalue(), 0o100644),
                            ("bad.ttf", b"not a font", 0o100644),
                            ("good.ttf", sample_font.read_bytes(), 0o100644)])
    report = FontStore(tmp_path / "data").import_zip(archive_path)
    assert report["added"] == 1
    assert report["duplicates"] == 1
    assert report["nested_archives"] == 1
    assert len(report["invalid"]) == 1


def test_no_200mb_application_limit(tmp_path, sample_font):
    archive_path = tmp_path / "large-collection.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(sample_font, "font.ttf")
        # >200 MiB uncompressed, compressible filler; streamed, never read into RAM.
        with archive.open("ignored-assets.bin", "w", force_zip64=True) as stream:
            chunk = b"\0" * (1024 * 1024)
            for _ in range(210):
                stream.write(chunk)
    report = FontStore(tmp_path / "data").import_zip(archive_path)
    assert report["added"] == 1
    assert report["ignored"] == 1
    assert not report["invalid"]


def test_woff_is_normalized_for_pillow(tmp_path, sample_font):
    webfont = tmp_path / "test.woff"
    with TTFont(sample_font) as font:
        font.flavor = "woff"
        font.save(webfont)
    store = FontStore(tmp_path / "data")
    assert store.install_font(webfont)["added"] == 1
    entry = store.entries()[0]
    assert Path(entry["path"]).suffix == ".ttf"
    assert render_text(entry["path"], "Question") is not None


def test_owner_fonts_are_searchable_and_not_dropped_by_google_limit(tmp_path, sample_font, monkeypatch):
    monkeypatch.setenv("FONT_LIMIT", "1")
    store = FontStore(tmp_path / "data")
    store.install_font(sample_font)
    catalog = FontCatalog(store)
    fonts = catalog.fonts("owner")
    assert len(fonts) == 1
    image = render_text(fonts[0]["path"], "Question")
    matches, count = match_fonts(image, "Question", fonts=fonts, top_k=30)
    assert count == 1
    assert matches[0].source == "owner"
    assert matches[0].family == "Test Owner Sans"
