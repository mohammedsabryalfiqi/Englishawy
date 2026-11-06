import sys
from pathlib import Path

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def sample_font(tmp_path_factory) -> Path:
    """Tiny original test font: real glyf/cmap/name/OS/2 tables, no downloaded fixture."""
    path = tmp_path_factory.mktemp("fonts") / "TestOwnerSans.ttf"
    order = [".notdef", "space"] + list("QuestionAB")
    cmap = {ord(c): c for c in "QuestionAB"} | {32: "space"}
    glyphs = {}
    for index, name in enumerate(order):
        pen = TTGlyphPen(None)
        if name not in {".notdef", "space"}:
            width = 380 + index * 17
            pen.moveTo((40, 0)); pen.lineTo((width, 0)); pen.lineTo((width, 620)); pen.lineTo((40, 620)); pen.closePath()
            pen.moveTo((120, 160)); pen.lineTo((120, 460)); pen.lineTo((width - 80, 460)); pen.lineTo((width - 80, 160)); pen.closePath()
        glyphs[name] = pen.glyph()
    builder = FontBuilder(1000, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap(cmap)
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (600, 40) for name in order})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "Test Owner Sans", "styleName": "Regular"})
    builder.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWeightClass=400)
    builder.setupPost()
    builder.save(path)
    return path
