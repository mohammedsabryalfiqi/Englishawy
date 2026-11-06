"""Isolated coarse-scan worker (separate OS process): JSON job on stdin, JSON result on stdout.

Imports only font_matcher (Pillow/OpenCV/NumPy) - never app.py, torch or Telegram.
"""
from __future__ import annotations

import io
import json
import sys


def main() -> None:
    job = json.load(sys.stdin)
    from PIL import Image

    import font_matcher as fm

    ref = Image.open(io.BytesIO(bytes.fromhex(job["ref_png_hex"]))).convert("L")
    ref.load()
    signature = fm._coarse_signature(ref)
    out = []
    for i, entry in enumerate(job["entries"]):
        value = fm.coarse_entry(entry, job["text"], signature, job["max_mb"])
        if value is not None:
            out.append([i, value])
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
