"""
identify_cli.py — تجربة المحرك من سطر الأوامر بدون تليجرام
==========================================================
الاستخدام:
    python identify_cli.py path/to/image.png
    python identify_cli.py image.jpg --text "Question" --out card.png
    python identify_cli.py image.jpg --top 5 --candidates 30 --limit 940

يطبع النص المقروء وأقرب الخطوط، ويحفظ بطاقة المقارنة PNG (اسم + معاينة، بدون روابط).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import identify
from services import Services


def main() -> int:
    ap = argparse.ArgumentParser(description="Englishawy v4 — تحديد الخطوط من Google ومكتبة المالك")
    ap.add_argument("image", help="مسار الصورة")
    ap.add_argument("--text", default=None, help="تجاوز الـOCR بنص يدوي")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="حدّ عدد الخطوط المفحوصة")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--binarize", action="store_true")
    ap.add_argument("--out", default="comparison_card.png")
    ap.add_argument("--source", choices=["all", "google", "owner"], default="all")
    args = ap.parse_args()

    p = Path(args.image)
    if not p.exists():
        print(f"الصورة غير موجودة: {p}", file=sys.stderr)
        return 1

    def prog(i, n):
        if i % 400 == 0:
            print(f"  مطابقة الأشكال {i}/{n}", flush=True)

    services = Services.create()
    fonts = services.catalog.fonts(args.source, google_limit=args.limit)
    if not fonts:
        print("المكتبة فارغة؛ شغّل setup_assets.py أو أضف خطوط المالك", file=sys.stderr)
        return 1

    r = identify(
        p.read_bytes(),
        top_n=args.top,
        candidates=args.candidates,
        force_invert=args.invert,
        force_binarize=args.binarize,
        text_override=args.text,
        fonts=fonts,
        progress=prog,
    )
    print(f"\nالنص المقروء : {r.text!r}  (ثقة {r.ocr.confidence:.0f}%)")
    print(f"المعالجة     : {r.prep.summary}")
    print(f"المكتبة      : {r.font_count} خط | DINOv2: {'نعم' if r.neural_used else 'لا'}")
    print(f"الزمن        : {r.elapsed_s:.1f}s  {r.timings}")
    print("\nأقرب الخطوط:")
    for i, m in enumerate(r.matches, 1):
        print(f"  {i}. {m.family:28s} final={getattr(m,'final_score',m.score):.3f} "
              f"shape={m.score:.3f} neural={(getattr(m,'neural_sim',None) or 0):.3f} source={m.source}")
    Path(args.out).write_bytes(r.card_png)
    print(f"\nبطاقة المقارنة محفوظة في: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
