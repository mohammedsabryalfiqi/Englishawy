"""
setup_assets.py — تنزيل أصول محرك Englishawy المحلي (بدون أي روابط خطوط للمستخدم)
=================================================================================
ينزّل مرة واحدة ويخزّن مؤقتاً داخل مجلد الكاش (افتراضياً .cache/englishawy):
  1. بيانات Tesseract العصبية (eng.traineddata).
  2. نموذج الرؤية DINOv2 (ViT-S/14) بصيغة ONNX لإعادة الترتيب العصبي.
  3. مكتبة خطوط Google (TTF) من سجل @expo-google-fonts على npm (بدون تسجيل دخول).

كل شيء يُخزَّن محلياً؛ البوت لا يرسل للمستخدم أي رابط تحميل خطوط، فقط الاسم + صورة معاينة.

الاستخدام:
    python setup_assets.py                 # تنزيل كل شيء (متعدد الخيوط، قابل للاستئناف)
    python setup_assets.py --workers 24    # عدد خيوط تنزيل الخطوط
    python setup_assets.py --limit 940     # تنزيل أول 940 خط فقط
    python setup_assets.py --fonts-only    # الخطوط فقط
    python setup_assets.py --models-only   # tessdata + DINOv2 فقط
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import io
import json
import os
import tarfile
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("ENGLISHAWY_CACHE", ROOT / ".cache" / "englishawy"))
FONTS_DIR = CACHE_DIR / "fonts"
MODELS_DIR = CACHE_DIR / "models"
TESSDATA_DIR = CACHE_DIR / "tessdata"
MANIFEST_PATH = ROOT / "font_manifest.json"

# مصادر الأصول (كلها وصول عام بدون مفتاح). يمكن تجاوز مسار النموذج بمتغير بيئة.
TESSDATA_URLS = [
    "https://api.github.com/repos/tesseract-ocr/tessdata_fast/contents/eng.traineddata",
    "https://api.github.com/repos/tesseract-ocr/tessdata/contents/eng.traineddata",
]
DINOV2_URLS = [
    "https://api.github.com/repos/QingTian-222/Sts2Mod_DrawAndGuessMod/contents/Models/dinov2_vits14.onnx",
]
NPM_REGISTRY = "https://registry.npmjs.org/@expo-google-fonts%2F{slug}"

_print_lock = threading.Lock()
_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        s.headers.update({"User-Agent": "englishawy-setup/3.0"})
        if token:
            s.headers["Authorization"] = f"Bearer {token}"
        _local.s = s
    return _local.s


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _download(url: str, dest: Path, raw: bool = False, min_bytes: int = 1000) -> bool:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Accept": "application/vnd.github.raw"} if raw else {}
    try:
        r = _session().get(url, headers=headers, timeout=180)
        if r.status_code == 200 and len(r.content) >= min_bytes:
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return True
        _log(f"  ! {url} -> HTTP {r.status_code} ({len(r.content)} bytes)")
    except Exception as exc:  # noqa: BLE001
        _log(f"  ! {url} -> {exc}")
    return False


def setup_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)

    tess = TESSDATA_DIR / "eng.traineddata"
    if not tess.exists():
        _log("[1/2] تنزيل بيانات Tesseract (eng.traineddata)...")
        for url in TESSDATA_URLS:
            if _download(url, tess, raw=True, min_bytes=100_000):
                break
    _log(f"  tessdata: {'OK' if tess.exists() else 'MISSING'}")

    dino = Path(os.getenv("DINOV2_ONNX_PATH", MODELS_DIR / "dinov2_vits14.onnx"))
    if not dino.exists():
        _log("[2/2] تنزيل نموذج DINOv2 (ONNX)...")
        for url in DINOV2_URLS:
            if _download(url, dino, raw=True, min_bytes=1_000_000):
                break
    _log(f"  dinov2:   {'OK' if dino.exists() else 'MISSING'}")


def _fetch_font(entry: dict) -> tuple[str, bool, str]:
    slug = entry["slug"]
    dest = FONTS_DIR / f"{slug}.ttf"
    if dest.exists() and dest.stat().st_size > 1000:
        return slug, True, "cached"
    npm = entry.get("npm") or slug
    # 1) npm (بدون مصادقة، بدون حد معدل صارم)
    try:
        r = _session().get(
            NPM_REGISTRY.format(slug=npm),
            headers={"Accept": "application/vnd.npm.install-v1+json"},
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            ver = data["dist-tags"]["latest"]
            tarball = data["versions"][ver]["dist"]["tarball"]
            tb = _session().get(tarball, timeout=90).content
            with tarfile.open(fileobj=io.BytesIO(tb), mode="r:gz") as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".ttf")]
                chosen = (
                    [m for m in members if "/400Regular/" in m.name]
                    or [m for m in members if "Regular" in m.name and "Italic" not in m.name]
                    or members
                )
                if chosen:
                    raw = tf.extractfile(chosen[0]).read()
                    if len(raw) > 1000:
                        dest.write_bytes(raw)
                        return slug, True, "npm"
    except Exception as exc:  # noqa: BLE001
        npm_err = str(exc)
    else:
        npm_err = "no ttf in package"
    # 2) احتياطي: Google Fonts blob عبر GitHub API (يستخدم GH_TOKEN إن وُجد)
    sha = entry.get("sha")
    if sha:
        try:
            r = _session().get(
                f"https://api.github.com/repos/google/fonts/git/blobs/{sha}",
                headers={"Accept": "application/vnd.github.raw"},
                timeout=60,
            )
            if r.status_code == 200 and len(r.content) > 1000:
                dest.write_bytes(r.content)
                return slug, True, "github"
        except Exception:  # noqa: BLE001
            pass
    return slug, False, npm_err


def setup_fonts(workers: int = 16, limit: int | None = None) -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    # أولوية للخطوط الكتابية والمذيلة حتى يكون العرض التجريبي غنياً
    order = {"handwriting": 0, "serif": 1, "display": 2, "sans-serif": 3, "monospace": 4}
    manifest.sort(key=lambda e: (order.get(e.get("category"), 9), e["slug"]))
    if limit:
        manifest = manifest[:limit]
    total = len(manifest)
    _log(f"تنزيل {total} خط Google إلى {FONTS_DIR} (workers={workers})...")
    ok = 0
    failed: list[str] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_font, e): e for e in manifest}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            slug, success, src = fut.result()
            if success:
                ok += 1
            else:
                failed.append(slug)
            if i % 100 == 0 or i == total:
                _log(f"  {i}/{total} | نجح {ok} | فشل {len(failed)} | {time.time()-t0:.0f}s")
    index = []
    for e in manifest:
        p = FONTS_DIR / f"{e['slug']}.ttf"
        if p.exists() and p.stat().st_size > 1000:
            index.append({**e, "file": str(p.relative_to(CACHE_DIR))})
    (FONTS_DIR / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    _log(f"اكتمل: {ok}/{total} خط محفوظ. الفهرس: {FONTS_DIR/'index.json'}")
    if failed:
        _log(f"  فشل ({len(failed)}): {', '.join(failed[:20])}{'...' if len(failed)>20 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description="تنزيل أصول Englishawy المحلية")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fonts-only", action="store_true")
    ap.add_argument("--models-only", action="store_true")
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not args.fonts_only:
        setup_models()
    if not args.models_only:
        setup_fonts(workers=args.workers, limit=args.limit)
    _log("انتهى الإعداد.")


if __name__ == "__main__":
    main()
