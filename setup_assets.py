"""Download resumible public assets outside Git; validate model checksums.

Usage: python setup_assets.py [--workers 16] [--limit 940] [--fonts-only|--models-only]
Google TTFs: npm @expo-google-fonts, with the Google Fonts GitHub blob fallback.
ZeroGPU uses the official HF facebook/dinov2-small model instead of the ONNX mirror.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

import requests

from config import CACHE_DIR, ROOT

FONTS_DIR = CACHE_DIR / "fonts"
MODELS_DIR = CACHE_DIR / "models"
TESSDATA_DIR = CACHE_DIR / "tessdata"
MANIFEST_PATH = ROOT / "font_manifest.json"
TESSDATA_URL = "https://api.github.com/repos/tesseract-ocr/tessdata_fast/contents/eng.traineddata"
TESSDATA_SHA256 = "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
# Public CPU fallback mirror, inspected input image[1,3,224,224] -> output[1,384].
# This is not represented as an official Facebook ONNX release.
DINOV2_URL = "https://api.github.com/repos/QingTian-222/Sts2Mod_DrawAndGuessMod/contents/Models/dinov2_vits14.onnx"
DINOV2_SHA256 = "9185fa6b149e181a21c8ca2cace43660856ebd1129d0aaa423c318667564ab87"
NPM_REGISTRY = "https://registry.npmjs.org/@expo-google-fonts%2F{slug}"
_print_lock = threading.Lock()
_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": "englishawy-space/4.0"})
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        if token:
            session.headers["Authorization"] = f"Bearer {token}"
        _local.session = session
    return _local.session


def _log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str, min_bytes: int) -> bool:
    if destination.is_file() and destination.stat().st_size >= min_bytes and _hash_file(destination) == expected_sha256:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        with _session().get(url, headers={"Accept": "application/vnd.github.raw"}, timeout=(15, 180), stream=True) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            size = 0
            with part.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if size < min_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError("asset checksum/size mismatch")
        os.replace(part, destination)
        return True
    except Exception as exc:
        _log(f"Asset download failed ({type(exc).__name__}); existing valid cache is preserved.")
        return False
    finally:
        part.unlink(missing_ok=True)


def setup_models(include_onnx: bool = True) -> dict:
    tess = TESSDATA_DIR / "eng.traineddata"
    tess_ok = _download(TESSDATA_URL, tess, TESSDATA_SHA256, 100_000)
    onnx_ok = True
    if include_onnx:
        onnx = Path(os.getenv("DINOV2_ONNX_PATH", MODELS_DIR / "dinov2_vits14.onnx"))
        # A deliberately supplied custom model is not overwritten with the mirror.
        if os.getenv("DINOV2_ONNX_PATH") and onnx.is_file():
            onnx_ok = True
        else:
            onnx_ok = _download(DINOV2_URL, onnx, DINOV2_SHA256, 1_000_000)
    return {"tesseract": tess_ok, "onnx": onnx_ok}


def _atomic_font(raw: bytes, destination: Path) -> None:
    if len(raw) < 1000 or raw[:4] not in {b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf"}:
        raise ValueError("invalid sfnt payload")
    part = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        part.write_bytes(raw)
        os.replace(part, destination)
    finally:
        part.unlink(missing_ok=True)


def _fetch_font(entry: dict) -> tuple[str, bool, str]:
    slug = entry["slug"]
    destination = FONTS_DIR / f"{slug}.ttf"
    if destination.is_file() and destination.stat().st_size > 1000:
        return slug, True, "cached"
    npm = entry.get("npm") or slug
    error = "no TTF in package"
    try:
        response = _session().get(NPM_REGISTRY.format(slug=npm),
                                  headers={"Accept": "application/vnd.npm.install-v1+json"}, timeout=(10, 30))
        if response.status_code == 200:
            data = response.json()
            version = data["dist-tags"]["latest"]
            tarball = data["versions"][version]["dist"]["tarball"]
            package = _session().get(tarball, timeout=(10, 90))
            package.raise_for_status()
            with tarfile.open(fileobj=io.BytesIO(package.content), mode="r:gz") as archive:
                members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".ttf")]
                chosen = ([m for m in members if "/400Regular/" in m.name]
                          or [m for m in members if "Regular" in m.name and "Italic" not in m.name]
                          or members)
                if chosen:
                    stream = archive.extractfile(chosen[0])
                    if stream:
                        _atomic_font(stream.read(), destination)
                        return slug, True, "npm"
    except Exception as exc:
        error = type(exc).__name__
    sha = entry.get("sha")
    if sha:
        try:
            response = _session().get(f"https://api.github.com/repos/google/fonts/git/blobs/{sha}",
                                      headers={"Accept": "application/vnd.github.raw"}, timeout=(10, 60))
            if response.status_code == 200:
                _atomic_font(response.content, destination)
                return slug, True, "github"
        except Exception:
            pass
    return slug, False, error


def _write_index(entries: list[dict]) -> None:
    index = [{**entry, "file": str((FONTS_DIR / f"{entry['slug']}.ttf").relative_to(CACHE_DIR))}
             for entry in entries if (FONTS_DIR / f"{entry['slug']}.ttf").is_file()]
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=FONTS_DIR, suffix=".tmp", delete=False) as stream:
        path = Path(stream.name)
        json.dump(index, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(path, FONTS_DIR / "index.json")
    finally:
        path.unlink(missing_ok=True)


def setup_fonts(workers: int = 16, limit: int | None = None, progress=None) -> dict:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    order = {"handwriting": 0, "serif": 1, "display": 2, "sans-serif": 3, "monospace": 4}
    manifest.sort(key=lambda entry: (order.get(entry.get("category"), 9), entry["slug"]))
    selected = manifest[:limit] if limit else manifest
    ok, failed = 0, []
    start = time.time()
    _write_index(manifest)  # Expose already cached fonts immediately.
    with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, 32))) as executor:
        futures = [executor.submit(_fetch_font, entry) for entry in selected]
        for index, future in enumerate(cf.as_completed(futures), 1):
            slug, success, source = future.result()
            if success:
                ok += 1
            else:
                failed.append(slug)
            if index % 100 == 0 or index == len(selected):
                _write_index(manifest)
                _log(f"Google fonts: {index}/{len(selected)}, available={ok}, failed={len(failed)}")
            if progress:
                progress(index, len(selected))
    _write_index(manifest)  # Never discard cached fonts outside a --limit subset.
    return {"requested": len(selected), "downloaded_or_cached": ok, "failed": failed,
            "available": sum((FONTS_DIR / f"{e['slug']}.ttf").is_file() for e in manifest),
            "elapsed_s": round(time.time() - start, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Englishawy assets")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fonts-only", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not args.fonts_only:
        _log(str(setup_models(include_onnx=not args.skip_onnx)))
    if not args.models_only:
        _log(str(setup_fonts(workers=args.workers, limit=args.limit)))


if __name__ == "__main__":
    main()
