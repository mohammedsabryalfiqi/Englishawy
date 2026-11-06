"""Fetch a real preview image for an externally matched font.

The identification API returns three specimen strips per match — uppercase,
lowercase and extra glyphs — drawn in the matched family. Those are the only
images still served for a font: the on-site custom-text renderer the old
implementation scraped (#ct / index.php?p=refresh) was removed upstream, so every
external row fell through to "Preview unavailable".

We download one specimen, validate it (real ink, sane size), re-encode it as a
local PNG and cache it outside Git. Only the provider's own HTTPS image hosts are
reachable; API credentials never leave the API call. The card labels a specimen
as a sample instead of passing it off as the user's own word.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

from config import CACHE_DIR

logger = logging.getLogger(__name__)
PREVIEW_DIR = CACHE_DIR / "previews"
PREVIEW_HOSTS = {"d1ly52g9wjvbd2.cloudfront.net", "img.whatfontis.com", "imgall.whatfontis.com",
                 "www.whatfontis.com", "whatfontis.com"}
MAX_IMAGE_BYTES = 3_000_000
MAX_CACHE_FILES = 256
CACHE_TTL = 7 * 86400
_CACHE_LOCK = threading.Lock()
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EnglishawyBot/5.2)", "Accept-Language": "en"}


def _safe_url(url: str, hosts: set[str]) -> str | None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    if (parsed.scheme != "https" or parsed.hostname not in hosts
            or parsed.username is not None or parsed.password is not None
            or port not in {None, 443}):
        return None
    # Fragments are client-side only; drop them so cached keys stay canonical.
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _get(url: str, hosts: set[str], limit: int, timeout: tuple[float, float], accept: str,
         deadline: float) -> bytes | None:
    """Bounded GET; redirects must stay on the same provider's HTTPS allowlist."""
    current = _safe_url(url, hosts)
    try:
        for _ in range(4):
            remaining = deadline - time.monotonic()
            if current is None or remaining <= 0:
                return None
            with requests.get(current, headers={**_HEADERS, "Accept": accept},
                              timeout=(min(timeout[0], remaining), min(timeout[1], remaining)),
                              stream=True, allow_redirects=False) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    current = _safe_url(urllib.parse.urljoin(current, response.headers.get("Location", "")), hosts)
                    continue
                if response.status_code != 200:
                    return None
                data = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    data.extend(chunk)
                    if len(data) > limit or time.monotonic() >= deadline:
                        return None
                return bytes(data)
    except requests.RequestException as exc:
        logger.info("Specimen preview request failed: %s", type(exc).__name__)
    return None


def _validated_image(raw: bytes) -> Image.Image | None:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        return None
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            if probe.width * probe.height > 4_000_000 or min(probe.size) < 8:
                return None
            probe.load()
            image = ImageOps.exif_transpose(probe).convert("RGBA")
        # Reject the lazy-loading spacer, blank responses, and solid error tiles.
        alpha = np.asarray(image.getchannel("A"))
        rgb = np.asarray(image.convert("RGB"))
        ink = (alpha > 16) & (np.min(rgb, axis=2) < 235)
        coverage = float(ink.mean())
        if not 0.0005 < coverage < 0.95:
            return None
        ys, xs = np.where(ink)
        if len(xs) == 0 or len(np.unique(xs)) < 4:
            return None
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        return background.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()
    return PREVIEW_DIR / f"{key}.png"


def _read_cache(path: Path) -> Image.Image | None:
    try:
        if not path.is_file() or time.time() - path.stat().st_mtime > CACHE_TTL:
            return None
        return _validated_image(path.read_bytes())
    except OSError:
        return None


def _write_cache(path: Path, image: Image.Image) -> None:
    temporary = path.with_name(f".{uuid.uuid4().hex}.part")
    try:
        with _CACHE_LOCK:
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            image.save(temporary, format="PNG")
            os.replace(temporary, path)
            files = sorted(PREVIEW_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[MAX_CACHE_FILES:]:
                old.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass  # A cache failure must never prevent a preview from being shown.
    finally:
        temporary.unlink(missing_ok=True)


def specimen_preview(urls, budget: float = 8.0) -> Image.Image | None:
    """Return the first usable specimen image, or None.

    ``urls`` are the provider's own image links from the identification response,
    already validated by the caller. Nothing is fetched from anywhere else.
    """
    deadline = time.monotonic() + max(0.1, budget)
    for url in urls or ():
        target = _safe_url(str(url), PREVIEW_HOSTS)
        if target is None:
            continue
        cache = _cache_path(target)
        cached = _read_cache(cache)
        if cached is not None:
            return cached
        raw = _get(target, PREVIEW_HOSTS, MAX_IMAGE_BYTES, (4, 6), "image/*", deadline) or b""
        image = _validated_image(raw)
        if image is not None:
            _write_cache(cache, image)
            return image
        if time.monotonic() >= deadline:
            break
    return None
