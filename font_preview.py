"""Render the user's actual word for external matches, without downloading fonts.

WhatFontIs' public font page exposes a custom-text image endpoint in the #ct
section. Discover that endpoint from the page (never guess a font ID), request
the OCR text, validate the image, and cache a re-encoded PNG outside Git.
Only public WhatFontIs HTTPS hosts are allowed; API credentials are never sent
to the page or preview server. A specimen alphabet is NOT passed off as the
user's word when custom rendering is unavailable.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
import time
import urllib.parse
import uuid
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

from config import CACHE_DIR

logger = logging.getLogger(__name__)
PREVIEW_DIR = CACHE_DIR / "previews"
PAGE_HOSTS = {"whatfontis.com", "www.whatfontis.com"}
PREVIEW_HOSTS = PAGE_HOSTS | {"img.whatfontis.com", "imgall.whatfontis.com"}
MAX_PAGE_BYTES = 1_500_000
MAX_IMAGE_BYTES = 3_000_000
MAX_CACHE_FILES = 256
CACHE_TTL = 7 * 86400
_CACHE_LOCK = threading.Lock()
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EnglishawyBot/5.1)", "Accept-Language": "en"}


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
    # Fragments are client-side only; drop them so canonical #ct links work.
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
        logger.info("WhatFontIs preview request failed: %s", type(exc).__name__)
    return None


class _PreviewImages(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls: list[tuple[bool, str]] = []
        self._in_custom = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("name") == "ct" or values.get("id") in {"ct", "sample"}:
            self._in_custom = True
        if tag != "img":
            return
        for key in ("data-src", "data-lazy-src", "src"):
            value = values.get(key)
            if value:
                self.urls.append((self._in_custom,
                                  urllib.parse.urljoin("https://www.whatfontis.com/", value)))


def custom_preview_endpoint(page_html: str, text: str) -> str | None:
    """Find the public #ct renderer and substitute the requested text safely."""
    parser = _PreviewImages()
    try:
        parser.feed(page_html[:MAX_PAGE_BYTES])
    except Exception:
        return None
    # Prefer the #ct/custom-text section over unrelated specimen/related-font images.
    for prefer_custom in (True, False):
        for in_custom, url in parser.urls:
            if prefer_custom and not in_custom:
                continue
            try:
                parts = urllib.parse.urlsplit(url)
                query = urllib.parse.parse_qs(parts.query)
            except ValueError:
                continue
            if (parts.scheme not in {"https", ""} or parts.hostname not in PREVIEW_HOSTS
                    or parts.path != "/index.php" or query.get("p") != ["refresh"]
                    or not re.fullmatch(r"\d{1,12}", query.get("id", [""])[0])):
                continue
            query["text"] = [text]
            return urllib.parse.urlunsplit(("https", parts.hostname, "/index.php",
                                            urllib.parse.urlencode(query, doseq=True), ""))
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


def _cache_path(page_url: str, text: str) -> Path:
    key = hashlib.sha256(f"{page_url}\x00{text}".encode()).hexdigest()
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


def whatfontis_preview(page_url: str, text: str, budget: float = 12.0) -> Image.Image | None:
    """Return a real image of TEXT, or None. Never substitute another font."""
    deadline = time.monotonic() + max(0.1, budget)
    text = " ".join(text.split())[:160]
    page_url = _safe_url(page_url, PAGE_HOSTS) if page_url else None
    if not text or page_url is None:
        return None
    cache = _cache_path(page_url, text)
    cached = _read_cache(cache)
    if cached is not None:
        return cached
    parts = urllib.parse.urlsplit(page_url)
    page_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path,
                                       urllib.parse.urlencode({"text": text}), ""))
    raw_page = _get(page_url, PAGE_HOSTS, MAX_PAGE_BYTES, (4, 6), "text/html", deadline)
    if not raw_page:
        return None
    endpoint = custom_preview_endpoint(raw_page.decode("utf-8", errors="replace"), text)
    if endpoint is None:
        logger.info("WhatFontIs custom preview endpoint unavailable for this font")
        return None
    image = _validated_image(_get(endpoint, PREVIEW_HOSTS, MAX_IMAGE_BYTES, (4, 8), "image/*", deadline) or b"")
    if image is not None:
        _write_cache(cache, image)
    return image
