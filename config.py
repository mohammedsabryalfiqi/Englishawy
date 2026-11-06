"""Shared, secret-safe configuration (v5 patched).

New in v5:
  CASCADE_THRESHOLD  — min external confidence (0..1) before owner fallback.
                       Default 0.78. Tune via Space Variables (no code change).
  WHATFONTIS_API_KEY — free key from https://www.whatfontis.com/API-identify-fonts-from-image.html
  MYFONTS_API_KEY + WHATTHEFONT_API_URL — only for MyFonts enterprise customers.
                       Without them WhatTheFont is offered as a 1-click manual
                       deep-link (first option) and the cascade continues.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("ENGLISHAWY_CACHE", ROOT / ".cache" / "englishawy")).resolve()
DEFAULT_OWNER_TELEGRAM_ID = 5231453921
FONT_SUFFIXES = {".ttf", ".otf", ".ttc", ".woff", ".woff2"}
UPLOAD_SUFFIXES = FONT_SUFFIXES | {".zip"}


def owner_telegram_id() -> int:
    """A malformed override must fail closed, never silently authorize someone else."""
    value = os.getenv("OWNER_TELEGRAM_ID", str(DEFAULT_OWNER_TELEGRAM_ID)).strip()
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError("OWNER_TELEGRAM_ID must be a positive integer") from exc
    if result <= 0:
        raise ValueError("OWNER_TELEGRAM_ID must be a positive integer")
    return result


def is_owner(user_id: int | None, chat_id: int | None = None, chat_type: str | None = None) -> bool:
    """Check Telegram's authenticated sender, not an ID supplied in a message/browser."""
    return (
        isinstance(user_id, int)
        and user_id == owner_telegram_id()
        and (chat_id is None or chat_id == user_id)
        and (chat_type is None or chat_type == "private")
    )


def cascade_threshold() -> float:
    """External-match acceptance threshold (default 0.78)."""
    try:
        v = float(os.getenv("CASCADE_THRESHOLD", "0.78"))
    except ValueError:
        return 0.78
    return max(0.0, min(0.99, v))


def whatfontis_api_key() -> str:
    return os.getenv("WHATFONTIS_API_KEY", "").strip()


def myfonts_api_key() -> str:
    return os.getenv("MYFONTS_API_KEY", "").strip()


def whatthefont_api_url() -> str:
    return os.getenv("WHATTHEFONT_API_URL", "").strip()


def data_dir() -> Path:
    """Use an attached volume when configured; otherwise explicitly temporary storage."""
    configured = os.getenv("ENGLISHAWY_DATA", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        return Path("/data/englishawy")
    return CACHE_DIR / "data"


def storage_status() -> dict:
    path = data_dir()
    # A writable directory is NOT proof of persistence. Only the operator can confirm
    # that this path is backed by an attached HF Volume / Storage Bucket.
    confirmed = os.getenv("ENGLISHAWY_PERSISTENT_STORAGE", "0").lower() in {"1", "true", "yes"}
    return {
        "path": str(path),
        "persistent_configured": confirmed,
        "label": "مسار تخزين دائم مُعدّ (تحقق من ربط Volume)" if confirmed else "تخزين مؤقت: قد تضيع الخطوط عند إعادة التشغيل",
    }


def public_base_url() -> str:
    explicit = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if explicit:
        if not explicit.startswith("https://"):
            raise ValueError("PUBLIC_BASE_URL must use HTTPS")
        return explicit
    host = os.getenv("SPACE_HOST", "").strip()
    if host:
        if "/" in host or not host.endswith(".hf.space"):
            raise ValueError("SPACE_HOST must be a *.hf.space hostname")
        return f"https://{host}"
    return ""


def positive_env(name: str, default: int, maximum: int = 100000) -> int:
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value
