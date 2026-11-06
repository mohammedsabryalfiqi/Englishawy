"""One searchable catalog: cached Google fonts + freshly uploaded owner faces."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from config import CACHE_DIR, ROOT
from font_store import FontStore

FONTS_DIR = CACHE_DIR / "fonts"


class FontCatalog:
    def __init__(self, store: FontStore):
        self.store = store
        self._lock = threading.RLock()
        self._google: list[dict] | None = None
        self._google_stamp = None

    def _google_entries(self) -> list[dict]:
        index = FONTS_DIR / "index.json"
        stamp = (index.stat().st_mtime_ns, index.stat().st_size) if index.exists() else None
        with self._lock:
            if self._google is not None and stamp == self._google_stamp:
                return list(self._google)
            entries = []
            if index.exists():
                try:
                    raw = json.loads(index.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raw = []
                for entry in raw:
                    path = (CACHE_DIR / entry["file"]).resolve()
                    if path.is_relative_to(FONTS_DIR.resolve()) and path.is_file():
                        entries.append({**entry, "path": str(path), "source": "google", "face_index": 0, "style": "Regular"})
            self._google, self._google_stamp = entries, stamp
            return list(entries)

    def fonts(self, source: str = "all", google_limit: int | None = None) -> list[dict]:
        if source not in {"all", "google", "owner"}:
            raise ValueError("مصدر الخطوط يجب أن يكون all أو google أو owner")
        google = self._google_entries() if source in {"all", "google"} else []
        limit = google_limit if google_limit is not None else int(os.getenv("FONT_LIMIT", "0"))
        if limit > 0:
            google = google[:limit]
        # Owner fonts are never silently dropped by the Google acceleration limit.
        owner = self.store.all_entries() if source in {"all", "owner"} else []
        return owner + google

    def stats(self) -> dict:
        return {"google": len(self._google_entries()), "owner": self.store.stats()}

    def invalidate(self) -> None:
        with self._lock:
            self._google = None
            self._google_stamp = None
