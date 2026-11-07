"""وضع الصيانة — مفتاح دائم يوقف ردود البوت العادية أثناء التحديث/إعادة التشغيل.

The flag lives in ``data_dir()/maintenance.json`` so it survives the process
restart itself: start maintenance, restart the bot / deploy, and visitors keep
seeing the maintenance notice until the admin turns it off. Admins and the
owner always bypass the gate, so they can keep testing and switch it off.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from config import data_dir

logger = logging.getLogger(__name__)

DEFAULT_MESSAGE = (
    "🛠️ البوت في وضع الصيانة مؤقتاً — بنحدّث المكتبة والسيرفر.\n"
    "هيرجع يشتغل تلقائياً بعد شوية. شكراً لصبرك 🙏"
)


class Maintenance:
    """Small persistent on/off flag with an optional custom notice."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else data_dir() / "maintenance.json"
        self._lock = threading.RLock()
        self._state = self._load()

    # ---- persistence ------------------------------------------------------
    def _load(self) -> dict:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            return raw
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash never leaves a half-written flag.
            tmp = self.path.with_name(f".{self.path.name}.tmp")
            tmp.write_text(json.dumps(self._state, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.error("maintenance flag could not be saved: %s", type(exc).__name__)

    # ---- public API -------------------------------------------------------
    def is_on(self) -> bool:
        with self._lock:
            return bool(self._state.get("on"))

    def message(self) -> str:
        with self._lock:
            text = str(self._state.get("message") or "").strip()
        return text or DEFAULT_MESSAGE

    def enable(self, text: str | None = None) -> None:
        """Turn the gate on. ``None`` keeps the current notice; '' restores the default."""
        with self._lock:
            message = (self._state.get("message") or "") if text is None else text
            self._state = {"on": True, "message": str(message or "").strip(),
                           "changed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._save()
        logger.warning("maintenance mode ENABLED")

    def disable(self) -> None:
        with self._lock:
            self._state = {**self._state, "on": False,
                           "changed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._save()
        logger.warning("maintenance mode disabled")

    def snapshot(self) -> dict:
        with self._lock:
            return {"on": bool(self._state.get("on")), "message": self.message(),
                    "changed_at": self._state.get("changed_at", "")}
