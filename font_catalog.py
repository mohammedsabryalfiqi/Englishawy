"""One searchable catalog: cached Google fonts + freshly uploaded owner faces.

v7 — the full library is ALWAYS searchable.

The old mirror treated "not copied to local disk yet" as "not in the library":
`fonts()` silently dropped every owner face whose mirror copy had not finished,
so a search right after a restart ran against a partial library (e.g. 2001 of
8131 faces) and answered with wrong names. Now every owner face keeps a usable
path at all times:

* mirrored copy (fast), else
* the original file on the network volume (correct, just slower).

At startup `warm()` copies the library to local disk in the background; the bot
waits for that copy (bounded) before the first search and shows honest progress
instead of a warning. Nothing is ever dropped and a failed copy is retried on
the next warm instead of poisoning the face for the rest of the process.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import CACHE_DIR, ROOT
from font_store import FontStore

FONTS_DIR = CACHE_DIR / "fonts"
logger = logging.getLogger(__name__)

# مكتبة المالك تعيش على Volume شبكي (/data): كل open/stat هناك بطيء، فمسح آلاف الخطوط منه
# كان "يعلق" بلا CPU ولا رام. نخلق نسخة محلية سريعة (قرص الحاوية) في الخلفية بعدة خيوط
# (الانتظار I/O فيتوازى)، والمسح يقرأ من النسخة المحلية أولاً.
MIRROR_DIR = Path(os.getenv("ENGLISHAWY_LOCAL_MIRROR", str(Path(tempfile.gettempdir()) / "englishawy-mirror")))
MIRROR_WORKERS = max(1, int(os.getenv("MIRROR_WORKERS", "32")))
MIRROR_ENABLED = os.getenv("MIRROR_OWNER_FONTS", "1").lower() not in {"0", "false", "no"}
MIRROR_MIN_FREE = 512 * 1024 * 1024
MIRROR_SYNC_LIMIT = int(os.getenv("MIRROR_SYNC_LIMIT", "200"))  # أقل من هذا العدد: انسخ فوراً وانتظر
# أقصى انتظار قبل أول بحث بعد التشغيل. بعدها يستمر النسخ في الخلفية والبحث
# يستخدم مسارات الملفات الأصلية (نتيجة صحيحة، أبطأ قليلاً).
MIRROR_WAIT_S = max(0.0, float(os.getenv("MIRROR_WAIT_S", "180")))


class FontCatalog:
    def __init__(self, store: FontStore):
        self.store = store
        self._lock = threading.RLock()
        self._google: list[dict] | None = None
        self._google_stamp = None
        self._mirror_lock = threading.Lock()
        self._mirrored: dict[str, str] = {}      # content_sha -> local path
        self._mirror_fallback: dict[str, str] = {}  # content_sha -> original path (never dropped)
        self._mirror_failed: set[str] = set()
        self._mirror_busy = False
        self._mirror_total = 0
        self._mirror_started = 0.0
        self._mirror_finished = 0.0
        # A boot-time wait happens once: if the copy is still running after the
        # bounded wait, the following searches must not each pay that wait
        # again. A new large batch (e.g. a GitHub import) re-arms it.
        self._waited = False

    # ---- local mirror -----------------------------------------------------
    def _mirror_target(self, entry: dict) -> Path:
        return MIRROR_DIR / f"{entry['content_sha']}{Path(entry['path']).suffix.lower()}"

    def _copy_one(self, entry: dict) -> None:
        sha = entry["content_sha"]
        target = self._mirror_target(entry)
        try:
            if not target.exists():
                MIRROR_DIR.mkdir(parents=True, exist_ok=True)
                size = os.path.getsize(entry["path"])
                if shutil.disk_usage(MIRROR_DIR).free < size + MIRROR_MIN_FREE:
                    raise OSError("no local disk space")
                part = target.with_name(f".part-{sha}-{threading.get_ident()}")
                shutil.copyfile(entry["path"], part)
                os.replace(part, target)
            with self._mirror_lock:
                self._mirrored[sha] = str(target)
                self._mirror_fallback.pop(sha, None)
        except Exception as exc:
            # لا نُسقط الخط من البحث: نُبقيه على مساره الأصلي حتى ولو كان القرص
            # المحلي ممتلئاً. النسخة المحلية تحسين سرعة، وليست شرطاً للنتيجة.
            logger.warning("mirror failed for one font (%s) — using its original path", type(exc).__name__)
            with self._mirror_lock:
                self._mirror_failed.add(sha)
                original = str(entry.get("path") or "")
                if original:
                    self._mirror_fallback[sha] = original

    def _mirror_worker(self, todo: list[dict]) -> None:
        try:
            logger.info("mirroring %d owner font files to local disk", len(todo))
            with ThreadPoolExecutor(max_workers=MIRROR_WORKERS, thread_name_prefix="mirror") as pool:
                list(pool.map(self._copy_one, todo))
            logger.info("owner font mirror finished")
        finally:
            with self._mirror_lock:
                self._mirror_busy = False
                self._mirror_finished = time.monotonic()

    def warm(self, owner_entries: list[dict] | None = None, retry_failed: bool = False) -> None:
        """Start (single-flight) background copying of owner fonts that are not local yet."""
        if not MIRROR_ENABLED:
            return
        entries = owner_entries if owner_entries is not None else self.store.all_entries()
        with self._mirror_lock:
            seen, todo = set(), []
            for e in entries:
                sha = e.get("content_sha")
                if not sha or sha in seen:
                    continue
                seen.add(sha)
                if sha in self._mirrored:
                    continue
                if sha in self._mirror_failed and not retry_failed:
                    continue
                target = self._mirror_target(e)
                if target.exists():       # قرص محلي: stat سريع
                    self._mirrored[sha] = str(target)
                    self._mirror_fallback.pop(sha, None)
                    continue
                todo.append(e)
            self._mirror_total = max(self._mirror_total, len(seen))
            if not todo or self._mirror_busy:
                if not todo and not self._mirror_busy and self._mirror_total:
                    self._mirror_finished = time.monotonic()
                return
            self._mirror_busy = True
            self._mirror_started = time.monotonic()
            self._waited = False      # new work: the next search waits for it once
        if len(todo) <= MIRROR_SYNC_LIMIT:
            self._mirror_worker(todo)   # دفعة صغيرة (رفع خط/ZIP صغير): جاهزة قبل أن نكمل
            return
        threading.Thread(target=self._mirror_worker, args=(todo,), name="mirror-coordinator", daemon=True).start()

    def mirror_progress(self) -> tuple[int, int]:
        with self._mirror_lock:
            return len(self._mirrored), max(self._mirror_total, len(self._mirrored))

    def mirror_available(self) -> tuple[int, int]:
        """Faces searchable right now (mirrored + still on their original path)."""
        with self._mirror_lock:
            return (len(self._mirrored) + len(self._mirror_fallback),
                    max(self._mirror_total, len(self._mirrored) + len(self._mirror_fallback)))

    def wait_until_ready(self, timeout: float | None = None, progress=None, interval: float = 1.0) -> tuple[int, int]:
        """Block (call from a worker thread) until the mirror is done or timeout.

        Returns the final ``(ready, total)``. Stops early when the worker is no
        longer running, so a failed face does not force the caller to wait the
        whole timeout.
        """
        if not MIRROR_ENABLED:
            return (0, 0)
        with self._mirror_lock:
            if self._waited and not self._mirror_busy:
                ready = len(self._mirrored)
                return ready, max(self._mirror_total, ready)
        deadline = time.monotonic() + (MIRROR_WAIT_S if timeout is None else max(0.0, timeout))
        while True:
            ready, total = self.mirror_progress()
            if progress is not None:
                try:
                    progress(ready, total)
                except Exception:  # progress UI must never break the wait
                    pass
            if total == 0 or ready >= total or time.monotonic() >= deadline:
                with self._mirror_lock:
                    self._waited = True
                return ready, total
            with self._mirror_lock:
                busy = self._mirror_busy
            if not busy:
                with self._mirror_lock:
                    self._waited = True
                return ready, total
            time.sleep(interval)

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
        if MIRROR_ENABLED and owner:
            self.warm(owner)
            with self._mirror_lock:
                mirrored = dict(self._mirrored)
                fallback = dict(self._mirror_fallback)
            resolved: list[dict] = []
            for e in owner:
                sha = e.get("content_sha") or ""
                # الترتيب: نسخة محلية → الملف الأصلي → أي مسار مسجّل في الفهرس.
                path = mirrored.get(sha) or fallback.get(sha) or e.get("path")
                if path:
                    resolved.append({**e, "path": path})
                else:
                    logger.warning("owner face has no usable path; skipped")
            owner = resolved
        return owner + google

    def stats(self) -> dict:
        owner = self.store.stats()
        ready, total = self.mirror_progress()
        available, _ = self.mirror_available()
        owner["mirrored"], owner["mirror_total"] = ready, total
        owner["available"] = available
        owner["mirror_busy"] = self._mirror_busy
        return {"google": len(self._google_entries()), "owner": owner}

    def invalidate(self) -> None:
        with self._lock:
            self._google = None
            self._google_stamp = None
