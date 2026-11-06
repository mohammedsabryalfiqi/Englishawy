"""One searchable catalog: cached Google fonts + freshly uploaded owner faces."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import CACHE_DIR, ROOT
from font_store import FontStore

FONTS_DIR = CACHE_DIR / "fonts"
logger = logging.getLogger(__name__)

# مكتبة المالك تعيش على Volume شبكي (/data): كل open/stat هناك بطيء، فمسح آلاف الخطوط منه
# كان "يعلق" بلا CPU ولا رام. نخلق نسخة محلية سريعة (قرص الحاوية) في الخلفية بعدة خيوط
# (الانتظار I/O فيتوازى)، والمسح يقرأ من النسخة المحلية فقط.
MIRROR_DIR = Path(os.getenv("ENGLISHAWY_LOCAL_MIRROR", str(Path(tempfile.gettempdir()) / "englishawy-mirror")))
MIRROR_WORKERS = max(1, int(os.getenv("MIRROR_WORKERS", "16")))
MIRROR_ENABLED = os.getenv("MIRROR_OWNER_FONTS", "1").lower() not in {"0", "false", "no"}
MIRROR_MIN_FREE = 512 * 1024 * 1024
MIRROR_SYNC_LIMIT = int(os.getenv("MIRROR_SYNC_LIMIT", "200"))  # أقل من هذا العدد: انسخ فوراً وانتظر


class FontCatalog:
    def __init__(self, store: FontStore):
        self.store = store
        self._lock = threading.RLock()
        self._google: list[dict] | None = None
        self._google_stamp = None
        self._mirror_lock = threading.Lock()
        self._mirrored: dict[str, str] = {}      # content_sha -> local path
        self._mirror_failed: set[str] = set()
        self._mirror_busy = False
        self._mirror_total = 0

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
        except Exception as exc:
            logger.warning("mirror failed for one font: %s", type(exc).__name__)
            with self._mirror_lock:
                self._mirror_failed.add(sha)

    def _mirror_worker(self, todo: list[dict]) -> None:
        try:
            logger.info("mirroring %d owner font files to local disk", len(todo))
            with ThreadPoolExecutor(max_workers=MIRROR_WORKERS, thread_name_prefix="mirror") as pool:
                list(pool.map(self._copy_one, todo))
            logger.info("owner font mirror finished")
        finally:
            with self._mirror_lock:
                self._mirror_busy = False

    def warm(self, owner_entries: list[dict] | None = None) -> None:
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
                if sha in self._mirrored or sha in self._mirror_failed:
                    continue
                if self._mirror_target(e).exists():       # قرص محلي: stat سريع
                    self._mirrored[sha] = str(self._mirror_target(e))
                    continue
                todo.append(e)
            self._mirror_total = len(seen)
            if not todo or self._mirror_busy:
                return
            self._mirror_busy = True
        if len(todo) <= MIRROR_SYNC_LIMIT:
            self._mirror_worker(todo)   # دفعة صغيرة (رفع خط/ZIP صغير): جاهزة قبل أن نكمل
            return
        threading.Thread(target=self._mirror_worker, args=(todo,), name="mirror-coordinator", daemon=True).start()

    def mirror_progress(self) -> tuple[int, int]:
        with self._mirror_lock:
            return len(self._mirrored), max(self._mirror_total, len(self._mirrored))

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
                local = dict(self._mirrored)
            # الجاهز محلياً فقط يدخل المسح؛ الباقي يُنسخ في الخلفية (لا ننتظر Volume بطيء)
            owner = [{**e, "path": local[e["content_sha"]]} for e in owner if e.get("content_sha") in local]
        return owner + google

    def stats(self) -> dict:
        owner = self.store.stats()
        ready, total = self.mirror_progress()
        owner["mirrored"], owner["mirror_total"] = ready, total
        return {"google": len(self._google_entries()), "owner": owner}

    def invalidate(self) -> None:
        with self._lock:
            self._google = None
            self._google_stamp = None
