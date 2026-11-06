"""Background GitHub font-repo ingestor (owner library auto-fill).

The owner pastes ANY GitHub font-repo URL (even a HUGE one). The bot/server:
  1) downloads the repo zipball itself (codeload.github.com, streaming),
  2) analyses + installs every font face via the existing FontStore
     (safe ZIP handling, table validation, SHA-256 dedupe),
  3) keeps running in a DAEMON THREAD so leaving the upload page / closing
     the browser NEVER stops it,
  4) records live progress in SQLite table `github_jobs` (poll from Gradio
     or Telegram /jobs), and fires an optional Telegram callback on completion.

Only stdlib + font_store are used. No new hosting dependency.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
CHUNK = 1024 * 1024


def parse_github_url(raw_url: str) -> tuple[str, str, str, str]:
    """Parse GitHub URL -> (owner, repo, branch, subpath).

    Accepts:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      owner/repo
      https://github.com/owner/repo/tree/<branch>[/sub/path]
      https://github.com/owner/repo/blob/<branch>/...
    """
    cleaned = (raw_url or "").strip().removesuffix(".git").rstrip("/")
    if not cleaned:
        raise ValueError("الصق رابط مستودع GitHub أولاً. مثال: https://github.com/google/fonts")
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned
    parsed = urlparse(cleaned)
    if "github.com" not in parsed.netloc.lower():
        raise ValueError("الرابط يجب أن يكون من موقع github.com")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        # maybe "owner/repo" without host path issue -> re-try
        raise ValueError("رابط المستودع غير مكتمل (مثال: https://github.com/google/fonts)")
    owner, repo = parts[0], parts[1]
    branch, subpath = "main", ""
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        branch = parts[3]
        subpath = "/".join(parts[4:])
    return owner, repo, branch, subpath


class GithubRepoIngestor:
    """Persistent background-job manager backed by the FontStore SQLite DB."""

    def __init__(self, store, notify_callback: Callable[[str], None] | None = None):
        self.store = store
        self.notify_callback = notify_callback
        self._lock = threading.RLock()
        with self.store._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS github_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_url TEXT NOT NULL,
                    repo_owner TEXT NOT NULL,
                    repo_name TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    subpath TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    stage_label TEXT NOT NULL DEFAULT '',
                    total_files INTEGER NOT NULL DEFAULT 0,
                    processed_files INTEGER NOT NULL DEFAULT 0,
                    added_faces INTEGER NOT NULL DEFAULT 0,
                    duplicate_faces INTEGER NOT NULL DEFAULT 0,
                    failed_files INTEGER NOT NULL DEFAULT 0,
                    bytes_downloaded INTEGER NOT NULL DEFAULT 0,
                    error_text TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)

    # -- public API -----------------------------------------------------
    def start_background_job(self, repo_url: str, branch_override: str = "",
                             subpath_override: str = "") -> dict:
        owner, repo, branch, subpath = parse_github_url(repo_url)
        if branch_override.strip():
            branch = branch_override.strip()
        if subpath_override.strip():
            subpath = subpath_override.strip().strip("/")
        now = time.time()
        with self.store._connect() as conn:
            cur = conn.execute(
                """INSERT INTO github_jobs
                   (repo_url, repo_owner, repo_name, branch, subpath, status, stage_label, created_at, updated_at)
                   VALUES (?,?,?,?,?,'queued','في قائمة الانتظار للبدء في الخلفية...',?,?)""",
                (f"https://github.com/{owner}/{repo}", owner, repo, branch, subpath, now, now),
            )
            job_id = int(cur.lastrowid)
        th = threading.Thread(target=self._run_job, args=(job_id, owner, repo, branch),
                              name=f"github-ingest-{job_id}", daemon=True)
        th.start()  # daemon => page close / browser close never stops it
        return self.get_job(job_id)

    def get_job(self, job_id: int) -> dict:
        with self.store._connect() as conn:
            row = conn.execute("SELECT * FROM github_jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else {}

    def list_jobs(self, limit: int = 20) -> list[dict]:
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM github_jobs ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
            return [dict(r) for r in rows]

    # -- internals ------------------------------------------------------
    def _update_job(self, job_id: int, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [job_id]
        with self.store._connect() as conn:
            conn.execute(f"UPDATE github_jobs SET {cols} WHERE id=?", vals)

    def _download_zipball(self, owner: str, repo: str, branch: str,
                          dest: Path, job_id: int) -> int:
        last_err: Exception | None = None
        for b in dict.fromkeys([branch, "main", "master"]):  # try requested, then defaults
            url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{b}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Englishawy-Bot/5.0"})
                downloaded = 0
                with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (8 * CHUNK) < CHUNK:
                            self._update_job(job_id, status="downloading",
                                             bytes_downloaded=downloaded,
                                             stage_label=f"جاري تنزيل المستودع في الخلفية ({downloaded/1048576:.1f} MB)...")
                if b != branch:
                    self._update_job(job_id, branch=b)
                return downloaded
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"تعذر تنزيل المستودع {owner}/{repo}: {last_err}")

    def _run_job(self, job_id: int, owner: str, repo: str, branch: str) -> None:
        try:
            self._update_job(job_id, status="downloading",
                             stage_label=f"جاري تنزيل {owner}/{repo} في الخلفية...")
            with TemporaryDirectory(prefix="gh-repo-", dir=self.store.tmp_dir) as work:
                zp = Path(work) / "repo.zip"
                downloaded = self._download_zipball(owner, repo, branch, zp, job_id)
                self._update_job(job_id, status="analyzing",
                                 stage_label="جاري فك الضغط وتحليل وتثبيت الخطوط...")

                def on_progress(done: int, total: int, name: str):
                    self._update_job(job_id, status="analyzing", total_files=total,
                                     processed_files=done,
                                     stage_label=f"فحص وتثبيت ({done}/{total}): {name[:50]}")

                report = self.store.import_zip(zp, progress=on_progress)
                self._update_job(job_id, status="completed",
                                 total_files=report.get("files", 0),
                                 processed_files=report.get("files", 0),
                                 added_faces=report.get("added", 0),
                                 duplicate_faces=report.get("duplicates", 0),
                                 failed_files=len(report.get("rejected", [])),
                                 bytes_downloaded=downloaded,
                                 stage_label=(f"✅ اكتمل في الخلفية: +{report.get('added',0)} وجه جديد، "
                                              f"{report.get('duplicates',0)} مكرر ({downloaded/1048576:.1f} MB)"))
                if self.notify_callback:
                    try:
                        self.notify_callback(
                            f"🎉 اكتمل تحميل وتثبيت مستودع GitHub في الخلفية!\n"
                            f"📦 {owner}/{repo}\n"
                            f"✨ وجوه جديدة: {report.get('added',0)}\n"
                            f"♻️ مكررات: {report.get('duplicates',0)}\n"
                            f"💾 الحجم: {downloaded/1048576:.1f} MB")
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001 - must persist failure, never raise in thread
            logger.error("Background GitHub job %s failed: %s", job_id, exc)
            self._update_job(job_id, status="failed", error_text=str(exc)[:300],
                             stage_label=f"⚠️ فشل التحميل الخلفي: {str(exc)[:180]}")
