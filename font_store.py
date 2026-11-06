"""Persistent owner-font library with streaming, sandbox-free ZIP ingestion.

There is deliberately NO application archive/font size limit. Disk capacity, the
reverse proxy, browser and hosting platform still impose real limits. We stream
members to disk, validate font tables, never use extractall(), and store metadata
in SQLite beside content-addressed font files. Only explicitly configured attached
storage survives a Space restart; see config.storage_status().
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import stat
import struct
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from config import FONT_SUFFIXES, data_dir
from font_analysis import inspect_face

Progress = Callable[[int, int, str], None]
CHUNK = 1024 * 1024
DISK_RESERVE = 64 * 1024 * 1024


class FontStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).resolve() if root else data_dir()
        self.fonts_dir = self.root / "fonts"
        self.tmp_dir = self.root / "tmp"
        self.db_path = self.root / "library.sqlite3"
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._size_cache: tuple[int, float] = (0, 0.0)
        self._size_busy = False
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE IF NOT EXISTS fonts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_sha TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    style TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'owner',
                    path TEXT NOT NULL,
                    analysis_json TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(content_sha, face_index)
                );
                CREATE INDEX IF NOT EXISTS fonts_enabled ON fonts(enabled);
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _check_disk(self, needed: int = CHUNK) -> None:
        if shutil.disk_usage(self.root).free < needed + DISK_RESERVE:
            raise RuntimeError("مساحة التخزين المتاحة لا تكفي بأمان. وسّع Volume أو احذف خطوطاً قديمة.")

    def _copy_atomic(self, source: Path, destination: Path) -> bool:
        if destination.exists():
            if os.path.samefile(source, destination) or self._sha256(source) == self._sha256(destination):
                return False
            raise RuntimeError("تعارض غير متوقع في ملف الخط")
        self._check_disk(source.stat().st_size)
        part = destination.with_name(f".part-{uuid.uuid4().hex}")
        try:
            with source.open("rb") as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(part, destination)
            return True
        finally:
            part.unlink(missing_ok=True)

    def install_font(self, path: str | Path, origin: str = "upload") -> dict:
        """Validate every face, install once by content hash, return added/duplicate counts."""
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("ملف الخط غير موجود أو رابط رمزي")
        suffix = path.suffix.lower()
        if suffix not in FONT_SUFFIXES:
            raise ValueError(f"صيغة غير مدعومة: {suffix}")
        digest = self._sha256(path)
        origin = " ".join(str(origin).split())[:240] or "upload"
        created: list[Path] = []
        records: list[tuple] = []
        with self._lock, tempfile.TemporaryDirectory(prefix="face-", dir=self.tmp_dir) as work:
            if suffix in {".woff", ".woff2"}:
                from fontTools.ttLib import TTFont
                # Pillow cannot consistently load webfonts. Normalize to sfnt while
                # retaining outlines/name/cmap tables; lazy input avoids a full read.
                with TTFont(str(path), lazy=True) as webfont:
                    if webfont.flavor not in {"woff", "woff2"}:
                        raise ValueError("ملف لا يحمل توقيع WOFF صالحاً")
                    normalized = Path(work) / "normalized.ttf"
                    webfont.flavor = None
                    webfont.save(str(normalized))
                sources = [(normalized, 0)]
            elif suffix == ".ttc":
                with path.open("rb") as stream:
                    header = stream.read(12)
                if len(header) != 12 or header[:4] not in {b"ttcf", b"OTTO"}:
                    raise ValueError("مجموعة خطوط TTC غير صالحة")
                count = struct.unpack(">I", header[8:12])[0] if header[:4] == b"ttcf" else 1
                if not 1 <= count <= 4096:
                    raise ValueError("عدد وجوه TTC غير صالح")
                sources = [(path, i) for i in range(count)]
            else:
                sources = [(path, 0)]

            with self._connect() as conn:
                existing = {(r["content_sha"], r["face_index"]) for r in conn.execute(
                    "SELECT content_sha, face_index FROM fonts WHERE content_sha=?", (digest,)
                )}
            for source, face_index in sources:
                if (digest, face_index) in existing:
                    continue
                meta = inspect_face(source, face_index)
                destination = self.fonts_dir / f"{digest}-{face_index}{meta['extension']}"
                if self._copy_atomic(source, destination):
                    created.append(destination)
                records.append((digest, face_index, meta["family"], meta["style"], "owner",
                                str(destination), meta["analysis_json"], origin, time.time(), 1))
            if not records:
                return {"added": 0, "duplicates": len(sources), "families": []}
            try:
                with self._connect() as conn:
                    conn.executemany("""INSERT OR IGNORE INTO fonts
                        (content_sha,face_index,family,style,category,path,analysis_json,origin,created_at,enabled)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""", records)
            except Exception:
                for item in created:
                    item.unlink(missing_ok=True)
                raise
        # Count only rows actually inserted (INSERT OR IGNORE may race another worker).
        return {"added": len(records), "duplicates": len(sources) - len(records),
                "families": sorted({r[2] for r in records})}

    def _spool_member(self, archive: zipfile.ZipFile, member: zipfile.ZipInfo, suffix: str) -> Path:
        handle = tempfile.NamedTemporaryFile(prefix="member-", suffix=suffix, dir=self.tmp_dir, delete=False)
        path = Path(handle.name)
        try:
            with handle, archive.open(member) as src:
                for chunk in iter(lambda: src.read(CHUNK), b""):
                    self._check_disk(len(chunk))
                    handle.write(chunk)
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def import_zip(self, path: str | Path, progress: Progress | None = None) -> dict:
        path = Path(path)
        if path.suffix.lower() != ".zip":
            raise ValueError("الأرشيف يجب أن يكون ZIP")
        report = {"archive": path.name[:180], "bytes": path.stat().st_size, "font_files": 0,
                  "added": 0, "duplicates": 0, "ignored": 0, "invalid": [], "nested_archives": 0}
        self._walk_zip(path, report, 0, progress)
        return report

    def _walk_zip(self, path: Path, report: dict, depth: int, progress: Progress | None) -> None:
        try:
            archive_context = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as exc:
            report["invalid"].append({"file": path.name[:180], "reason": f"أرشيف تالف: {type(exc).__name__}"})
            return
        with archive_context as archive:
            members = archive.infolist()
            for index, member in enumerate(members, 1):
                name = member.filename.replace("\\", "/")
                display = name.rsplit("/", 1)[-1][:180] or "(unnamed)"
                if progress:
                    progress(index, max(len(members), 1), display)
                if member.is_dir():
                    continue
                mode = member.external_attr >> 16
                parts = name.split("/")
                unsafe_name = name.startswith("/") or ".." in parts or (len(name) > 1 and name[1] == ":")
                if unsafe_name or stat.S_ISLNK(mode) or member.flag_bits & 1:
                    report["ignored"] += 1
                    continue
                suffix = Path(name).suffix.lower()
                if suffix not in FONT_SUFFIXES | {".zip"}:
                    report["ignored"] += 1
                    continue
                temporary = None
                try:
                    temporary = self._spool_member(archive, member, suffix)
                    if suffix == ".zip":
                        report["nested_archives"] += 1
                        if depth >= 3:
                            report["invalid"].append({"file": display, "reason": "تجاوز عمق الأرشيفات المتداخلة"})
                        else:
                            self._walk_zip(temporary, report, depth + 1, None)
                    else:
                        report["font_files"] += 1
                        result = self.install_font(temporary, origin=report["archive"])
                        report["added"] += result["added"]
                        report["duplicates"] += result["duplicates"]
                except Exception as exc:
                    # Never abort a whole collection because one font is malformed.
                    reason = str(exc)
                    if not isinstance(exc, (ValueError, RuntimeError, OSError, zipfile.BadZipFile)) and "font" not in type(exc).__name__.lower():
                        reason = f"{type(exc).__name__}: تعذر تحليل الخط"
                    report["invalid"].append({"file": display, "reason": reason[:240]})
                finally:
                    if temporary:
                        temporary.unlink(missing_ok=True)

    @staticmethod
    def _row_dict(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["source"] = "owner"
        item["slug"] = f"owner:{item['id']}"
        return item

    def entries(self, query: str = "", limit: int = 100, offset: int = 0) -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        with self._connect() as conn:
            if query:
                rows = conn.execute("SELECT * FROM fonts WHERE enabled=1 AND (family LIKE ? OR style LIKE ?) ORDER BY family,id LIMIT ? OFFSET ?",
                                    (f"%{query}%", f"%{query}%", limit, offset)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM fonts WHERE enabled=1 ORDER BY family,id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        # لا نفحص is_file لكل صف: على Volume شبكي كل stat بطيء وكان يجمّد لوحة المالك.
        return [self._row_dict(r) for r in rows]

    def all_entries(self) -> list[dict]:
        with self._connect() as conn:
            # بدون analysis_json: عمود ثقيل لا يحتاجه المسح ويضاعف الذاكرة مع المكتبات الكبيرة
            rows = conn.execute("SELECT id, content_sha, face_index, family, style, category, path, origin "
                                "FROM fonts WHERE enabled=1 ORDER BY id").fetchall()
        return [self._row_dict(r) for r in rows]

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) faces, COUNT(DISTINCT family) families, COUNT(DISTINCT content_sha) files FROM fonts WHERE enabled=1").fetchone()
        return {"faces": row["faces"], "families": row["families"], "files": row["files"],
                "bytes": self._bytes_cached()}

    def _bytes_cached(self) -> int:
        """Total size needs one stat() per file - minutes on a network Volume. Compute it in the
        background (single-flight, refreshed every 10 min) and always answer instantly."""
        value, stamp = self._size_cache
        if time.time() - stamp > 600 and not self._size_busy:
            self._size_busy = True
            threading.Thread(target=self._refresh_size, name="library-size", daemon=True).start()
        return value

    def _refresh_size(self) -> None:
        try:
            total = 0
            for p in self.fonts_dir.iterdir():
                if not p.name.startswith(".part-"):
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
            self._size_cache = (total, time.time())
        finally:
            self._size_busy = False

    def delete(self, font_id: int) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT path FROM fonts WHERE id=?", (int(font_id),)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM fonts WHERE id=?", (int(font_id),))
            shared = conn.execute("SELECT 1 FROM fonts WHERE path=?", (row["path"],)).fetchone()
            if not shared:
                Path(row["path"]).unlink(missing_ok=True)
        return True

    def rebuild_index(self) -> int:
        """Recover metadata if font files survived but the SQLite index was lost."""
        added = 0
        for path in sorted(self.fonts_dir.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"} and not path.name.startswith(".part-"):
                added += self.install_font(path, origin="recovered")["added"]
        return added

    def cleanup_temp(self) -> None:
        """Called at startup, when no ingestion is running; never touches installed fonts."""
        for path in self.tmp_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def format_import_report(report: dict) -> str:
    lines = [f"📦 {report['archive']}",
             f"✅ وجوه خطوط جديدة: {report['added']} | مكررة: {report['duplicates']}",
             f"ملفات خطوط مفحوصة: {report['font_files']} | ملفات أخرى متجاهلة: {report['ignored']}"]
    if report["nested_archives"]:
        lines.append(f"أرشيفات متداخلة: {report['nested_archives']}")
    if report["invalid"]:
        lines.append(f"⚠️ تعذر قبول {len(report['invalid'])} ملفاً:")
        lines.extend(f"• {x['file']}: {x['reason']}" for x in report["invalid"][:8])
    return "\n".join(lines)
