"""Tiny diagnostics: where is every thread RIGHT NOW? (code paths only - no tokens, no user data)"""
from __future__ import annotations

import sys
import threading
import traceback


def dump_threads(limit_chars: int = 12000) -> str:
    names = {t.ident: t.name for t in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        stack = traceback.extract_stack(frame)[-8:]
        lines = [f"  {f.filename.split('/')[-1]}:{f.lineno} in {f.name}" for f in stack]
        parts.append(f"# {names.get(ident, ident)}\n" + "\n".join(lines))
    return "\n".join(parts)[:limit_chars]
