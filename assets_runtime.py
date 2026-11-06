"""Non-blocking, single-flight asset preparation for a public Gradio Space."""
from __future__ import annotations

import os
import threading

_state = {"phase": "idle", "done": 0, "total": 0, "error": ""}
_lock = threading.Lock()
_thread: threading.Thread | None = None


def snapshot() -> dict:
    with _lock:
        return dict(_state)


def start() -> None:
    global _thread
    if os.getenv("ENGLISHAWY_AUTO_SETUP", "1").lower() in {"0", "false", "no"}:
        with _lock:
            _state["phase"] = "disabled"
        return
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _state.update(phase="preparing", done=0, total=0, error="")
        _thread = threading.Thread(target=_run, name="englishawy-assets", daemon=True)
        _thread.start()


def _run() -> None:
    import setup_assets
    try:
        models = setup_assets.setup_models(include_onnx=True)
        if not models["tesseract"]:
            raise RuntimeError("تعذر تنزيل بيانات Tesseract. شغّل setup_assets.py أو تحقق من الشبكة.")
        def progress(done, total):
            with _lock:
                _state.update(done=done, total=total)
        result = setup_assets.setup_fonts(workers=int(os.getenv("ASSET_WORKERS", "12")), progress=progress)
        with _lock:
            _state.update(phase="ready", done=result["available"], total=result["requested"])
            if result["failed"]:
                _state["error"] = f"{len(result['failed'])} خطوط Google غير متاحة؛ المكتبة الحالية تعمل."
            if not models["onnx"]:
                _state["error"] += " نموذج CPU غير متاح؛ استخدم ZeroGPU أو جهّز النموذج."
    except Exception as exc:
        with _lock:
            _state.update(phase="error", error=f"{type(exc).__name__}: {exc}"[:400])
