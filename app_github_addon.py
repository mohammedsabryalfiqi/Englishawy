"""Gradio add-on (v5): GitHub repo URL -> background download/install/progress.

Wiring (3 lines in your app.py):
  1) from app_github_addon import register_github_tab, start_github_import, github_job_status
  2) inside build_ui(), in the owner_group AFTER the ZIP import button, call:
         github_url, github_branch, github_start, github_status, github_jobs = register_github_tab(owner_group)
         github_start.click(start_github_import, inputs=[github_url, github_branch, token_state],
                            outputs=[github_status, github_jobs])
  3) add a 15s gr.Timer to refresh: timer.tick(github_job_status, outputs=github_jobs)

How it survives page close: start_github_import() only INSERTs a row and starts
a daemon thread inside github_ingestor.py. The Gradio request returns instantly
with the job id; the thread keeps downloading/analyzing/installing in the
Space process. Re-opening the page (or /jobs in Telegram) reads progress from
the `github_jobs` SQLite table.
"""
from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger(__name__)


_SERVICES = None


def bind_services(services) -> None:
    """app.py hands over its singleton. Never `import app` from here: when app.py runs as
    `python app.py` that re-executes the whole file as a second module (second Services,
    second Telegram bridge, duplicate UI)."""
    global _SERVICES
    _SERVICES = services


def _services():
    if _SERVICES is None:
        raise RuntimeError("Services not bound; call bind_services(services) from app.py")
    return _SERVICES


def _job_table() -> list[list]:
    try:
        jobs = _services().ingestor.list_jobs(limit=15)
    except Exception:
        return []
    rows = []
    for j in jobs:
        rows.append([j["id"], f"{j['repo_owner']}/{j['repo_name']}", j["branch"],
                     j["status"], j["stage_label"][:80]])
    return rows


def start_github_import(repo_url: str, branch: str, token: str):
    try:
        _services().auth.require(token)
    except PermissionError as exc:
        raise gr.Error(str(exc)) from exc
    if not (repo_url or "").strip():
        raise gr.Error("الصق رابط مستودع GitHub أولاً. مثال: https://github.com/google/fonts")
    try:
        job = _services().ingestor.start_background_job((repo_url or "").strip(),
                                                        branch_override=(branch or "").strip())
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    msg = (f"🚀 بدأ التحميل في الخلفية: {job['repo_owner']}/{job['repo_name']} "
           f"(مهمة #{job['id']}). يمكنك إغلاق الصفحة — البوت/الخادم يكمل وحده. تابع من الجدول بالأسفل أو /jobs في تليجرام.")
    return msg, _job_table()


def github_job_status():
    return _job_table()


def register_github_tab(owner_group):
    """Call INSIDE the owner_group in build_ui(). Returns bound components."""
    with owner_group:
        gr.Markdown("### 🚀 سحب مستودع خطوط ضخم من GitHub (يعمل في الخلفية)\n"
                    "الصق رابط أي مستودع (حتى لو آلاف الخطوط). البوت/الخادم يحمّله ويثبّته ويحلّله **وحده في الخلفية** — "
                    "تخرج من الصفحة وترجع تلاقي التقدم محفوظاً.")
        with gr.Row():
            github_url = gr.Textbox(label="رابط مستودع GitHub",
                                    placeholder="https://github.com/google/fonts", scale=4)
            github_branch = gr.Textbox(label="الفرع (اختياري)", placeholder="main", scale=1)
            github_start = gr.Button("⬇️ ابدأ السحب الخلفي", variant="primary", scale=1)
        github_status = gr.Markdown()
        github_jobs = gr.Dataframe(headers=["ID", "المستودع", "الفرع", "الحالة", "المرحلة"],
                                   label="مهام السحب الخلفية (تتحدث تلقائياً)", interactive=False, wrap=True,
                                   value=_job_table())
    return github_url, github_branch, github_start, github_status, github_jobs
