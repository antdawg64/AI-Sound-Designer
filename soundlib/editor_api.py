"""The editor's page and API, added to the web app (web_app.py) with attach().

It is kept apart from Modal so the same code also runs on a laptop for testing
(scripts/editor_dev_server.py).

    GET  /edit/<job>                    the editor page
    GET  /api/edit/<job>                the clips, the saved edit, the last render
    GET  /api/edit/<job>/audio/<file>   one clip's sound
    POST /api/edit/<job>/save           save the edit (and the tuning record)
    POST /api/edit/<job>/render         make the edited video and stems
    GET  /api/edit/<job>/file/<name>    download an edited file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from . import config as C
from . import editor

ROOT = Path(__file__).resolve().parent.parent


def attach(api, *, jobs: Path, get_job: Callable[[str], dict | None], refresh: Callable[[], None],
           commit: Callable[[], None], keep_tuning_record: Callable[[str, dict], None],
           downloads: dict, web_folder: Path | None = None) -> None:
    """Add the editor to a FastAPI app.

    jobs: the folder holding one folder per job. get_job: job id -> its status.
    refresh / commit: pick up / publish file changes on shared storage.
    keep_tuning_record(path, record): store one editing session's record for good.
    """
    from fastapi import Body, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    page = next(p for p in ((web_folder or ROOT / "web") / "editor.html", ROOT / "web" / "editor.html")
                if p.exists()).read_text()

    def editable(job_id: str, fresh: bool = False) -> tuple[dict, Path, dict]:
        """The job, its out/ folder and its list of clips, or a 404."""
        job = get_job(job_id) if job_id.isalnum() else None
        if not job or job.get("state") != "done":
            raise HTTPException(404, "That video wasn't found, or it isn't finished yet.")
        out = jobs / job_id / "out"
        if fresh or not (out / editor.MANIFEST).exists():
            refresh()
        manifest = editor.load_manifest(out)
        if manifest is None:
            raise HTTPException(404, "This video was made before the editor existed. "
                                     "Please make it again to fine-tune it.")
        return job, out, manifest

    def keep(job_id: str, job: dict, manifest: dict, state: dict, body: dict) -> None:
        """Store what this person changed. Never breaks the page if it fails."""
        try:
            session = "".join(c for c in str(body.get("session", "")) if c.isalnum())[:32] or "none"
            report = jobs / job_id / "report.json"
            record = editor.tuning_record(
                manifest, state, job=job, session=session, events=body.get("events") or [],
                report=json.loads(report.read_text()) if report.exists() else None,
                renders=int(body.get("renders") or 0))
            if record["events"] or record["changed"]:   # opened and touched nothing: no record
                keep_tuning_record(f"/edits/{job_id}/{session}.json", record)
        except Exception as exc:
            print(f"tuning data not saved for {job_id}: {exc}", flush=True)

    @api.get("/edit/{job_id}", response_class=HTMLResponse)
    def edit_page(job_id: str):
        return page

    @api.get("/api/edit/{job_id}")
    def edit_info(job_id: str):
        job, out, manifest = editable(job_id, fresh=True)
        rendered = out / editor.EDITED / "render.json"
        return {"job": {k: job[k] for k in ("id", "name", "seconds", "test_tones") if k in job},
                "manifest": manifest, "state": editor.load_state(out, manifest),
                "rendered": json.loads(rendered.read_text()) if rendered.exists() else None,
                "keep_days": C.WEB_KEEP_DAYS, "gain_range_db": editor.GAIN_RANGE_DB}

    @api.get("/api/edit/{job_id}/audio/{name}")
    def edit_audio(job_id: str, name: str):
        job, out, manifest = editable(job_id)
        known = {c["file"] for c in manifest["clips"]} | {"music.flac", "voice.flac"}
        path = out / editor.FOLDER / name
        if name not in known or not path.exists():
            raise HTTPException(404, "Sound not found.")
        return FileResponse(path, media_type="audio/flac",
                            headers={"Cache-Control": "private, max-age=86400"})

    @api.post("/api/edit/{job_id}/save")
    def edit_save(job_id: str, body: dict = Body(...)):
        job, out, manifest = editable(job_id)
        state = editor.clean_state(body.get("state"), manifest)
        editor.save_state(out, state)
        commit()
        keep(job_id, job, manifest, state, body)
        return {"saved": True}

    @api.post("/api/edit/{job_id}/render")
    def edit_render(job_id: str, body: dict = Body(...)):
        job, out, manifest = editable(job_id)
        state = editor.clean_state(body.get("state"), manifest)
        editor.save_state(out, state)
        video = jobs / job_id / f"input{job.get('ext', '.mp4')}"
        if not video.exists():
            refresh()
        try:
            result = editor.render(out, video, state)
        except Exception as exc:
            print(f"edit render failed for {job_id}: {exc}", flush=True)
            raise HTTPException(500, "We couldn't make your version. Please try again.")
        commit()
        keep(job_id, job, manifest, state, body)
        return {"rendered": result["rendered"]}

    @api.get("/api/edit/{job_id}/file/{name}")
    def edit_download(job_id: str, name: str, download: int = 0):
        job, out, manifest = editable(job_id)
        if name not in downloads:
            raise HTTPException(404, "Not found.")
        path = out / editor.EDITED / name
        if not path.exists():
            refresh()
        if not path.exists():
            raise HTTPException(404, "File not found. Please try again in a moment.")
        nice, kind = downloads[name]
        filename = f"{job.get('name', 'video')}-edited-{nice}"
        return FileResponse(path, media_type=kind, filename=filename if download else None,
                            content_disposition_type="attachment" if download else "inline",
                            headers={"Cache-Control": "no-store"})
