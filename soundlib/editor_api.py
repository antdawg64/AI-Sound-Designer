"""The editor's page and API, added to the web app (web_app.py) with attach().

It is kept apart from Modal so the same code also runs on a laptop for testing
(scripts/editor_dev_server.py).

    GET  /edit/<job>                    the editor page
    GET  /api/edit/<job>                the clips, the saved edit, the last render
    GET  /api/edit/<job>/audio/<file>   one clip's sound
    POST /api/edit/<job>/save           save the edit (and the tuning record)
    POST /api/edit/<job>/render         make the edited video and stems
    GET  /api/edit/<job>/file/<name>    download an edited file
    GET  /api/edit/<job>/video          the picture, as a small copy that jumps to any frame at once
    POST /api/edit/<job>/add            ask for one more sound, by describing it
    GET  /api/edit/<job>/add/<id>       how that sound is coming along, and the clip when made
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path
from typing import Callable

from . import config as C
from . import editor

ROOT = Path(__file__).resolve().parent.parent


def attach(api, *, jobs: Path, get_job: Callable[[str], dict | None], refresh: Callable[[], None],
           commit: Callable[[], None], keep_tuning_record: Callable[[str, dict], None],
           downloads: dict, web_folder: Path | None = None,
           adds=None, start_add: Callable[[str, str, dict], None] | None = None) -> None:
    """Add the editor to a FastAPI app.

    jobs: the folder holding one folder per job. get_job: job id -> its status.
    refresh / commit: pick up / publish file changes on shared storage.
    keep_tuning_record(path, record): store one editing session's record for good.
    adds: a dict-like store, job id -> {add id: how that added sound is coming along}.
    start_add(job, add id, ask): start making an added sound in the background; it must
    end by putting {"state": "done" | "failed", ...} in adds. Without these two the
    editor has no "Add a sound".
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
        for _ in range(4):     # just finished: the files can lag, or a download blocks the refresh
            if (out / editor.MANIFEST).exists():
                break
            time.sleep(0.7)
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

    def video_of(job_id: str, job: dict) -> Path:
        video = jobs / job_id / f"input{job.get('ext', '.mp4')}"
        if not video.exists():
            refresh()
        return video

    making_preview = threading.Lock()

    @api.get("/api/edit/{job_id}/video")
    def edit_video(job_id: str):
        """The picture for the editor. Made the first time it is asked for (a few seconds)."""
        job, out, manifest = editable(job_id)
        path = out / editor.FOLDER / editor.PREVIEW
        if not path.exists():
            with making_preview:
                if not path.exists():
                    try:
                        editor.preview_video(video_of(job_id, job), out)
                        commit()
                    except Exception as exc:
                        print(f"editor preview not made for {job_id}: {exc}", flush=True)
                        raise HTTPException(404, "No preview.")
        return FileResponse(path, media_type="video/mp4",
                            headers={"Cache-Control": "private, max-age=86400"})

    @api.post("/api/edit/{job_id}/add")
    def edit_add(job_id: str, body: dict = Body(...)):
        job, out, manifest = editable(job_id)
        if adds is None or start_add is None:
            raise HTTPException(404, "Adding sounds isn't available here.")
        lengths = sorted(float(s) for s in C.EDITOR_ADD_SECONDS)
        refine = None
        if body.get("refine"):
            # Make one of the video's short sounds again, changed ("punchier", "softer").
            old = next((c for c in manifest["clips"] if c["id"] == body.get("refine")), None)
            tweak = " ".join(str(body.get("tweak", "")).split())[:80]
            if old is None or old["kind"] != "moment" or len(tweak) < 2:
                raise HTTPException(400, "Please say how the sound should change.")
            prompt = old["label"].split(": ", 1)[-1] if old.get("role") in ("whoosh", "detail") else old["label"]
            at = float(old.get("time", old["start"]) if old.get("role", "main") == "main" else old["start"])
            seconds = float(old.get("seconds") or min(lengths, key=lambda s: abs(s - old["duration"])))
            step = 1 if "longer" in tweak.lower() else -1 if "shorter" in tweak.lower() else 0
            seconds = lengths[min(max(lengths.index(seconds) + step, 0), len(lengths) - 1)] if seconds in lengths else seconds
            refine = {"of": old["id"], "previous": old.get("made_as") or prompt, "tweak": tweak}
        else:
            prompt = " ".join(str(body.get("prompt", "")).split())[:200]
            if len(prompt) < 3:
                raise HTTPException(400, "Please describe the sound in a few words.")
            try:
                at, seconds = float(body.get("time")), float(body.get("seconds"))
            except (TypeError, ValueError):
                raise HTTPException(400, "Please choose where the sound goes.")
            if not 0 <= at <= manifest["duration"] or seconds not in lengths:
                raise HTTPException(400, "Please choose where the sound goes.")
        mine = dict(adds.get(job_id) or {})
        under_way = [a for a in mine.values() if a.get("state") == "working" and time.time() - a["asked"] < 900]
        if len(under_way) >= 5:
            raise HTTPException(429, "Five sounds are already being made. Please wait for one to finish.")
        have = sum(bool(c.get("added")) for c in manifest["clips"]) + len(under_way)
        if have >= int(C.EDITOR_ADD_MAX):
            raise HTTPException(429, f"You can add up to {int(C.EDITOR_ADD_MAX)} sounds to one video.")
        add_id = "a" + secrets.token_hex(4)
        ask = {"prompt": prompt, "time": round(at, 4), "seconds": seconds, **({"refine": refine} if refine else {})}
        mine[add_id] = {"state": "working", "asked": time.time(), **ask}
        adds[job_id] = mine
        start_add(job_id, add_id, ask)
        return {"id": add_id, "prompt": prompt, "time": ask["time"]}

    @api.get("/api/edit/{job_id}/add/{add_id}")
    def edit_added(job_id: str, add_id: str):
        job, out, manifest = editable(job_id)
        row = (adds.get(job_id) or {}).get(add_id) if adds is not None else None
        if not row:
            raise HTTPException(404, "That sound wasn't found.")
        if row["state"] == "working" and time.time() - row["asked"] > 900:
            return {"state": "failed", "error": "It took too long. Please try again."}
        if row["state"] != "done":
            return {"state": row["state"], "error": row.get("error")}
        for _ in range(4):          # just made on another machine: its files can lag a moment
            refresh()
            manifest = editor.load_manifest(out)
            clip = next((c for c in manifest["clips"] if c["id"] == add_id), None)
            if clip:
                piece = next(p for p in editor.initial_pieces(manifest) if p["src"] == add_id)
                return {"state": "done", "clip": clip, "piece": piece}
            time.sleep(0.7)
        return {"state": "working"}

    @api.get("/edit/{job_id}", response_class=HTMLResponse)
    def edit_page(job_id: str):
        return page

    @api.get("/api/edit/{job_id}")
    def edit_info(job_id: str):
        job, out, manifest = editable(job_id, fresh=True)
        rendered = out / editor.EDITED / "render.json"
        try:
            from . import media

            fps = float(media.probe(video_of(job_id, job))["fps"]) or 30.0
        except Exception:
            fps = 30.0
        # Sounds still being made (asked for before this page was opened or reloaded).
        have = {c["id"] for c in manifest["clips"]}
        pending = [{"id": i, "prompt": a.get("prompt", ""), "time": a.get("time", 0.0),
                    "tweak": (a.get("refine") or {}).get("tweak")}
                   for i, a in ((adds.get(job_id) or {}) if adds is not None else {}).items()
                   if a.get("state") == "working" and time.time() - a.get("asked", 0) < 900 and i not in have]
        return {"job": {k: job[k] for k in ("id", "name", "seconds", "test_tones") if k in job},
                "pending": pending,
                "fps": round(min(max(fps, 5.0), 120.0), 4),
                "can_add": adds is not None and start_add is not None,
                "add_seconds": list(C.EDITOR_ADD_SECONDS), "add_max": int(C.EDITOR_ADD_MAX),
                "manifest": manifest, "initial": editor.initial_pieces(manifest),
                "state": editor.load_state(out, manifest),
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
        video = video_of(job_id, job)
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
