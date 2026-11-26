"""The public web page for AI Sound Designer, running on Modal (Phase 3).

Someone opens the page, picks a video, and gets it back with AI sound effects and
music, plus the music and the effects as separate downloads.

How it fits together:
  web      - the page itself and its small API (upload, progress, downloads)
  run_job  - runs the same pipeline as make_sound.py for one uploaded video, using
             the AI models in the "ai-sound-designer" app (modal_app.py)
  cleanup  - once a day, deletes jobs older than web_keep_days

Every sound setting comes from settings.json, exactly as for make_sound.py and tune.py,
so a tuning round that changes settings.json changes the web page too after a redeploy:

    python make_sound.py --deploy        (deploys modal_app.py and this file)

Licence rules (see settings.json "web"): the effects model is not licensed in the EU,
the UK or South Korea, so the page refuses uploads from there, checked two ways: the
country the person says they are in, and the country their internet address is in
(DB-IP Lite database, CC BY 4.0, credited on the page).
"""

import json
import time
from pathlib import Path

import modal

from soundlib import config as C

APP_NAME = "ai-sound-designer-web"
app = modal.App(APP_NAME)
ROOT = Path(__file__).resolve().parent

jobs_volume = modal.Volume.from_name("ai-sound-designer-jobs", create_if_missing=True)
# What people change in the editor, kept for good (jobs are deleted after a few days) so
# analyze_edits.py can learn better defaults from it. One small file per editing session.
TUNING_VOLUME = "ai-sound-designer-tuning-data"
# The shared password that keeps the site private (key SITE_PASSWORD). See soundlib/gate.py.
SITE_PASSWORD_SECRET = "site-password"
job_status = modal.Dict.from_name("ai-sound-designer-jobs", create_if_missing=True)
JOBS = Path("/jobs")
GEO_DB = "/geo/country.mmdb"
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
DOWNLOADS = {   # file in the job's out/ folder -> (download name, kind)
    "final.mp4": ("with-sound.mp4", "video/mp4"),
    "music.wav": ("music.wav", "audio/wav"),
    "sfx.wav": ("sound-effects.wav", "audio/wav"),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "curl", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy>=2.0", "scipy>=1.14", "soundfile>=0.13", "pyloudnorm>=0.1",
        "scenedetect>=0.6", "opencv-python-headless>=4.10", "onnxruntime>=1.20",
        "fastapi[standard]>=0.115", "python-multipart>=0.0.18", "maxminddb>=2.6",
    )
    # Country lookup database (free DB-IP Lite, refreshed whenever this line changes).
    .run_commands(
        "mkdir -p /geo && cd /geo && for m in $(date +%Y-%m) $(date -d '-1 month' +%Y-%m) "
        "$(date -d '-2 month' +%Y-%m); do curl -fsSL "
        "https://download.db-ip.com/free/dbip-country-lite-$m.mmdb.gz -o c.gz && break; done; "
        "(gunzip -c c.gz > country.mmdb && rm c.gz) || echo 'country database not downloaded'",
    )
    .env({"SOUND_SETTINGS": "/root/settings.json"})
    .add_local_dir(ROOT / "soundlib", "/root/soundlib", ignore=["**/__pycache__"])
    .add_local_file(ROOT / "make_sound.py", "/root/make_sound.py")
    .add_local_dir(ROOT / "web", "/root/web")
    .add_local_file(C.SETTINGS_FILE, "/root/settings.json")
)


# --- Small helpers ---------------------------------------------------------------

def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _update(job_id: str, **changes) -> dict:
    job = {**job_status.get(job_id, {}), **changes, "updated": time.time()}
    job_status[job_id] = job
    return job


def _country_of(ip: str | None) -> str | None:
    """Two-letter country code for an internet address, or None if unknown."""
    if not ip:
        return None
    try:
        import maxminddb

        with maxminddb.open_database(GEO_DB) as db:
            found = db.get(ip) or {}
        return (found.get("country") or {}).get("iso_code")
    except Exception:
        return None


def page_url() -> str:
    return modal.Function.from_name(APP_NAME, "web").get_web_url()


# --- The work: one uploaded video -------------------------------------------------

@app.function(image=image, volumes={str(JOBS): jobs_volume}, cpu=4.0, memory=8192,
              timeout=3600, max_containers=3)
def run_job(job_id: str, test_tones: bool = False) -> dict:
    """Make the sound for one uploaded video. Progress goes to the job status as it runs."""
    import traceback

    import make_sound as M

    jobs_volume.reload()
    folder = JOBS / job_id
    job = job_status.get(job_id, {})
    video = folder / f"input{job.get('ext', '.mp4')}"

    def progress(text: str) -> None:
        print(text, flush=True)
        line = text.strip()
        if len(line) > 4 and line[0].isdigit() and line[1:3] == "/6":
            _update(job_id, step=line[4:], step_number=int(line[0]))

    M.say = progress
    _update(job_id, state="working", step="Starting up", step_number=0, started=time.time())
    try:
        backend = M.TestTones() if test_tones else M.Cloud()
        report = M.make_sound(video, folder / "out", backend, vibe=job.get("vibe") or None,
                              parts=job.get("parts", "both"), detail=job.get("detail") or None)
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        jobs_volume.commit()
        _update(job_id, state="done", step="Finished", step_number=6, passed=report["passed"],
                cost=report["cost"]["total_dollars"], cost_parts=cost_summary(report["cost"]),
                wall_seconds=report["wall_seconds"], test_tones=test_tones)
        return {"passed": report["passed"], "cost": report["cost"]["total_dollars"]}
    except Exception as exc:
        traceback.print_exc()
        (folder / "error.txt").write_text(traceback.format_exc())
        jobs_volume.commit()
        _update(job_id, state="failed", error=str(exc)[:300])
        return {"passed": False, "error": str(exc)}


@app.function(image=image, volumes={str(JOBS): jobs_volume}, schedule=modal.Cron("17 4 * * *"))
def cleanup() -> int:
    """Delete jobs older than web_keep_days (the files and the status)."""
    import shutil

    jobs_volume.reload()
    cutoff, removed = time.time() - float(C.WEB_KEEP_DAYS) * 86400, 0
    for folder in JOBS.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            job_status.pop(folder.name, None)
            for add_id in job_status.pop(f"adds:{folder.name}", None) or []:
                job_status.pop(f"add:{folder.name}:{add_id}", None)
            removed += 1
    jobs_volume.commit()
    return removed


@app.function(image=image, volumes={str(JOBS): jobs_volume}, cpu=2.0, memory=4096, timeout=900,
              max_containers=6)
def add_sound(job_id: str, add_id: str, ask: dict) -> dict:
    """Make one more sound for a finished video, asked for in the editor by describing it."""
    import traceback

    import make_sound as M

    def note(**changes) -> None:      # its own row, so sounds made side by side never clash
        key = f"add:{job_id}:{add_id}"
        job_status[key] = {**(job_status.get(key) or {}), **changes}

    try:
        jobs_volume.reload()
        job = job_status.get(job_id, {})
        backend = M.TestTones() if job.get("test_tones") else M.Cloud()
        clip, cost = M.add_sound(JOBS / job_id / f"input{job.get('ext', '.mp4')}", JOBS / job_id / "out",
                                 backend, add_id, ask["prompt"], ask["time"], ask["seconds"])
        jobs_volume.commit()
        note(state="done", dollars=cost["dollars"], from_library=cost["from_library"])
        _update(job_id, added_sounds=int(job_status.get(job_id, {}).get("added_sounds", 0)) + 1,
                added_cost=round(float(job_status.get(job_id, {}).get("added_cost", 0)) + cost["dollars"], 4))
        return {"made": True, **cost}
    except Exception as exc:
        traceback.print_exc()
        note(state="failed", error="We couldn't make that sound. Please try again, or describe it differently.")
        return {"made": False, "error": str(exc)[:300]}


# --- What people change in the editor ----------------------------------------------

@app.function(image=image, volumes={"/tuning": modal.Volume.from_name(TUNING_VOLUME, create_if_missing=True)},
              timeout=600)
def export_edits() -> list[dict]:
    """Every saved editing-session record. `python analyze_edits.py` calls this, and so
    does the /insights page. Records hold no video, no audio and nothing about people."""
    from soundlib import insights

    return insights.read_records(Path("/tuning"))


class _Adds:
    """How the sounds added in the editor are coming along, kept with the job statuses."""

    def get(self, job_id: str):
        ids = job_status.get(f"adds:{job_id}") or []
        old = ids if isinstance(ids, dict) else {}       # the first version kept all rows together
        return {i: row for i in ids if (row := job_status.get(f"add:{job_id}:{i}") or old.get(i))}

    def __setitem__(self, job_id: str, value: dict) -> None:
        known = job_status.get(f"adds:{job_id}") or []
        if isinstance(known, dict):                       # move first-version rows to their own keys
            for add_id, row in known.items():
                job_status[f"add:{job_id}:{add_id}"] = row
            known = list(known)
        for add_id, row in value.items():
            if add_id not in known:           # rows already there belong to the machine making them
                job_status[f"add:{job_id}:{add_id}"] = row
        job_status[f"adds:{job_id}"] = list(dict.fromkeys([*known, *value]))


def cost_summary(cost: dict) -> dict:
    """What one video cost, in credits (1 credit = 1 US cent), split the way the owner reads
    it: planning (Claude), music, sound effects. From the "cost" part of a job's report."""
    rate = C.GPU_DOLLARS_PER_SECOND
    music = sum(float(c.get("dollars", 0)) for c in cost.get("api_calls", []))
    effects = 0.0
    for call in cost.get("gpu_calls", []):
        dollars = (call.get("gpu_seconds", 0) + call.get("load_seconds", 0)) * rate.get(call.get("gpu"), 0)
        if "music" in str(call.get("what", "")).lower():
            music += dollars
        else:
            effects += dollars
    planning = float(cost.get("claude_dollars", 0) or 0)
    total = float(cost.get("total_dollars", planning + music + effects) or 0)
    credits = lambda dollars: round(dollars * 100, 1)
    return {"credits": credits(total), "planning": credits(planning), "music": credits(music),
            "effects": credits(effects), "other": max(0.0, credits(total - planning - music - effects)),
            "gpu_seconds": cost.get("gpu_seconds", 0)}


# --- The page and its API ----------------------------------------------------------

@app.function(image=image, volumes={str(JOBS): jobs_volume}, timeout=900, scaledown_window=300,
              secrets=[modal.Secret.from_name(SITE_PASSWORD_SECRET)])
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="ai-sound-designer")
def web():
    import secrets
    import shutil

    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    from soundlib import media

    api = FastAPI(title="AI Sound Designer", docs_url=None, redoc_url=None)
    page = next(p for p in (Path("/root/web/index.html"), ROOT / "web" / "index.html")
                if p.exists()).read_text()
    blocked = {c.upper() for c in C.WEB_BLOCKED_COUNTRIES}

    import os

    from soundlib import gate
    # --- Whose video is it. A video made while signed in belongs to that account: only its
    # maker (and the site owner) can open it, its files and its editor. Videos from before
    # this was recorded have no maker and stay open to any signed-in person with the link.
    def account_of(request: Request) -> str:
        person = getattr(request.state, "person", None) or {}
        return str(person.get("uid") or person.get("email") or "").strip().lower()

    def is_site_owner(request: Request) -> bool:
        email = (getattr(request.state, "person", None) or {}).get("email", "").strip().lower()
        return bool(email) and hashlib.sha256(email.encode()).hexdigest() in {
            str(h).lower() for h in C.WEB_ADMIN_EMAIL_SHA256}

    @api.middleware("http")     # added before the gate, so it runs after the gate knows who this is
    async def own_videos_only(request: Request, call_next):
        parts = request.url.path.split("/")
        if len(parts) >= 4 and parts[1] == "api" and parts[2] in ("job", "edit"):
            maker = (await job_status.get.aio(parts[3]) or {}).get("owner")
            if maker and maker != account_of(request) and not is_site_owner(request):
                return JSONResponse({"detail": "That video wasn't found."}, status_code=404)
        return await call_next(request)

    gate.attach(api, os.environ.get("SITE_PASSWORD", ""), str(C.WEB_LOGIN_URL or ""), Path("/root/web"),
                supabase_url=str(C.WEB_SUPABASE_URL or ""), supabase_anon_key=str(C.WEB_SUPABASE_ANON_KEY or ""))

    def visitor_ip(request: Request) -> str | None:
        forwarded = request.headers.get("x-forwarded-for", "")
        return forwarded.split(",")[0].strip() or (request.client.host if request.client else None)

    @api.get("/", response_class=HTMLResponse)
    @api.get("/j/{job_id}", response_class=HTMLResponse)
    @api.get("/videos", response_class=HTMLResponse)
    def home(job_id: str | None = None):
        return page

    @api.get("/api/config")
    def config(request: Request):
        country = _country_of(visitor_ip(request))
        return {"min_seconds": C.WEB_MIN_VIDEO_SECONDS, "max_seconds": C.WEB_MAX_VIDEO_SECONDS,
                "max_mb": C.WEB_MAX_UPLOAD_MB, "keep_days": C.WEB_KEEP_DAYS, "blocked": sorted(blocked),
                "your_country": country, "you_are_blocked": country in blocked,
                "left_today": max(0, int(C.WEB_DAILY_JOB_LIMIT) - job_status.get(f"count:{_today()}", 0))}

    @api.post("/api/upload")
    def upload(request: Request, video: UploadFile = File(...), country: str = Form(...),
               agree: str = Form(""), vibe: str = Form(""), parts: str = Form("both"),
               test_tones: str = Form(""), detail: str = Form("balanced")):
        country = country.strip().upper()
        if detail not in ("minimal", "balanced", "rich"):
            detail = "balanced"
        ip_country = _country_of(visitor_ip(request))
        if agree != "yes":
            raise HTTPException(400, "Please tick the box to accept the terms.")
        if country in blocked or ip_country in blocked:
            raise HTTPException(403, "Sorry, this service isn't available in the EU, the UK or "
                                     "South Korea, because of the sound model's licence.")
        if len(country) != 2:
            raise HTTPException(400, "Please choose the country you're in.")
        if parts not in ("both", "effects", "music"):
            raise HTTPException(400, "Please choose what sound you want.")
        ext = Path(video.filename or "video.mp4").suffix.lower()
        if ext not in VIDEO_TYPES:
            raise HTTPException(400, "Please upload a video file (MP4, MOV, M4V, MKV or WEBM).")
        today = f"count:{_today()}"
        if job_status.get(today, 0) >= int(C.WEB_DAILY_JOB_LIMIT):
            raise HTTPException(429, "Today's limit of videos has been reached. Please try "
                                     "again tomorrow.")

        job_id = secrets.token_urlsafe(12).replace("-", "a").replace("_", "b")
        folder = JOBS / job_id
        folder.mkdir(parents=True)
        target = folder / f"input{ext}"
        limit = int(C.WEB_MAX_UPLOAD_MB) * 1024 * 1024
        with target.open("wb") as out:
            shutil.copyfileobj(video.file, out, length=1024 * 1024)
        if target.stat().st_size > limit:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(413, f"That file is too big (the limit is {C.WEB_MAX_UPLOAD_MB} MB).")
        try:
            info = media.probe(target)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, "That file couldn't be read as a video.")
        seconds = info["duration"]
        if not float(C.WEB_MIN_VIDEO_SECONDS) <= seconds <= float(C.WEB_MAX_VIDEO_SECONDS):
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, f"Your video is {seconds:.0f} seconds long. For now it needs "
                                     f"to be between {C.WEB_MIN_VIDEO_SECONDS} and "
                                     f"{C.WEB_MAX_VIDEO_SECONDS} seconds.")

        job = {"id": job_id, "state": "queued", "step": "Waiting to start", "step_number": 0,
               "created": time.time(), "name": Path(video.filename or "video").stem[:80],
               "ext": ext, "seconds": round(seconds, 1), "vibe": vibe.strip()[:120],
               "parts": parts, "detail": detail, "country": country, "ip_country": ip_country}
        maker = account_of(request)
        if maker:
            job["owner"] = maker
            job_status[f"mine:{maker}"] = (job_status.get(f"mine:{maker}") or [])[-199:] + [job_id]
        (folder / "job.json").write_text(json.dumps(job, indent=2))
        jobs_volume.commit()
        job_status[job_id] = job
        job_status[today] = job_status.get(today, 0) + 1
        run_job.spawn(job_id, test_tones=test_tones == "yes")
        return {"id": job_id}

    def cost_of(job: dict) -> dict | None:
        """The cost split of a finished job; worked out from its report for older jobs."""
        if job.get("cost_parts"):
            return job["cost_parts"]
        report = JOBS / job["id"] / "report.json"
        try:
            if not report.exists():
                jobs_volume.reload()
            parts = cost_summary(json.loads(report.read_text())["cost"])
        except Exception:
            return {"credits": round(float(job["cost"]) * 100, 1)} if job.get("cost") is not None else None
        job_status[job["id"]] = job | {"cost_parts": parts}
        return parts

    def with_added(parts: dict | None, job: dict) -> dict | None:
        """The cost with the sounds added in the editor afterwards counted in."""
        extra = round(float(job.get("added_cost") or 0) * 100, 1)
        if not parts or not job.get("added_sounds"):
            return parts
        return parts | {"credits": round(parts["credits"] + extra, 1), "added": extra,
                        "added_sounds": int(job["added_sounds"])}

    @api.get("/api/my-videos")
    def my_videos(request: Request):
        """The signed-in person's videos that still exist, newest first."""
        maker = account_of(request)
        ids = (job_status.get(f"mine:{maker}") or []) if maker else []
        found = [job for job in (job_status.get(i) for i in ids) if job]
        if maker and len(found) != len(ids):      # some were deleted after web_keep_days
            job_status[f"mine:{maker}"] = [job["id"] for job in found]
        keep = float(C.WEB_KEEP_DAYS) * 86400
        public = ("id", "state", "created", "name", "seconds", "vibe", "parts", "test_tones")
        rows = [{k: job[k] for k in public if k in job} | {
            "days_left": max(0, -int(-(job["created"] + keep - time.time()) // 86400))} for job in reversed(found)]
        if is_site_owner(request):
            for row, job in zip(rows, reversed(found)):
                if job.get("cost") is not None:
                    row["credits"] = round(float(job["cost"]) * 100, 1)
        return JSONResponse({"videos": rows, "accounts": bool(maker), "keep_days": C.WEB_KEEP_DAYS},
                            headers={"Cache-Control": "no-store"})

    @api.get("/api/job/{job_id}")
    def status(request: Request, job_id: str):
        job = job_status.get(job_id)
        if not job:
            raise HTTPException(404, "That job wasn't found. Jobs are kept for "
                                     f"{C.WEB_KEEP_DAYS} days.")
        public = ("id", "state", "step", "step_number", "created", "started", "name", "seconds",
                  "vibe", "parts", "error", "wall_seconds", "test_tones")
        reply = {k: job[k] for k in public if k in job} | {"now": time.time()}
        if job.get("state") == "done" and is_site_owner(request):   # costs: for the owner's eyes only
            reply["cost"] = with_added(cost_of(job), job)
        return reply

    @api.get("/api/job/{job_id}/{name}")
    def download(job_id: str, name: str, download: int = 0):
        job = job_status.get(job_id)
        if not job or job.get("state") != "done" or name not in DOWNLOADS:
            raise HTTPException(404, "Not ready.")
        path = JOBS / job_id / "out" / name
        if not path.exists():
            try:
                jobs_volume.reload()
            except Exception:
                pass   # another download is open in this machine; the file may still appear
        if not path.exists():
            raise HTTPException(404, "File not found. Please try again in a moment.")
        nice, kind = DOWNLOADS[name]
        filename = f"{job.get('name', 'video')}-{nice}"
        return FileResponse(path, media_type=kind, filename=filename if download else None,
                            content_disposition_type="attachment" if download else "inline")

    # The editor (fine-tune the sound after it is made): pages and API in soundlib/editor_api.py
    from soundlib import editor_api

    tuning_volume = modal.Volume.from_name(TUNING_VOLUME, create_if_missing=True)

    def refresh() -> None:
        try:
            jobs_volume.reload()
        except Exception:
            pass   # a download is open in this machine; what is on disk is used

    def keep_tuning_record(path: str, record: dict) -> None:
        import io

        with tuning_volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(json.dumps(record).encode()), path)

    editor_api.attach(api, jobs=JOBS, get_job=job_status.get, refresh=refresh,
                      commit=jobs_volume.commit, keep_tuning_record=keep_tuning_record,
                      downloads=DOWNLOADS, web_folder=Path("/root/web"),
                      adds=_Adds(), start_add=lambda job, add, ask: add_sound.spawn(job, add, ask))

    # Insights: what people change in the editor, for the owner only.
    import hashlib

    from fastapi.responses import Response

    from soundlib import insights

    admins = {str(h).lower() for h in C.WEB_ADMIN_EMAIL_SHA256}
    open_site = not os.environ.get("SITE_PASSWORD")
    accounts = bool(str(C.WEB_SUPABASE_URL or "").strip() and str(C.WEB_SUPABASE_ANON_KEY or "").strip())

    def owner_only(request: Request) -> None:
        """With accounts on, only the owner's addresses get in (see web_admin_email_sha256)."""
        if open_site or not accounts:
            return
        email = (getattr(request.state, "person", None) or {}).get("email", "").strip().lower()
        if not email or hashlib.sha256(email.encode()).hexdigest() not in admins:
            raise HTTPException(404, "Not found.")

    def studied() -> tuple[list[dict], dict]:
        records = export_edits.remote()
        return records, insights.study(records)

    @api.get("/insights", response_class=HTMLResponse)
    def insights_page(request: Request):
        owner_only(request)
        return HTMLResponse(insights.report_html(studied()[1]), headers={"Cache-Control": "no-store"})

    @api.get("/api/insights/{name}")
    def insights_file(request: Request, name: str):
        owner_only(request)
        records, result = studied()
        day = _today()
        if name == "download":
            return Response(insights.bundle(records, result), media_type="application/zip", headers={
                "Content-Disposition": f'attachment; filename="editor-edits-{day}.zip"'})
        if name in ("clips.csv", "jobs.csv"):
            return Response(insights.csv_text(result[name[:-5] + "_rows"]), media_type="text/csv", headers={
                "Content-Disposition": f'attachment; filename="editor-{name[:-4]}-{day}.csv"'})
        raise HTTPException(404, "Not found.")

    return api
