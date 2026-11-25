#!/usr/bin/env python3
"""Try the editor on this computer, with no Modal and no cost.

    python make_sound.py some_video.mp4 --dry-run --out outputs/demo
    python scripts/editor_dev_server.py outputs/demo some_video.mp4
    then open http://localhost:8766/edit/demo

Edits are saved in the output folder, and the tuning records in tuning_data/raw/
(the same layout analyze_edits.py reads).
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from soundlib import editor_api  # noqa: E402


def make_app(out_dir: Path, video: Path, records: Path) -> FastAPI:
    jobs = Path(tempfile.mkdtemp(prefix="editor_dev_"))
    (jobs / "demo").mkdir()
    (jobs / "demo" / "out").symlink_to(out_dir.resolve())
    (jobs / "demo" / f"input{video.suffix}").symlink_to(video.resolve())
    if (out_dir / "report.json").exists():
        shutil.copyfile(out_dir / "report.json", jobs / "demo" / "report.json")
    job = {"id": "demo", "state": "done", "name": video.stem, "ext": video.suffix,
           "seconds": 0, "test_tones": True}
    api = FastAPI()

    def keep(path: str, record: dict) -> None:
        target = records / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record))

    @api.get("/api/job/{job_id}/final.mp4")
    def final(job_id: str):
        if job_id != "demo":
            raise HTTPException(404)
        return FileResponse(out_dir / "final.mp4", media_type="video/mp4")

    editor_api.attach(api, jobs=jobs, get_job=lambda i: job if i == "demo" else None,
                      refresh=lambda: None, commit=lambda: None, keep_tuning_record=keep,
                      downloads={"final.mp4": ("with-sound.mp4", "video/mp4"),
                                 "music.wav": ("music.wav", "audio/wav"),
                                 "sfx.wav": ("sound-effects.wav", "audio/wav")})
    return api


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    app = make_app(Path(sys.argv[1]), Path(sys.argv[2]), ROOT / "tuning_data" / "raw")
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[3]) if len(sys.argv) > 3 else 8766)
