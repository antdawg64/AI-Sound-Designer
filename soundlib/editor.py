"""The sound editor: what it needs from a finished job, and what it does with edits.

After the AI has made the sound, the web page offers a timeline where every effect is
its own clip that can be moved, muted, turned up or down, or removed, with one volume
for all the effects and one for the music.

  build()          - called by make_sound: saves each effect as its own small file in
                     <out>/editor/, plus a list of them (editor.json)
  clean_state()    - tidies an edit sent by the browser (never trust the browser)
  render()         - makes a new final.mp4 / sfx.wav / music.wav from an edit
  tuning_record()  - one self-contained record of what a person changed, for
                     analyze_edits.py to learn better defaults from

An "edit" (the state) looks like:
    {"effects_db": 0, "music_db": 0, "effects_muted": false, "music_muted": false,
     "clips": {"bg0": {"start": 1.25, "gain_db": -3, "muted": false, "deleted": false}}}
A clip that is not listed is unchanged.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from . import config as C
from . import mixing
from .media import read_audio, run, to_stereo_48k

FOLDER = "editor"            # inside the job's out/ folder
MANIFEST = "editor.json"
STATE = "edit.json"
EDITED = "edited"            # where render() writes, inside out/
GAIN_RANGE_DB = (-40.0, 12.0)
SCHEMA = 1


# --- Saving the pieces when the sound is made --------------------------------------

def _write_flac(path: Path, data: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(path), np.clip(data, -1.0, 1.0), C.SAMPLE_RATE, format="FLAC", subtype="PCM_16")


def _level_db(data: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)))) if len(data) else 0.0
    return round(20 * np.log10(rms), 1) if rms > 1e-7 else -120.0


def build(out_dir: Path, *, segments: list[dict], spots: list[dict], layers: list[np.ndarray],
          music: np.ndarray, voice_wav: Path | None, shots: list[dict], duration: float,
          plan: dict, caption: str, parts: str, work: Path) -> dict:
    """Save everything the editor needs next to the finished files. Returns the manifest.

    `layers` are the full-length effect layers from mixing.effects_layers, in the same
    order as segments + spots.
    """
    sr = C.SAMPLE_RATE
    folder = out_dir / FOLDER
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    rank = {"quiet": 0, "medium": 1, "loud": 2}
    loudness_of = {s["shot"]: max((fx["loudness"] for fx in s["effects"]), key=rank.get, default="medium")
                   for s in plan.get("shots", [])}
    moments = {round(m["time"], 2): m for m in plan.get("moments", [])}

    clips = []
    for item, layer in zip([*segments, *spots], layers):
        heard = np.flatnonzero(np.abs(layer).max(axis=1) > 0)
        if not len(heard):
            continue
        a, b = int(heard[0]), int(heard[-1]) + 1
        is_moment = "time" in item
        clip_id = f"m{item['index'] - 1000}" if is_moment else f"bg{item['index']}"
        _write_flac(folder / f"{clip_id}.flac", layer[a:b])
        clip = {"id": clip_id, "kind": "moment" if is_moment else "background",
                "label": item["prompt"], "start": round(a / sr, 6),
                "duration": round((b - a) / sr, 6), "file": f"{clip_id}.flac",
                "level_db": _level_db(layer[a:b])}
        if is_moment:
            clip["time"] = item["time"]
            clip["loudness"] = moments.get(round(item["time"], 2), {}).get("loudness", "medium")
        else:
            clip["shots"] = item.get("shots") or [item["shot"]]
            clip["loudness"] = max((loudness_of.get(n, "medium") for n in clip["shots"]), key=rank.get)
        clips.append(clip)

    # What the browser plays for the music. With a voice, the dip under the voice is
    # already in it (that dip depends only on the voice, so it stays right at any volume).
    has_music, has_voice = parts != "effects", voice_wav is not None
    preview_music = music
    if has_voice:
        shutil.copyfile(voice_wav, work / "editor_voice.wav")
        voice, vsr = read_audio(voice_wav)
        _write_flac(folder / "voice.flac", to_stereo_48k(voice, vsr))
        if has_music:
            mixing.save_stem(work / "editor_music.wav", music)
            run(["ffmpeg", "-y", "-v", "error", "-i", work / "editor_music.wav", "-i", voice_wav,
                 "-filter_complex", f"[0:a][1:a]sidechaincompress={C.DUCK}[out]", "-map", "[out]",
                 "-ar", C.SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_f32le", work / "editor_ducked.wav"])
            preview_music = mixing.load_stereo(work / "editor_ducked.wav")
    if has_music:
        _write_flac(folder / "music.flac", preview_music)

    # The finished mix is brought to one overall loudness; the browser needs the same
    # push so the preview sounds as loud as the download.
    total = int(round(duration * sr))
    together = np.zeros((total, 2), dtype=np.float32)
    for layer in layers:
        together += layer[:total]
    if has_music:
        together[: len(preview_music)] += preview_music[:total]
    if has_voice:
        v = to_stereo_48k(voice, vsr)[:total]
        together[: len(v)] += v
    preview_gain = mixing.loudness_gain(together, C.TARGET_LUFS, C.TARGET_TRUE_PEAK_DB)

    manifest = {
        "schema": SCHEMA, "duration": round(duration, 4), "parts": parts,
        "has_music": has_music, "has_voice": has_voice,
        "music": {"file": "music.flac", "label": caption} if has_music else None,
        "voice": {"file": "voice.flac"} if has_voice else None,
        "preview_gain_db": round(20 * float(np.log10(max(preview_gain, 1e-6))), 2),
        "shots": [{"start": s["start"], "end": s["end"]} for s in shots],
        "summary": plan.get("summary", ""),
        "clips": clips,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(out_dir: Path) -> dict | None:
    path = out_dir / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


# --- Edits from the browser -----------------------------------------------------------

def _number(value, low: float, high: float, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(value) else float(min(max(value, low), high))


def clean_state(state: dict | None, manifest: dict) -> dict:
    """An edit with every value checked and kept in range. Unchanged clips are left out."""
    state = state if isinstance(state, dict) else {}
    given = state.get("clips") if isinstance(state.get("clips"), dict) else {}
    lo, hi = GAIN_RANGE_DB
    clips = {}
    for clip in manifest["clips"]:
        mine = given.get(clip["id"])
        if not isinstance(mine, dict):
            continue
        entry = {
            "start": round(_number(mine.get("start"), 0.1 - clip["duration"],
                                   manifest["duration"] - 0.1, clip["start"]), 6),
            "gain_db": round(_number(mine.get("gain_db"), lo, hi, 0.0), 2),
            "muted": bool(mine.get("muted")), "deleted": bool(mine.get("deleted")),
        }
        if (abs(entry["start"] - clip["start"]) > 0.0005 or entry["gain_db"] or entry["muted"]
                or entry["deleted"]):
            clips[clip["id"]] = entry
    return {
        "effects_db": round(_number(state.get("effects_db"), lo, hi, 0.0), 2),
        "music_db": round(_number(state.get("music_db"), lo, hi, 0.0), 2),
        "effects_muted": bool(state.get("effects_muted")),
        "music_muted": bool(state.get("music_muted")),
        "clips": clips,
    }


def is_changed(state: dict) -> bool:
    return bool(state["clips"] or state["effects_db"] or state["music_db"]
                or state["effects_muted"] or state["music_muted"])


def load_state(out_dir: Path, manifest: dict) -> dict:
    path = out_dir / STATE
    try:
        saved = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        saved = {}
    return clean_state(saved, manifest)


def save_state(out_dir: Path, state: dict) -> None:
    (out_dir / STATE).write_text(json.dumps(state, indent=2))


# --- Making the edited files ----------------------------------------------------------

def effects_track(out_dir: Path, manifest: dict, state: dict) -> np.ndarray:
    """The effects track with the edit applied (clips moved, turned up or down, removed)."""
    sr = C.SAMPLE_RATE
    total = int(round(manifest["duration"] * sr))
    out = np.zeros((total, 2), dtype=np.float32)
    if state["effects_muted"]:
        return out
    for clip in manifest["clips"]:
        edit = state["clips"].get(clip["id"], {})
        if edit.get("deleted") or edit.get("muted"):
            continue
        audio, _ = read_audio(out_dir / FOLDER / clip["file"])
        a = int(round(edit.get("start", clip["start"]) * sr))
        skip = max(0, -a)
        a, b = max(a, 0), min(total, a + len(audio))
        if b > a:
            gain = 10 ** ((edit.get("gain_db", 0.0) + state["effects_db"]) / 20)
            out[a:b] += audio[skip:skip + b - a] * gain
    return out


def render(out_dir: Path, video: Path, state: dict) -> dict:
    """Make edited/final.mp4, edited/sfx.wav and edited/music.wav from an edit."""
    manifest = load_manifest(out_dir)
    if manifest is None:
        raise FileNotFoundError("this job has no editor files")
    state = clean_state(state, manifest)
    duration, sr = manifest["duration"], C.SAMPLE_RATE
    work = Path(tempfile.mkdtemp(prefix="edit_"))
    try:
        sfx = effects_track(out_dir, manifest, state)
        music = mixing.load_stereo(out_dir / "music.wav")
        music = music * (0.0 if state["music_muted"] else 10 ** (state["music_db"] / 20))
        mixing.save_stem(work / "sfx.wav", sfx)
        mixing.save_stem(work / "music.wav", music.astype(np.float32))
        voice = out_dir / FOLDER / "voice.flac" if manifest["has_voice"] else None
        final = work / "final.mp4"
        silent = not manifest["has_voice"] and not (np.any(np.abs(sfx) > 1e-5) or np.any(np.abs(music) > 1e-5))
        if silent:   # everything is muted: nothing to bring to a loudness, so just add silence
            run(["ffmpeg", "-y", "-v", "error", "-i", video, "-f", "lavfi", "-i",
                 f"anullsrc=r={sr}:cl=stereo", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", C.AAC_BITRATE, "-t", f"{duration:.4f}",
                 "-movflags", "+faststart", final])
            measured = {"lufs": float("-inf"), "true_peak_db": float("-inf")}
        else:
            measured = mixing.mix(video, work / "sfx.wav", work / "music.wav", voice, final,
                                  work, duration)
        target = out_dir / EDITED
        target.mkdir(exist_ok=True)
        for name in ("final.mp4", "sfx.wav", "music.wav"):
            shutil.move(str(work / name), str(target / name))
        result = {"rendered": time.time(), "state": state,
                  "lufs": None if not np.isfinite(measured["lufs"]) else round(measured["lufs"], 1)}
        (target / "render.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- Tuning data ----------------------------------------------------------------------

def tuning_record(manifest: dict, state: dict, *, job: dict, report: dict | None,
                  session: str, events: list, renders: int = 0) -> dict:
    """Everything worth learning from one person's editing session, in one record.

    It stands on its own (the job's files are deleted after a few days): what the AI
    made, the settings it used, and what the person changed. No video, no audio, and
    nothing that says who the person is.
    """
    report = report or {}
    clips = []
    for clip in manifest["clips"]:
        edit = state["clips"].get(clip["id"], {})
        start = edit.get("start", clip["start"])
        clips.append({
            "id": clip["id"], "kind": clip["kind"], "label": clip["label"],
            "loudness": clip.get("loudness"), "start": clip["start"],
            "duration": clip["duration"], "level_db": clip.get("level_db"),
            "moved_seconds": round(start - clip["start"], 3),
            "gain_db": edit.get("gain_db", 0.0), "muted": bool(edit.get("muted")),
            "deleted": bool(edit.get("deleted")),
        })
    kept = []
    for event in events[:3000] if isinstance(events, list) else []:
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            kept.append({k: (v if isinstance(v, (int, float, bool)) or v is None else str(v)[:80])
                         for k, v in list(event.items())[:8]})
    return {
        "schema": SCHEMA, "job_id": job.get("id"), "session": session, "saved": time.time(),
        "test_tones": bool(job.get("test_tones") or report.get("test_tones_only")),
        "parts": manifest.get("parts", "both"), "duration": manifest["duration"],
        "has_voice": manifest["has_voice"], "vibe": job.get("vibe") or None,
        "summary": manifest.get("summary", ""),
        "music": (report.get("music") or {}) | {"label": (manifest.get("music") or {}).get("label")},
        "settings": report.get("settings") or C.current(),
        "mix": {k: state[k] for k in ("effects_db", "music_db", "effects_muted", "music_muted")},
        "clips": clips, "events": kept, "renders": int(renders), "changed": is_changed(state),
    }
