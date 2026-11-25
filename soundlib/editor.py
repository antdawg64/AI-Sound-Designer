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

An "edit" (the state) is the list of pieces on the timeline. A piece plays part of one
of the saved effect files (its source), from `in` to `out` seconds inside that file,
starting at `start` seconds in the video:
    {"effects_db": 0, "music_db": 0, "effects_muted": false, "music_muted": false,
     "pieces": [{"id": "bg1.0", "src": "bg1", "in": 0.0, "out": 2.4, "start": 5.2,
                 "gain_db": 0, "muted": false, "deleted": false}]}
To begin with (initial_pieces) every key moment is one piece and every background clip
is cut into one piece per shot. People can cut pieces further and duplicate them.
Neighbouring pieces overlap by a few milliseconds with matching fades, so an untouched
cut is inaudible and a moved piece never clicks.
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
SCHEMA = 2
EDGE = 0.005              # half the length of the fade at a cut, in seconds
MIN_PIECE = 0.05          # shortest piece, in seconds
MAX_PIECES = 400


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
        "shot_labels": _shot_labels(plan, shots),
        "clips": clips,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def _shot_labels(plan: dict, shots: list[dict]) -> list[str]:
    """What the plan asked for in each shot, worded the way the effects model was told."""
    from . import planning

    labels = []
    for n, shot in enumerate(shots):
        try:
            labels.append(planning.segment_prompt(
                plan, {"shot": n, "shots": [n], "start": shot["start"], "end": shot["end"]})[0])
        except Exception:
            labels.append("")
    return labels


def load_manifest(out_dir: Path) -> dict | None:
    path = out_dir / MANIFEST
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if "shot_labels" not in manifest:      # a job made before clips were split by shot
        plan = out_dir / "plan.json"
        manifest["shot_labels"] = (_shot_labels(json.loads(plan.read_text()), manifest["shots"])
                                   if plan.exists() else [])
    return manifest


# --- Pieces: what is on the timeline ---------------------------------------------------

def initial_pieces(manifest: dict) -> list[dict]:
    """The timeline as the AI made it: each key moment is one piece, and each background
    clip (which can span several shots) is cut into one piece per shot."""
    labels, shots = manifest.get("shot_labels") or [], manifest["shots"]
    pieces = []
    for clip in manifest["clips"]:
        cuts = []
        if clip["kind"] == "background":
            for n in clip.get("shots", [])[1:]:
                at = shots[n]["start"] - clip["start"] if n < len(shots) else -1
                if 0.2 < at < clip["duration"] - 0.2 and (not cuts or at - cuts[-1] > 0.2):
                    cuts.append(round(at, 6))
        edges = [0.0, *cuts, clip["duration"]]
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            label = clip["label"]
            if cuts:
                middle = clip["start"] + (a + b) / 2
                shot = next((n for n in clip["shots"] if n < len(shots)
                             and shots[n]["start"] <= middle < shots[n]["end"]), None)
                if shot is not None and shot < len(labels) and labels[shot]:
                    label = labels[shot]
            pieces.append({"id": f"{clip['id']}.{i}" if cuts else clip["id"], "src": clip["id"],
                           "in": a, "out": b, "start": round(clip["start"] + a, 6),
                           "gain_db": 0.0, "muted": False, "deleted": False, "label": label})
    return pieces


def _label_for(first: list[dict], src: str, middle: float, fallback: str) -> str:
    """The label of the original piece of this source that covers `middle` (seconds into it)."""
    mine = [p for p in first if p["src"] == src]
    for p in mine:
        if p["in"] <= middle < p["out"]:
            return p["label"]
    return mine[-1]["label"] if mine else fallback


# --- Edits from the browser -----------------------------------------------------------

def _number(value, low: float, high: float, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(value) else float(min(max(value, low), high))


def clean_state(state: dict | None, manifest: dict) -> dict:
    """An edit with every value checked and kept in range."""
    state = state if isinstance(state, dict) else {}
    first = initial_pieces(manifest)
    sources = {c["id"]: c for c in manifest["clips"]}
    lo, hi = GAIN_RANGE_DB
    given = state.get("pieces")
    pieces, seen = [], set()
    for n, mine in enumerate(given[:MAX_PIECES] if isinstance(given, list) else first):
        clip = sources.get(mine.get("src")) if isinstance(mine, dict) else None
        if clip is None:
            continue
        a = _number(mine.get("in"), 0.0, clip["duration"] - MIN_PIECE, 0.0)
        b = _number(mine.get("out"), a + MIN_PIECE, clip["duration"], clip["duration"])
        if b - a < MIN_PIECE:
            continue
        piece_id = "".join(c for c in str(mine.get("id", "")) if c.isalnum() or c in "._-")[:40]
        if not piece_id or piece_id in seen:
            piece_id = f"p{n}x{len(seen)}"
        seen.add(piece_id)
        pieces.append({
            "id": piece_id, "src": clip["id"], "in": round(a, 6), "out": round(b, 6),
            "start": round(_number(mine.get("start"), 0.1 - (b - a), manifest["duration"] - 0.1,
                                   clip["start"] + a), 6),
            "gain_db": round(_number(mine.get("gain_db"), lo, hi, 0.0), 2),
            "muted": bool(mine.get("muted")), "deleted": bool(mine.get("deleted")),
            "label": _label_for(first, clip["id"], (a + b) / 2, clip["label"]),
        })
    return {
        "effects_db": round(_number(state.get("effects_db"), lo, hi, 0.0), 2),
        "music_db": round(_number(state.get("music_db"), lo, hi, 0.0), 2),
        "effects_muted": bool(state.get("effects_muted")),
        "music_muted": bool(state.get("music_muted")),
        "pieces": pieces,
    }


def _same(a: dict, b: dict) -> bool:
    return (a["src"] == b["src"] and abs(a["in"] - b["in"]) < 1e-4 and abs(a["out"] - b["out"]) < 1e-4
            and abs(a["start"] - b["start"]) < 5e-4 and a["gain_db"] == b["gain_db"]
            and a["muted"] == b["muted"] and a["deleted"] == b["deleted"])


def is_changed(state: dict, manifest: dict) -> bool:
    first = initial_pieces(manifest)
    return bool(state["effects_db"] or state["music_db"] or state["effects_muted"]
                or state["music_muted"] or len(first) != len(state["pieces"])
                or not all(_same(a, b) for a, b in zip(first, state["pieces"])))


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
    """The effects track with the edit applied (pieces cut, moved, copied, turned up or
    down, removed)."""
    sr = C.SAMPLE_RATE
    total = int(round(manifest["duration"] * sr))
    out = np.zeros((total, 2), dtype=np.float32)
    if state["effects_muted"]:
        return out
    sources = {c["id"]: c for c in manifest["clips"]}
    loaded: dict[str, np.ndarray] = {}
    edge = int(round(EDGE * sr))
    for piece in state["pieces"]:
        if piece["deleted"] or piece["muted"]:
            continue
        if piece["src"] not in loaded:
            loaded[piece["src"]] = read_audio(out_dir / FOLDER / sources[piece["src"]]["file"])[0]
        source = loaded[piece["src"]]
        a, b = int(round(piece["in"] * sr)), min(len(source), int(round(piece["out"] * sr)))
        # A cut inside the file gets a short fade that reaches a little past the cut, so
        # two neighbouring pieces left in place add up to exactly the uncut sound.
        lead = edge if a > 0 else 0
        tail = edge if b < len(source) else 0
        lead, tail = min(lead, a), min(tail, len(source) - b)
        audio = source[a - lead:b + tail].copy()
        if lead:
            audio[:2 * lead] *= np.linspace(0.0, 1.0, 2 * lead, dtype=np.float32)[: len(audio[:2 * lead]), None]
        if tail:
            n = min(2 * tail, len(audio))
            audio[-n:] *= np.linspace(1.0, 0.0, 2 * tail, dtype=np.float32)[-n:, None]
        at = int(round(piece["start"] * sr)) - lead
        skip = max(0, -at)
        at, end = max(at, 0), min(total, at + len(audio))
        if end > at:
            gain = 10 ** ((piece["gain_db"] + state["effects_db"]) / 20)
            out[at:end] += audio[skip:skip + end - at] * gain
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
    first = {p["id"]: p for p in initial_pieces(manifest)}
    sources = {c["id"]: c for c in manifest["clips"]}
    clips = []
    for piece in state["pieces"]:
        clip, was = sources[piece["src"]], first.get(piece["id"])
        whole = any(p["src"] == piece["src"] and abs(p["in"] - piece["in"]) < 1e-4
                    and abs(p["out"] - piece["out"]) < 1e-4 for p in first.values())
        clips.append({
            "id": piece["id"], "src": piece["src"], "kind": clip["kind"], "label": piece["label"],
            # ai = as the AI made it; cut = the person cut it; copy = the person duplicated it
            "origin": "cut" if not whole else "ai" if was is not None else "copy",
            "loudness": clip.get("loudness"), "in": piece["in"], "out": piece["out"],
            "start": round(clip["start"] + piece["in"], 4),
            "duration": round(piece["out"] - piece["in"], 4), "level_db": clip.get("level_db"),
            "moved_seconds": round(piece["start"] - (clip["start"] + piece["in"]), 3),
            "gain_db": piece["gain_db"], "muted": piece["muted"], "deleted": piece["deleted"],
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
        "clips": clips, "events": kept, "renders": int(renders),
        "changed": is_changed(state, manifest),
    }
