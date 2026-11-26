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
  save_added()     - keeps a sound the person asked for in the editor ("Add a sound")
  preview_video()  - a small copy of the video that jumps to any frame at once

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
MUSIC_STEM = "music_stem.flac"
EDITED = "edited"            # where render() writes, inside out/
ADDED = "added"              # inside editor/: one small file per sound the person added
PREVIEW = "preview.mp4"      # inside editor/: the picture the editor shows
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


ROLE_NAMES = {"main": "", "whoosh": "Whoosh: ", "detail": "Detail: "}


def build(out_dir: Path, *, segments: list[dict], spots: list[dict], layers: list[np.ndarray],
          ambience: dict | None = None,
          music: np.ndarray, voice_wav: Path | None, duck_key: Path | None = None,
          kept: str | None = None, shots: list[dict], duration: float,
          plan: dict, caption: str, parts: str, work: Path) -> dict:
    """Save everything the editor needs next to the finished files. Returns the manifest.

    `layers` are the full-length effect layers from mixing.effects_layers, in the same
    order as segments + spots (+ the steady background of the place, when there is one).
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
    for item, layer in zip([*segments, *spots, *([ambience] if ambience else [])], layers):
        heard = np.flatnonzero(np.abs(layer).max(axis=1) > 0)
        if not len(heard):
            continue
        a, b = int(heard[0]), int(heard[-1]) + 1
        is_moment, is_ambience = "time" in item, bool(item.get("ambience"))
        role = item.get("role", "main") if is_moment else None
        if is_ambience:
            clip_id = "amb"
        elif not is_moment:
            clip_id = f"bg{item['index']}"
        elif role == "main":
            clip_id = f"m{item['index'] - 1000}"
        else:   # a supporting layer of key moment N: mNw0 (whoosh), mNd1 (detail)
            clip_id = f"m{item['moment']}{role[0]}{item['index'] % 10}"
        _write_flac(folder / f"{clip_id}.flac", layer[a:b])
        clip = {"id": clip_id, "kind": "ambience" if is_ambience else "moment" if is_moment else "background",
                "label": ROLE_NAMES.get(role, "") + item["prompt"], "start": round(a / sr, 6),
                "duration": round((b - a) / sr, 6), "file": f"{clip_id}.flac",
                "level_db": _level_db(layer[a:b])}
        if (item.get("library") or {}).get("used"):    # a real recording from the library
            clip["recording"] = {k: item["library"].get(k) for k in ("source", "id", "title", "licence")}
        if is_ambience:
            clip["loudness"] = "quiet"
        elif is_moment:
            clip["time"] = item["time"]
            clip["role"] = role
            clip["moment"] = item.get("moment", item["index"] - 1000)
            clip["loudness"] = moments.get(round(item["time"], 2), {}).get("loudness", "medium")
        else:
            clip["shots"] = item.get("shots") or [item["shot"]]
            clip["loudness"] = max((loudness_of.get(n, "medium") for n in clip["shots"]), key=rank.get)
            if item.get("loudness"):     # a sound made on its own knows its own prominence
                clip["loudness"] = item["loudness"]
            if item.get("source"):
                clip["source"] = item["source"]
        clips.append(clip)

    # What the browser plays for the music. With a voice, the dip under the voice is
    # already in it (that dip depends only on the voice, so it stays right at any volume).
    has_music, has_voice = parts != "effects", voice_wav is not None
    preview_music = music
    if has_voice:
        voice, vsr = read_audio(voice_wav)
        _write_flac(folder / "voice.flac", to_stereo_48k(voice, vsr))
        if duck_key is not None:
            key, ksr = read_audio(duck_key)
            _write_flac(folder / "duck_key.flac", to_stereo_48k(key, ksr))
            if has_music and C.MIX_STYLE == "pro":
                preview_music = mixing.preview_music(music, to_stereo_48k(key, ksr))
            elif has_music:
                mixing.save_stem(work / "editor_music.wav", music)
                run(["ffmpeg", "-y", "-v", "error", "-i", work / "editor_music.wav", "-i", duck_key,
                     "-filter_complex", f"[0:a][1:a]sidechaincompress={C.DUCK}[out]", "-map", "[out]",
                     "-ar", C.SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_f32le", work / "editor_ducked.wav"])
                preview_music = mixing.load_stereo(work / "editor_ducked.wav")
    if has_music:
        _write_flac(folder / "music.flac", preview_music)
        # The music as it goes into the mix. render() starts from this, because with the
        # pro mix the music.wav download is the music as it sits in the finished video.
        import soundfile as sf

        sf.write(str(folder / MUSIC_STEM), np.clip(music, -1.0, 1.0), sr, format="FLAC", subtype="PCM_24")

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
        # The sound kept from the video itself: all of it ("original") or only the talking ("voice").
        "voice": {"file": "voice.flac", "kind": kept or "voice",
                  "label": "Original audio" if kept == "original" else "Voice from your video",
                  "duck": duck_key is not None} if has_voice else None,
        "preview_gain_db": round(20 * float(np.log10(max(preview_gain, 1e-6))), 2),
        "mix_style": C.MIX_STYLE,
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
    manifest["clips"] = manifest["clips"] + added_clips(out_dir)
    if "shot_labels" not in manifest:      # a job made before clips were split by shot
        plan = out_dir / "plan.json"
        manifest["shot_labels"] = (_shot_labels(json.loads(plan.read_text()), manifest["shots"])
                                   if plan.exists() else [])
    return manifest


# --- Pieces: what is on the timeline ---------------------------------------------------

def initial_pieces(manifest: dict) -> list[dict]:
    """The timeline as the AI made it: each key moment is one piece, each background
    sound made on its own is one piece, and a mixed background clip (which can span
    several shots) is cut into one piece per shot."""
    labels, shots = manifest.get("shot_labels") or [], manifest["shots"]
    pieces = []
    for clip in manifest["clips"]:
        cuts = []
        # A sound made on its own (one source, like a motor running through several
        # shots) stays in one piece; an old-style stretch with several sounds mixed in
        # it is cut per shot.
        if clip["kind"] == "background" and not clip.get("source"):
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
        "voice_db": round(_number(state.get("voice_db"), lo, hi, 0.0), 2) if manifest["has_voice"] else 0.0,
        "voice_muted": bool(state.get("voice_muted", _voice_off_at_first(manifest))) and manifest["has_voice"],
        "pieces": pieces,
    }


def _voice_off_at_first(manifest: dict) -> bool:
    """The video's own sound is on the timeline but switched off until the person turns it on."""
    return bool((manifest.get("voice") or {}).get("off_at_first"))


def _same(a: dict, b: dict) -> bool:
    return (a["src"] == b["src"] and abs(a["in"] - b["in"]) < 1e-4 and abs(a["out"] - b["out"]) < 1e-4
            and abs(a["start"] - b["start"]) < 5e-4 and a["gain_db"] == b["gain_db"]
            and a["muted"] == b["muted"] and a["deleted"] == b["deleted"])


def is_changed(state: dict, manifest: dict) -> bool:
    first = initial_pieces(manifest)
    return bool(state["effects_db"] or state["music_db"] or state["effects_muted"]
                or state["music_muted"] or state["voice_db"]
                or state["voice_muted"] != (_voice_off_at_first(manifest) and manifest["has_voice"])
                or len(first) != len(state["pieces"])
                or not all(_same(a, b) for a, b in zip(first, state["pieces"])))


def load_state(out_dir: Path, manifest: dict) -> dict:
    path = out_dir / STATE
    try:
        saved = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        saved = {}
    voice = manifest.get("voice") or {}
    if voice.get("off_at_first") and path.exists() and path.stat().st_mtime < voice.get("since", 0):
        saved.pop("voice_muted", None)      # saved before this track existed: it starts switched off
    state = clean_state(saved, manifest)
    # A sound added in the editor whose page was closed before it arrived: put it on the
    # timeline now, where it was asked for.
    placed = {p["src"] for p in state["pieces"]}
    clips = {c["id"]: c for c in manifest["clips"]}
    for new in [p for p in initial_pieces(manifest)
                if p["src"] not in placed and clips[p["src"]].get("added")]:
        replaced = clips[new["src"]].get("refine_of")      # a remake takes the old one's place
        for old in state["pieces"]:
            if replaced and old["src"] == replaced:
                old["deleted"] = True
        state["pieces"].append(new)
    return state


def sources_added(manifest: dict, src: str) -> bool:
    return any(c["id"] == src and c.get("added") for c in manifest["clips"])


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
        stem = out_dir / FOLDER / MUSIC_STEM
        music = mixing.load_stereo(stem if stem.exists() else out_dir / "music.wav")
        music = music * (0.0 if state["music_muted"] else 10 ** (state["music_db"] / 20))
        mixing.save_stem(work / "sfx.wav", sfx)
        mixing.save_stem(work / "music.wav", music.astype(np.float32))
        voice, duck_key = None, None
        voice_on = manifest["has_voice"] and not state["voice_muted"]
        if manifest["has_voice"]:
            # A muted voice track goes in as silence, so the music still dips under the
            # talking exactly as it does in the editor's preview.
            voice = work / "voice.wav"
            kept = mixing.load_stereo(out_dir / FOLDER / "voice.flac")
            kept = kept * (10 ** (state["voice_db"] / 20) if voice_on else 0.0)
            mixing.save_stem(voice, kept.astype(np.float32))
            key = out_dir / FOLDER / "duck_key.flac"
            wants = (manifest.get("voice") or {}).get("duck", True)
            duck_key = key if key.exists() else out_dir / FOLDER / "voice.flac" if wants else None
        final = work / "final.mp4"
        silent = not voice_on and not (np.any(np.abs(sfx) > 1e-5) or np.any(np.abs(music) > 1e-5))
        if silent:   # everything is muted: nothing to bring to a loudness, so just add silence
            run(["ffmpeg", "-y", "-v", "error", "-i", video, "-f", "lavfi", "-i",
                 f"anullsrc=r={sr}:cl=stereo", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", C.AAC_BITRATE, "-t", f"{duration:.4f}",
                 "-movflags", "+faststart", final])
            measured = {"lufs": float("-inf"), "true_peak_db": float("-inf")}
        else:
            # Mixed the way the job was made, whatever today's default is.
            usual = C.MIX_STYLE
            C.override({"mix_style": manifest.get("mix_style", "classic")})
            try:
                measured = mixing.mix(video, work / "sfx.wav", work / "music.wav", voice, final,
                                      work, duration, duck_key=duck_key,
                                      deliver=(work / "sfx.wav", work / "music.wav"))
            finally:
                C.override({"mix_style": usual})
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


# --- The video's own sound, when both effects and music were made ------------------------

def offer_original(out_dir: Path, video: Path) -> bool:
    """When both effects and music were asked for, the video's own sound is replaced (only
    talking is kept). So that it is never simply gone, it is put on the editor's timeline
    as "Original audio", switched off: one press brings it back, at any volume. Done once
    per job, the first time its editor is opened. Returns True if the track was added."""
    from . import media

    path = out_dir / MANIFEST
    manifest = json.loads(path.read_text())
    if manifest.get("has_voice") or manifest.get("original_checked"):
        return False
    manifest["original_checked"] = True
    work = Path(tempfile.mkdtemp(prefix="original_"))
    try:
        if media.probe(video)["has_audio"]:
            wav = media.extract_audio(video, work / "original.wav", manifest["duration"])
            data, sr = read_audio(wav)
            sound = to_stereo_48k(data, sr)
            if mixing.lufs(sound) > C.CHECK_SILENT_LUFS:
                _write_flac(out_dir / FOLDER / "voice.flac", mixing.set_loudness(sound, C.VOICE_LUFS))
                manifest["has_voice"] = True
                manifest["voice"] = {"file": "voice.flac", "kind": "original", "duck": False, "off_at_first": True,
                                     "since": time.time(),
                                     "label": "Original audio (switched off: press Muted to hear it)"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    path.write_text(json.dumps(manifest, indent=2))
    return bool(manifest["has_voice"])


# --- Sounds added in the editor ---------------------------------------------------------

def added_clips(out_dir: Path) -> list[dict]:
    """The sounds the person added, oldest first. Each has its own small file, so two
    being made at once never overwrite each other."""
    clips = []
    for path in sorted((out_dir / FOLDER / ADDED).glob("*.json")):
        try:
            clip = json.loads(path.read_text())
        except ValueError:
            continue
        if (out_dir / FOLDER / clip["file"]).exists():
            clips.append(clip)
    return sorted(clips, key=lambda c: c.get("created", 0))


def moment_level_db(out_dir: Path) -> float:
    """How loud the loud part of a typical key moment of this video is, so an added sound
    sits among them. Without key moments: a little over the background sounds."""
    path = out_dir / MANIFEST
    clips = json.loads(path.read_text())["clips"] if path.exists() else []

    def levels(wanted) -> list[float]:
        found = []
        for clip in clips:
            if wanted(clip) and (out_dir / FOLDER / clip["file"]).exists():
                audio = read_audio(out_dir / FOLDER / clip["file"])[0]
                found.append(mixing._loud_part_db(audio.mean(axis=1)))
        return [v for v in found if v > -90]

    moments = levels(lambda c: c["kind"] == "moment" and c.get("role", "main") == "main")
    if moments:
        return float(np.median(moments))
    beds = levels(lambda c: c["kind"] == "background")
    return float(np.median(beds)) + 3.0 if beds else -30.0


def save_added(out_dir: Path, add_id: str, audio: np.ndarray, *, start: float, time_s: float,
               prompt: str, duration: float, recording: dict | None = None,
               made_as: str | None = None, seconds: float | None = None,
               refine_of: str | None = None, tweak: str | None = None) -> dict:
    """Keep one added sound (stereo, 48 kHz, already levelled) and return its clip, shaped
    like the key moments the AI made. `start`: where `audio` begins in the video."""
    sr = C.SAMPLE_RATE
    if start < 0:
        audio, start = audio[int(round(-start * sr)):], 0.0
    audio = audio[: max(0, int(round((duration - start) * sr)))]
    heard = np.flatnonzero(np.abs(audio).max(axis=1) > 1e-5) if len(audio) else []
    if not len(heard):
        raise ValueError("the sound came out silent")
    a, b = int(heard[0]), int(heard[-1]) + 1
    folder = out_dir / FOLDER
    (folder / ADDED).mkdir(parents=True, exist_ok=True)
    _write_flac(folder / f"{add_id}.flac", audio[a:b])
    clip = {"id": add_id, "kind": "moment", "role": "main", "label": prompt, "added": True,
            "start": round(start + a / sr, 6), "duration": round((b - a) / sr, 6),
            "file": f"{add_id}.flac", "level_db": _level_db(audio[a:b]), "time": round(time_s, 4),
            "loudness": "medium", "created": time.time()}
    if recording:
        clip["recording"] = recording
    # made_as: the description it was really made from; seconds: the length asked for;
    # refine_of / tweak: the sound this one replaces and how it was to change.
    for key, value in (("made_as", made_as), ("seconds", seconds), ("refine_of", refine_of), ("tweak", tweak)):
        if value:
            clip[key] = value
    (folder / ADDED / f"{add_id}.json").write_text(json.dumps(clip, indent=2))
    return clip


# --- The picture the editor shows -------------------------------------------------------

def preview_video(video: Path, out_dir: Path) -> Path:
    """A small, silent copy of the video in which every few frames can be shown on their
    own. The finished video is made to be watched from start to end; jumping around in it
    is slow (worst on phones). In this copy the picture follows the playhead at once."""
    target = out_dir / FOLDER / PREVIEW
    if target.exists():
        return target
    s = int(C.EDITOR_PREVIEW_SHORT_SIDE)
    part = target.with_name(f"preview-{time.time_ns()}.part.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-an",
         "-vf", f"scale='if(gt(iw,ih),-2,min({s},ih))':'if(gt(iw,ih),min({s},ih),-2)'",
         "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
         "-g", int(C.EDITOR_PREVIEW_KEY_EVERY), "-keyint_min", 1, "-sc_threshold", 0, "-bf", 0,
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", part])
    part.replace(target)
    return target


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
            "role": clip.get("role"), "moment": clip.get("moment"),
            # ai = as the AI made it; cut = the person cut it; copy = the person duplicated it;
            # asked = a sound the person asked for in the editor by describing it
            "origin": "asked" if clip.get("added") and whole and was is not None
                      else "cut" if not whole else "ai" if was is not None else "copy",
            "asked_for": bool(clip.get("added")), "made_as": clip.get("made_as"),
            "remade_from": clip.get("refine_of"), "tweak": clip.get("tweak"),
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
        "kept_audio": (manifest.get("voice") or {}).get("kind") if manifest["has_voice"] else None,
        "mix": {k: state[k] for k in ("effects_db", "music_db", "effects_muted", "music_muted",
                                      "voice_db", "voice_muted")},
        "clips": clips, "events": kept, "renders": int(renders),
        "changed": is_changed(state, manifest),
    }
