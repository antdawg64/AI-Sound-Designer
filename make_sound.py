#!/usr/bin/env python3
"""Add synced sound effects and original music to a video.

    python make_sound.py path/to/video.mp4

Writes final.mp4 (the video with its new sound), music.wav and sfx.wav into
outputs/<video name>/, then checks the result automatically.

Other ways to run it:
    python make_sound.py --all          every video in test_videos/
    python make_sound.py --deploy       (re)install the cloud side on Modal
    python make_sound.py video.mp4 --versions
        Three different takes (different seeds and music) to pick from, in
        outputs/<video name>/take_1, take_2, take_3.
    python make_sound.py video.mp4 --vibe suspenseful
        Tell the music what feeling to go for.
    python make_sound.py video.mp4 --dry-run
        Test tones instead of AI sound. No Modal, no cost. Only for testing the
        plumbing: the result is NOT a real sound design.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from soundlib import config as C
from soundlib import editor, library, lyria, media, mixing, planning, proxyfix
from soundlib.checks import check_outputs

ROOT = Path(__file__).parent
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
# The Modal app with the AI models (modal_app.py). SOUND_APP names a separate copy, so a
# trial version can be tried without touching the one the live web page uses.
APP_NAME = os.environ.get("SOUND_APP") or "ai-sound-designer"


def say(text: str) -> None:
    print(text, flush=True)


# Work that runs ahead in the background (the music, while the effects are being made)
# keeps its progress lines here, and they are shown when its step comes up, so the
# progress people see still goes 1, 2, 3, 4, 5, 6.
_held = threading.local()


def _takes(seg: dict) -> int:
    """How many takes are made of this sound, to keep the best one."""
    return int(seg.get("takes") or C.FOLEY_CANDIDATES)


# --- Cloud backend (Modal) -----------------------------------------------------

def _judge(seg: dict) -> dict | None:
    """How the best of several takes is picked for this sound (TAKE_JUDGE strict)."""
    if C.TAKE_JUDGE != "strict" or _takes(seg) <= 1:
        return None
    onset = seg["time"] - seg["clip_start"] if "time" in seg and seg.get("role", "main") == "main" else None
    return {"reject": list(C.TAKE_REJECT), "onset": onset}


class Cloud:
    """Talks to the deployed Modal app."""

    def __init__(self) -> None:
        proxyfix.apply()
        import modal

        self.modal = modal
        self.gpu_calls: list[dict] = []
        self.api_calls: list[dict] = []   # paid calls to outside services (Google Lyria)
        self.claude: dict = {}
        self._lock = threading.Lock()    # several calls can be under way at once

    def deploy(self) -> None:
        import modal_app
        import web_app

        say("Installing the cloud side on Modal (the first time builds the model images, "
            "which can take 10 to 20 minutes)...")
        with self.modal.enable_output():
            modal_app.app.deploy()
            try:   # the second effects model; needs the Hugging Face secret (see modal_stable.py)
                import modal_stable

                modal_stable.app.deploy()
            except Exception as exc:
                say(f"Stable Audio was not installed ({exc}). Everything else is unaffected.")
            if os.environ.get("SOUND_APP"):   # a trial copy of the models: leave the live page alone
                return
            web_app.app.deploy()
        say(f"Web page: {web_app.page_url()}")

    def _lookup(self):
        try:
            plan_fn = self.modal.Function.from_name(APP_NAME, "plan_sound")
            plan_fn.hydrate()
        except self.modal.exception.NotFoundError:
            self.deploy()
            plan_fn = self.modal.Function.from_name(APP_NAME, "plan_sound")
        foley = self.modal.Cls.from_name(APP_NAME, "Foley")()
        music = self.modal.Cls.from_name(APP_NAME, "Music")
        return plan_fn, foley, music

    def connect(self) -> None:
        self.plan_fn, self.foley, self.music = self._lookup()

    def plan(self, frames: list[dict], shots: list[dict], duration: float, speech: list[dict],
             vibe: str | None = None) -> dict:
        payload = [{"shot": f["shot"], "time": f["time"], "jpeg": Path(f["path"]).read_bytes()}
                   for f in frames]
        result = self.plan_fn.remote(payload, shots, duration, speech, vibe, C.PLAN_HINT or None)
        self.claude = result["usage"]
        return result["plan"]

    def _add_cost(self, usage: dict) -> None:
        with self._lock:
            self.claude = {**usage, "dollars": round(usage["dollars"] + self.claude.get("dollars", 0.0), 4)}

    def missed(self, frames: list[dict], windows: list[dict], duration: float, plan: dict) -> list[dict]:
        fn = self.modal.Function.from_name(APP_NAME, "plan_missed")
        result = fn.remote(frames, windows, duration, plan, planning.look_model(), int(C.SECOND_LOOK_MAX_EXTRA), C.LOOK_EFFORT or None)
        self._add_cost(result["usage"])
        return result["actions"]

    def pins(self, bursts: list[dict]) -> dict:
        fn = self.modal.Function.from_name(APP_NAME, "plan_pins")
        result = fn.remote(bursts, planning.look_model(), C.LOOK_EFFORT or None)
        self._add_cost(result["usage"])
        return result["pins"]

    def compose_scored(self, caption: str, bpm: int, seconds: float, keyscale: str,
                       sections: list[dict]) -> tuple[np.ndarray, int]:
        return self._collect(
            self.music(model=C.MUSIC_MODEL).generate_scored.remote_gen(
                caption, bpm, seconds, sections, seed=C.SEED, steps=C.MUSIC_STEPS,
                guidance=C.MUSIC_GUIDANCE, keyscale=keyscale),
            "music (scored)")

    def cues(self, frames: list[dict], shots: list[dict], duration: float, plan: dict) -> dict:
        payload = [{"shot": f["shot"], "time": f["time"], "jpeg": Path(f["path"]).read_bytes()}
                   for f in frames]
        fn = self.modal.Function.from_name(APP_NAME, "plan_cues")
        result = fn.remote(payload, shots, duration, plan)
        self._add_cost(result["usage"])
        return result["cues"]

    def _collect(self, stream, what: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        meta, chunks = None, []
        for item in stream:
            if meta is None:
                meta = item
            else:
                chunks.append(item)
        if meta is None:
            raise RuntimeError(f"{what}: the model returned nothing")
        self.gpu_calls.append({"what": what, "gpu": meta["gpu"],
                               "gpu_seconds": round(meta["gpu_seconds"], 2),
                               "load_seconds": round(meta["load_seconds"], 2),
                               **({"match_scores": [round(x, 3) for x in meta["scores"]],
                                   "picked": meta["picked"]} if meta.get("scores") else {})})
        data, sr = sf.read(io.BytesIO(b"".join(chunks)), dtype="float32", always_2d=True)
        return data, sr

    def effects(self, clip: Path, prompt: str, seg: dict) -> tuple[np.ndarray, int]:
        real = library.find(self, prompt, seg)    # a real recording, when one clearly fits
        if real is not None:
            return real
        return self._collect(
            self.foley.generate.remote_gen(
                clip.read_bytes(), prompt, seed=C.SEED + seg["index"], steps=C.FOLEY_STEPS,
                guidance=C.FOLEY_GUIDANCE, neg_prompt=C.FOLEY_NEG_PROMPT,
                candidates=_takes(seg), judge=_judge(seg)),
            f"effects segment {seg['index']}")

    def effects_text(self, asks: list[dict]) -> list[tuple[np.ndarray, int]]:
        """Sounds from text alone (Stable Audio). asks: {prompt, seconds, seed}."""
        import soundfile as sf

        model = self.modal.Cls.from_name(APP_NAME + "-stable", "StableAudio")()
        parts: dict[int, list[bytes]] = {}
        for item in model.generate.remote_gen(asks, steps=int(C.STABLE_STEPS), guidance=float(C.STABLE_GUIDANCE),
                                              neg_prompt=C.STABLE_NEG_PROMPT):
            if "part" in item:
                parts.setdefault(item["i"], []).append(item["part"])
            elif item.get("done"):
                self.gpu_calls.append({"what": f"{len(asks)} effects from text", "gpu": item["gpu"],
                                       "gpu_seconds": round(item["gpu_seconds"], 2),
                                       "load_seconds": round(item["load_seconds"], 2)})
        if len(parts) != len(asks):
            raise RuntimeError(f"effects from text: asked for {len(asks)} sounds, got {len(parts)}")
        return [sf.read(io.BytesIO(b"".join(parts[i])), dtype="float32", always_2d=True)
                for i in range(len(asks))]

    def compose_lyria(self, prompt: str, seconds: float) -> tuple[np.ndarray, int]:
        """Music from Google Lyria (its own small Modal app, lyria_app.py)."""
        import soundfile as sf

        fn = self.modal.Function.from_name("ai-sound-designer-lyria", "compose")
        lyria_settings = {k: v for k, v in C.current().items() if k.startswith("lyria_")}
        stream = fn.remote_gen(prompt, C.LYRIA_NEGATIVE_PROMPT, seconds, C.SEED, C.LYRIA_MODEL, lyria_settings)
        meta = next(stream)
        flac = b"".join(stream)
        self.api_calls.append({"what": "music (Google Lyria)", "model": meta["model"],
                               "clips": meta["clips"], "dollars": meta["dollars"]})
        return sf.read(io.BytesIO(flac), dtype="float32", always_2d=True)

    def compose(self, caption: str, bpm: int, seconds: float, keyscale: str = "") -> tuple[np.ndarray, int]:
        return self._collect(
            self.music(model=C.MUSIC_MODEL).generate.remote_gen(
                caption, bpm, seconds, seed=C.SEED, steps=C.MUSIC_STEPS, guidance=C.MUSIC_GUIDANCE,
                keyscale=keyscale),
            "music")


# --- Test-tone backend (no AI, no cost) ------------------------------------------

class TestTones:
    """Stands in for the AI models so the rest of the pipeline can be tested for free."""

    gpu_calls: list[dict] = []
    api_calls: list[dict] = []
    claude: dict = {}

    def connect(self) -> None:
        pass

    def compose_lyria(self, prompt: str, seconds: float) -> tuple[np.ndarray, int]:
        return self.compose(prompt, 110, seconds)

    def plan(self, frames, shots, duration, speech, vibe=None) -> dict:
        return {
            "summary": "TEST PLAN (no AI was used)",
            "shots": [{"shot": s["index"], "effects": [
                {"start": s["start"], "end": s["end"], "description": "test noise",
                 "source": "test machine", "position": ("left", "center", "right")[s["index"] % 3],
                 "loudness": ("quiet", "medium", "loud")[s["index"] % 3]}]} for s in shots],
            "moments": [{"time": round(duration * f, 2), "description": f"test moment {n}",
                         "loudness": "loud", "position": "right",
                         "layers": [{"role": "whoosh", "description": f"test whoosh {n}"},
                                    {"role": "detail", "description": f"test detail {n}"}][: n % 3]}
                        for n, f in enumerate((0.2, 0.5, 0.8), start=1)],
            "ambience": "test room tone",
            "music": {"mood": vibe or "test", "genre": "test tone", "tempo_bpm": 110,
                      "instruments": ["sine wave"], "cut_times": []},
        }

    def missed(self, frames, windows, duration, plan) -> list[dict]:
        return [{"time": round(duration * f, 2), "description": f"test missed action {n}",
                 "loudness": "quiet", "position": "left"} for n, f in enumerate((0.21, 0.35, 0.65), start=1)]

    def pins(self, bursts) -> dict:
        return {b["index"]: {"visible": True, "time": b["frames"][len(b["frames"]) // 2 + 2]["time"],
                             "what_changes": "test"} for b in bursts if len(b["frames"]) > 4}

    def effects(self, clip: Path, prompt: str, seg: dict) -> tuple[np.ndarray, int]:
        sr = C.SAMPLE_RATE
        n = int((seg["clip_end"] - seg["clip_start"]) * sr)
        rng = np.random.default_rng(C.SEED + seg["index"])
        noise = rng.standard_normal(n).astype(np.float32)
        noise = np.convolve(noise, np.ones(24, dtype=np.float32) / 24, mode="same")
        swell = 0.6 + 0.4 * np.sin(np.linspace(0, 6 * np.pi, n, dtype=np.float32))
        return (0.4 * noise * swell)[:, None], sr

    def effects_text(self, asks: list[dict]) -> list[tuple[np.ndarray, int]]:
        made = []
        for ask in asks:   # a short silence first, like a real model that does not start at once
            seg = {"index": ask["seed"] - C.SEED, "clip_start": 0.0, "clip_end": ask["seconds"] - 0.2}
            noise, sr = self.effects(None, ask["prompt"], seg)
            made.append((np.concatenate([np.zeros((int(0.2 * sr), 1), dtype=np.float32), noise]), sr))
        return made

    def compose_scored(self, caption, bpm, seconds, keyscale, sections):
        return self.compose(caption + str(len(sections)), bpm, seconds)

    def cues(self, frames, shots, duration, plan) -> dict:
        mid = shots[len(shots) // 2]["start"] if len(shots) > 1 else duration / 2
        return {"key": "A minor",
                "sections": [{"start": 0.0, "end": mid, "intensity": 2, "description": "test quiet"},
                             {"start": mid, "end": duration, "intensity": 5, "description": "test loud"}],
                "hits": [{"time": round(duration * 0.75, 2), "strength": "big", "what": "test hit"}]}

    def compose(self, caption: str, bpm: int, seconds: float, keyscale: str = "") -> tuple[np.ndarray, int]:
        sr = C.SAMPLE_RATE
        t = np.arange(int(seconds * sr), dtype=np.float32) / sr
        root = 220.0 * 2 ** ((C.SEED + len(caption)) % 12 / 12)
        chord = sum(np.sin(2 * np.pi * root * f * t) for f in (1.0, 1.26, 1.5)) / 3
        beat = 0.5 + 0.5 * (np.modf(t * bpm / 60.0)[0] < 0.25)
        return (0.3 * chord * beat).astype(np.float32)[:, None], sr


# --- Remembering results -----------------------------------------------------------

class Cached:
    """Wraps a backend and keeps every plan, effects clip and music track on disk.

    Asking again for exactly the same thing (same clip, words, seed and model
    settings) costs nothing. That is what makes tuning cheap: a variant that only
    changes the mix reuses all the generated sound, and one that only changes the
    music reuses all the effects.
    """

    def __init__(self, backend, folder: Path) -> None:
        self.backend, self.folder = backend, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.kind = "tones" if isinstance(backend, TestTones) else "ai"
        self.hits = 0
        self.connected = False

    gpu_calls = property(lambda self: self.backend.gpu_calls,
                         lambda self, v: setattr(self.backend, "gpu_calls", v))
    api_calls = property(lambda self: self.backend.api_calls,
                         lambda self, v: setattr(self.backend, "api_calls", v))
    claude = property(lambda self: self.backend.claude)

    def connect(self) -> None:
        pass  # connects lazily (see _connect): a fully cached run never touches Modal

    def _connect(self) -> None:
        if not self.connected:
            self.backend.connect()
            self.connected = True

    def _key(self, *parts) -> str:
        h = hashlib.sha256(self.kind.encode())
        for part in parts:
            h.update(b"|")
            h.update(part if isinstance(part, bytes) else json.dumps(part, sort_keys=True).encode())
        return h.hexdigest()[:24]

    def _audio(self, key: str, make) -> tuple[np.ndarray, int]:
        import soundfile as sf

        path = self.folder / f"{key}.flac"
        if path.exists():
            self.hits += 1
            return sf.read(path, dtype="float32", always_2d=True)
        self._connect()
        data, sr = make()
        sf.write(path, data, sr, format="FLAC", subtype="PCM_24")
        return data, sr

    def plan(self, frames, shots, duration, speech, vibe=None) -> dict:
        jpegs = b"".join(Path(f["path"]).read_bytes() for f in frames)
        key = self._key("plan", hashlib.sha256(jpegs).hexdigest(), shots, speech, vibe or "",
                        C.PLAN_MODEL, hashlib.sha256(planning.SYSTEM.encode()).hexdigest(),
                        *([C.PLAN_HINT] if C.PLAN_HINT else []))
        path = self.folder / f"{key}.plan.json"
        if path.exists():
            self.hits += 1
            self.backend.claude = {}
            return json.loads(path.read_text())
        self._connect()
        plan = self.backend.plan(frames, shots, duration, speech, vibe)
        path.write_text(json.dumps(plan, indent=2))
        return plan

    def _json(self, key: str, kind: str, make):
        path = self.folder / f"{key}.{kind}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())
        self._connect()
        made = make()
        path.write_text(json.dumps(made, indent=2))
        return made

    def missed(self, frames, windows, duration, plan) -> list[dict]:
        jpegs = hashlib.sha256(b"".join(f["jpeg"] for f in frames)).hexdigest()
        key = self._key("missed", jpegs, windows, plan["shots"], plan["moments"], planning.look_model(), *([C.LOOK_EFFORT] if C.LOOK_EFFORT else []),
                        C.SECOND_LOOK_MAX_EXTRA, hashlib.sha256(planning.LOOK_SYSTEM.encode()).hexdigest())
        return self._json(key, "missed", lambda: self.backend.missed(frames, windows, duration, plan))

    def pins(self, bursts) -> dict:
        jpegs = hashlib.sha256(b"".join(f["jpeg"] for b in bursts for f in b["frames"])).hexdigest()
        key = self._key("pins", jpegs, [(b["index"], b["description"], b["planned"]) for b in bursts],
                        planning.look_model(), *([C.LOOK_EFFORT] if C.LOOK_EFFORT else []),
                        hashlib.sha256(planning.PIN_SYSTEM.encode()).hexdigest())
        pins = self._json(key, "pins", lambda: self.backend.pins(bursts))
        return {int(k): v for k, v in pins.items()}

    def effects(self, clip: Path, prompt: str, seg: dict) -> tuple[np.ndarray, int]:
        key = self._key("fx", hashlib.sha256(clip.read_bytes()).hexdigest(), prompt,
                        C.SEED + seg["index"], C.FOLEY_STEPS, C.FOLEY_GUIDANCE,
                        C.FOLEY_NEG_PROMPT, C.FOLEY_MODEL_SIZE,
                        *([f"best of {_takes(seg)}"] if _takes(seg) > 1 else []),
                        *([_judge(seg)] if _judge(seg) else []), *library.cache_tag(seg))
        # Which real recording was used (or why none) is kept next to the saved sound.
        note = self.folder / f"{key}.library.json"
        if note.exists():
            seg["library"] = json.loads(note.read_text())
        made = self._audio(key, lambda: self.backend.effects(clip, prompt, seg))
        if "library" in seg and not note.exists():
            note.write_text(json.dumps(seg["library"]))
        return made

    def effects_text(self, asks: list[dict]) -> list[tuple[np.ndarray, int]]:
        import soundfile as sf

        keys = [self._key("stable", a["prompt"], a["seconds"], a["seed"], C.STABLE_STEPS,
                          C.STABLE_GUIDANCE, C.STABLE_NEG_PROMPT) for a in asks]
        missing = [i for i, k in enumerate(keys) if not (self.folder / f"{k}.flac").exists()]
        self.hits += len(asks) - len(missing)
        if missing:
            self._connect()
            for i, (data, sr) in zip(missing, self.backend.effects_text([asks[i] for i in missing])):
                sf.write(self.folder / f"{keys[i]}.flac", data, sr, format="FLAC", subtype="PCM_24")
        return [sf.read(self.folder / f"{k}.flac", dtype="float32", always_2d=True) for k in keys]

    def compose_scored(self, caption, bpm, seconds, keyscale, sections):
        key = self._key("scored", caption, bpm, seconds, keyscale, sections, C.SEED, C.MUSIC_STEPS,
                        C.MUSIC_MODEL, C.MUSIC_GUIDANCE)
        return self._audio(key, lambda: self.backend.compose_scored(caption, bpm, seconds, keyscale, sections))

    def compose_lyria(self, prompt: str, seconds: float) -> tuple[np.ndarray, int]:
        key = self._key("lyria", prompt, seconds, C.SEED, C.LYRIA_MODEL, C.LYRIA_NEGATIVE_PROMPT)
        return self._audio(key, lambda: self.backend.compose_lyria(prompt, seconds))

    def cues(self, frames, shots, duration, plan) -> dict:
        jpegs = b"".join(Path(f["path"]).read_bytes() for f in frames)
        key = self._key("cues", hashlib.sha256(jpegs).hexdigest(), shots, plan, C.PLAN_MODEL,
                        hashlib.sha256(planning.CUE_SYSTEM.encode()).hexdigest())
        path = self.folder / f"{key}.cues.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())
        self._connect()
        cues = self.backend.cues(frames, shots, duration, plan)
        path.write_text(json.dumps(cues, indent=2))
        return cues

    def compose(self, caption: str, bpm: int, seconds: float, keyscale: str = "") -> tuple[np.ndarray, int]:
        key = self._key("music", caption, bpm, seconds, C.SEED, C.MUSIC_STEPS, C.MUSIC_MODEL,
                        *([C.MUSIC_GUIDANCE] if "turbo" not in C.MUSIC_MODEL else []),
                        *([keyscale] if keyscale else []))
        return self._audio(key, lambda: self.backend.compose(caption, bpm, seconds, keyscale))


# --- The pipeline -----------------------------------------------------------------

PARTS = ("both", "effects", "music")


def _snap(video: Path, spot: dict, duration: float, pinned: bool = False) -> None:
    """Slide a key moment's sound so the instant it really starts lands on the instant
    of the action (MOMENT_SNAP). The action is looked for in the picture near the
    planned time; the sound's start is found by listening to it."""
    sr, reach = C.SAMPLE_RATE, float(C.MOMENT_PIN_REACH if pinned else C.MOMENT_SNAP_MAX_SECONDS)
    # A moment Claude pinned to its frame is already exact; otherwise look for movement.
    seen = None if pinned else media.action_instant(video, spot["time"], reach)
    target = seen if seen is not None else spot["time"]
    starts = spot["clip_start"] + (mixing.onset_sample(spot["audio"], 12.0) + int(0.015 * sr)) / sr
    shift = target - starts
    spot["snap"] = {"planned": spot["time"], "seen_in_picture": seen, "sound_started": round(starts, 3),
                    "moved_seconds": round(shift, 3) if abs(shift) <= reach else 0.0}
    if abs(shift) > reach or abs(shift) < 0.01:
        return          # too far off to be the same event: leave it where the model put it
    length = spot["clip_end"] - spot["clip_start"]
    spot["clip_start"] = round(spot["clip_start"] + shift, 3)
    if spot["clip_start"] < 0:
        spot["audio"] = spot["audio"][int(round(-spot["clip_start"] * sr)):]
        length += spot["clip_start"]
        spot["clip_start"] = 0.0
    spot["clip_end"] = round(min(duration, spot["clip_start"] + length), 3)
    spot["time"] = round(target, 3)


def text_effects(plan: dict, duration: float, backend, layered: bool) -> tuple[list, list, dict | None]:
    """Every planned sound made from its description alone (sfx_engine stable_audio) and
    lined up with its planned time. Returns (background sounds, key-moment sounds, the
    steady background of the place or None), shaped like the video-driven ones.

    A model that works from text does not know when in its clip the sound should
    start, so each result is lined up by listening to it: a background sound and a
    moment start where the sound really begins, and a whoosh is placed so that its
    loudest point lands on the moment.
    """
    sr = C.SAMPLE_RATE
    rank = {"quiet": 0, "medium": 1, "loud": 2}
    # The same sound carrying on into the next shot is made once, as one long sound.
    sounds: list[dict] = []
    for fx in sorted(({**fx, "shot": s["shot"]} for s in plan["shots"] for fx in s["effects"]),
                     key=lambda f: f["start"]):
        earlier = next((m for m in reversed(sounds) if m["description"].lower() == fx["description"].lower()
                        and abs(m["end"] - fx["start"]) < 0.25), None)
        if earlier:
            earlier["end"] = fx["end"]
            earlier["shots"].append(fx["shot"])
            earlier["loudness"] = max(earlier["loudness"], fx["loudness"], key=rank.get)
        else:
            sounds.append({**fx, "shots": [fx["shot"]]})
    segments, spots, asks = [], [], []

    def ask(item: dict, seconds: float) -> None:
        asks.append({"prompt": item["prompt"], "seconds": round(float(seconds), 2), "seed": C.SEED + item["index"]})

    for k, fx in enumerate(sounds):
        length = min(46.0, max(1.0, fx["end"] - fx["start"]))
        segments.append({"index": k, "shot": fx["shots"][0], "shots": sorted(set(fx["shots"])),
                         "start": fx["start"], "end": fx["end"], "clip_start": fx["start"],
                         "clip_end": round(min(duration, fx["start"] + length), 3),
                         "prompt": fx["description"], "loudness": fx["loudness"], "gain_db": 0.0, "text": True})
        ask(segments[-1], length + 0.5)
    if C.PLAN_DETAIL == "close":
        for i, m in enumerate(plan["moments"]):
            spots.append({"index": 1000 + i, "time": m["time"], "prompt": m["description"],
                          "gain_db": C.LOUDNESS_DB[m["loudness"]]})
            ask(spots[-1], C.STABLE_MOMENT_SECONDS)
            for j, layer in enumerate(m.get("layers") or [] if layered else []):
                spots.append({"index": 2000 + 10 * i + j, "time": m["time"], "moment": i, "role": layer["role"],
                              "prompt": layer["description"],
                              "gain_db": C.LOUDNESS_DB[m["loudness"]] + float(C.MOMENT_LAYER_DB[layer["role"]])})
                ask(spots[-1], C.STABLE_MOMENT_SECONDS)
    ambience = None
    if layered and plan.get("ambience"):
        ambience = {"index": 3000, "ambience": True, "prompt": plan["ambience"],
                    "clip_start": 0.0, "clip_end": float(C.AMBIENCE_SECONDS)}
        ask(ambience, float(C.AMBIENCE_SECONDS) + 0.5)
    for a in asks:
        say(f"    {a['prompt']}")
    made = [media.to_stereo_48k(audio, rate) for audio, rate in backend.effects_text(asks)] if asks else []

    for item, audio in zip([*segments, *spots, *([ambience] if ambience else [])], made):
        role = item.get("role", "main")
        if "time" not in item:                      # background sound or the place: from its real start
            item["audio"] = audio[mixing.onset_sample(audio):]
            if item.get("text"):
                # Start a little early and run a little long, so one sound fades into
                # the next across a cut instead of switching.
                early = min(0.25, item["clip_start"])
                item["clip_start"] = round(item["clip_start"] - early, 3)
                item["clip_end"] = round(min(duration, item["clip_end"] + 0.25,
                                             item["clip_start"] + len(item["audio"]) / sr), 3)
        elif role == "whoosh":                      # its loudest point lands on the moment
            peak = mixing.peak_sample(audio)
            first = max(0, peak - int(float(C.WHOOSH_BEFORE_SECONDS) * sr))
            item["audio"] = audio[first: peak + int(0.1 * sr)]
            item["clip_start"] = round(item["time"] - (peak - first) / sr, 3)
        else:                                       # the moment itself, or a detail just after it
            first = mixing.onset_sample(audio)
            item["audio"] = audio[first: first + int(1.5 * sr)]
            item["clip_start"] = round(item["time"] + (0.05 if role == "detail" else -0.015), 3)
        if "time" in item:
            if item["clip_start"] < 0:              # would start before the video does
                item["audio"] = item["audio"][int(round(-item["clip_start"] * sr)):]
                item["clip_start"] = 0.0
            item["clip_end"] = round(min(duration, item["clip_start"] + len(item["audio"]) / sr), 3)
    spots = [s for s in spots if len(s["audio"]) > int(0.3 * sr) and s["clip_end"] > s["clip_start"]]
    return segments, spots, ambience


def _try_lyria(backend, caption: str, bpm: int, cues: dict | None, duration: float):
    """Music from Google Lyria, or None if Google fails (the run then carries on with
    the ACE-Step music, so a video never fails because of the outside service)."""
    prompt = lyria.story_prompt(caption, bpm, cues["key"] if cues else "",
                                cues["sections"] if cues else [], duration)
    say(f"    Google Lyria ({C.LYRIA_MODEL}): {prompt}")
    try:
        return backend.compose_lyria(prompt, float(duration))
    except Exception as exc:
        say(f"    Google Lyria failed ({str(exc)[:200]}); using the ACE-Step music instead")
        return None


DETAILS = ("minimal", "balanced", "rich")


def add_sound(video: Path, out_dir: Path, backend, add_id: str, prompt: str, time_s: float,
              seconds: float, describe=None, refine: dict | None = None) -> tuple[dict, dict]:
    """One more sound for a finished video, asked for in the editor by describing it.

    It is made the way a key moment is (a real recording when one clearly fits, otherwise
    generated from that piece of the video and the description), then slid so that the
    sound really starts on the chosen instant, and levelled to sit among the video's
    other key moments. Returns (the editor's clip, what it cost).

    prompt: the person's own words. describe(frames, words, seconds, summary, previous,
    tweak) -> {"prompt", "usage"}: Claude looks at the frames around the instant and
    writes the description the sound is really made from (people who know what they
    want to hear are often not good at describing it). Without it the words are used
    as they are. refine: {"of": the sound being remade, "previous": what that one was
    made from, "tweak": how it should change}; a remade sound is always generated, so
    it cannot come back as the same recording.
    """
    sr = C.SAMPLE_RATE
    duration = float(editor.load_manifest(out_dir)["duration"])
    time_s = float(min(max(time_s, 0.0), duration - 0.05))
    start = max(0.0, time_s - float(C.SPOT_BEFORE_SECONDS))
    end = min(duration, start + float(seconds))
    start = max(0.0, min(start, end - 1.0))            # near the very end: still a full second
    spot = {"index": 1200 + int(hashlib.sha256(add_id.encode()).hexdigest(), 16) % 700,
            "time": round(time_s, 3), "prompt": prompt, "gain_db": 0.0,
            "clip_start": round(start, 3), "clip_end": round(end, 3),
            "takes": max(int(C.FOLEY_CANDIDATES), int(C.SPOT_CANDIDATES))}
    words, refine, extra = prompt, refine or {}, 0.0
    if refine.get("tweak") and not describe:
        prompt = f"{refine.get('previous') or words}, {refine['tweak']}"
    if describe:
        try:
            around = [("before", time_s - 0.4), ("chosen", time_s), ("after", time_s + 0.3),
                      ("after", min(time_s + max(0.7, seconds * 0.6), duration - 0.05))]
            frames = media.frames_at(video, [max(0.0, t) for _, t in around], 640)
            nearest = min(frames, key=lambda f: abs(f["time"] - time_s)) if frames else None
            for f in frames:
                f["chosen"] = f is nearest
            summary = (editor.load_manifest(out_dir) or {}).get("summary", "")
            told = describe(frames, words, seconds, summary, refine.get("previous"), refine.get("tweak"))
            prompt, extra = told["prompt"], float(told["usage"].get("dollars", 0.0))
        except Exception as exc:        # the sound is still made, from the words as typed
            print(f"description not improved for {add_id}: {exc}", flush=True)
    spot["prompt"] = prompt
    work = Path(tempfile.mkdtemp(prefix="add_sound_"))
    usual_library = C.SFX_LIBRARY
    try:
        backend.connect()
        clip = media.cut_clip(video, spot, work / "clip.mp4")
        if refine.get("tweak"):
            C.override({"sfx_library": "off"})
        audio, got_sr = backend.effects(clip, prompt, spot)
        spot["audio"] = media.to_stereo_48k(audio, got_sr)
        _snap(video, spot, duration, pinned=True)       # its start lands on the chosen frame
        audio = mixing.clean_clip(mixing.level_clip(spot["audio"]), moment=seconds <= 2.0)
        audio = audio.astype(np.float32).copy()
        fade_in, fade_out = min(int(0.03 * sr), len(audio) // 2), min(int(0.3 * sr), len(audio) // 2)
        if fade_in:
            audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
        if fade_out:
            audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
        now = mixing._loud_part_db(audio.mean(axis=1))
        if now > -90:
            audio = audio * 10 ** ((editor.moment_level_db(out_dir) - now) / 20)
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak > 0.89:
            audio = audio * (0.89 / peak)
        lib = spot.get("library") or {}
        made = editor.save_added(
            out_dir, add_id, audio.astype(np.float32), start=spot["clip_start"], time_s=spot["time"],
            prompt=words, duration=duration, made_as=prompt, seconds=seconds,
            refine_of=refine.get("of"), tweak=refine.get("tweak"),
            recording={k: lib.get(k) for k in ("source", "id", "title", "licence")} if lib.get("used") else None)
    finally:
        C.override({"sfx_library": usual_library})
        shutil.rmtree(work, ignore_errors=True)
    rate = C.GPU_DOLLARS_PER_SECOND
    gpu = sum((c.get("gpu_seconds", 0) + c.get("load_seconds", 0)) * rate.get(c.get("gpu"), 0)
              for c in backend.gpu_calls)
    api = sum(float(c.get("dollars", 0)) for c in backend.api_calls)
    cost = {"dollars": round(gpu + api + extra + float(backend.claude.get("dollars", 0.0)), 4),
            "from_library": bool(lib.get("used"))}
    return made, cost


def make_sound(video: Path, out_dir: Path, backend, keep_work: bool = False,
               vibe: str | None = None, music_flavour: str = "", parts: str = "both",
               detail: str | None = None) -> dict:
    """detail: how many sound effects, "minimal", "balanced" or "rich" (default: the
    sound_detail setting). Each level is a set of setting changes for this one video."""
    detail = detail or C.SOUND_DETAIL
    if detail not in DETAILS:
        raise ValueError(f"detail must be one of {DETAILS}, not {detail!r}")
    level = dict(C.SOUND_DETAIL_LEVELS.get(detail) or {})
    before = {name: getattr(C, name.upper()) for name in level}
    C.override(level)
    try:
        report = _make_sound(video, out_dir, backend, keep_work, vibe, music_flavour, parts, detail)
    finally:
        C.override(before)
    return report


def _make_sound(video: Path, out_dir: Path, backend, keep_work: bool = False,
                vibe: str | None = None, music_flavour: str = "", parts: str = "both",
                detail: str = "balanced") -> dict:
    """parts: "both" (effects and music), "effects" (effects only) or "music" (music only).
    The part that is left out is written as a silent file, so every output always exists."""
    if parts not in PARTS:
        raise ValueError(f"parts must be one of {PARTS}, not {parts!r}")
    global say
    outer = say

    def routed(text: str) -> None:      # lines from background work wait for their step
        held = getattr(_held, "lines", None)
        held.append(text) if held is not None else outer(text)

    say = routed
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="make_sound_"))
    try:
        say(f"\n=== {video.name} ===")
        info = media.probe(video)
        duration = info["duration"]
        silence = np.zeros((int(round(duration * C.SAMPLE_RATE)), 2), dtype=np.float32)
        say(f"1/6 Reading the video: {duration:.1f}s, {info['width']}x{info['height']}, "
            f"{'has' if info['has_audio'] else 'no'} existing audio")

        shots = media.detect_shots(video, duration)
        # Sound already in the video. Asking for only music or only effects keeps all of
        # it (the person's own effects or music stay), and the editor shows it as
        # "Original audio". Asking for both keeps only the talking, as the new effects
        # and music replace the rest. duck_key: what the music dips under (the talking).
        speech, voice_wav, duck_key, kept = [], None, None, None
        if info["has_audio"]:
            original = media.extract_audio(video, work / "original.wav", duration)
            speech = media.detect_speech(original)
            data, sr = media.read_audio(original)
            talking = mixing.set_loudness(media.gate_to_speech(data, speech, sr), C.VOICE_LUFS) if speech else None
            if parts != "both" and mixing.lufs(media.to_stereo_48k(data, sr)) > C.CHECK_SILENT_LUFS:
                kept, voice_wav = "original", work / "voice.wav"
                media.write_audio(voice_wav, mixing.set_loudness(media.to_stereo_48k(data, sr), C.VOICE_LUFS))
                if talking is not None:
                    duck_key = work / "duck_key.wav"
                    media.write_audio(duck_key, talking)
            elif talking is not None:
                kept, voice_wav = "voice", work / "voice.wav"
                media.write_audio(voice_wav, talking)
                duck_key = voice_wav
        # Kept original sound with no talking in it (their own effects or music): the new
        # sound sits just under it instead of far below, as it would under a voice.
        under_original = C.VOICE_LUFS - 6.0 if kept == "original" and not speech else None
        say(f"    {len(shots)} shot(s); speech in {len(speech)} place(s)")

        say("2/6 Planning the sound with Claude")
        backend.connect()
        second_look = C.PLAN_SECOND_LOOK and C.PLAN_DETAIL == "close" and parts != "music"
        dense_job = None
        if second_look and C.RUN_PARALLEL:    # read from the video while Claude makes the plan
            step = 1.0 / float(C.SECOND_LOOK_FPS)
            early = ThreadPoolExecutor(1, thread_name_prefix="frames")
            dense_job = early.submit(media.frames_at, video, list(np.arange(step / 2, duration, step)),
                                     int(C.SECOND_LOOK_LONG_SIDE))
            early.shutdown(wait=False)
        frames = media.extract_keyframes(video, shots, work / "frames")
        plan = planning.clean_plan(backend.plan(frames, shots, duration, speech, vibe), shots, duration)
        say(f"    {plan['summary']}")
        closer = {}
        if second_look:
            say("    Looking again, closely, for actions with no sound yet")
            step = 1.0 / float(C.SECOND_LOOK_FPS)
            dense = dense_job.result() if dense_job else media.frames_at(
                video, list(np.arange(step / 2, duration, step)), int(C.SECOND_LOOK_LONG_SIDE))
            found = planning.add_missed(plan, backend.missed(
                dense, planning.look_windows(shots, duration), duration, plan), duration)
            for m in found:
                say(f"      found at {m['time']:.2f}s: {m['description']}")
            closer["second_look"] = {"frames": len(dense), "found": found}
        if C.MOMENT_TIMING == "frame" and C.PLAN_DETAIL == "close" and parts != "music" and plan["moments"]:
            say("    Finding the exact frame of each key moment")
            fps, around = media.frame_rate(video), float(C.MOMENT_PIN_WINDOW)
            bursts = []
            for i, m in enumerate(plan["moments"]):
                times = list(np.arange(max(0.0, m["time"] - around), min(duration, m["time"] + around), 1.0 / fps))
                bursts.append({"index": i, "description": m["description"], "planned": m["time"],
                               "frames": media.frames_at(video, times, int(C.MOMENT_PIN_LONG_SIDE))})
            moved = planning.apply_pins(plan, backend.pins(bursts), bursts)
            for m in moved:
                say(f"      {m['planned']:.2f}s -> {m['frame']:.3f}s: {m['description']} ({m['what_changes']})")
            closer["frame_timing"] = {"moments": len(bursts), "pinned": moved}

        # The music does not depend on the effects, so with RUN_PARALLEL it is composed while
        # the effects are being made. Its progress lines are held back until step 4.
        def compose_music():
            music_plan = plan["music"]
            bpm, miss = planning.pick_tempo(music_plan["tempo_bpm"], music_plan["cut_times"])
            caption = planning.music_caption(music_plan, music_flavour)
            say(f"    {caption} at {bpm} BPM")
            cues = None
            if C.MUSIC_SCORING != "single" and parts != "effects":
                cues = planning.clean_cues(backend.cues(frames, shots, duration, plan), shots, duration)
                cues["sections"] = planning.settle_sections(cues["sections"], bpm, duration)
                for sec in cues["sections"]:
                    say(f"    {sec['start']:.1f}-{sec['end']:.1f}s, intensity {sec['intensity']}: {sec['description']}")
                for hit in cues["hits"]:
                    say(f"    {hit['strength']} hit at {hit['time']:.1f}s: {hit['what']}")
            if parts == "effects":
                music = silence
            elif C.MUSIC_PROVIDER == "lyria" and (lyria_take := _try_lyria(backend, caption, bpm, cues, duration)):
                music = mixing.shape_music(*lyria_take, duration)
            elif cues and C.MUSIC_SCORING == "sections":
                pieces = []
                for sec in cues["sections"]:
                    length = sec["end"] - sec["start"]
                    part_caption = planning.section_caption(music_plan, music_flavour, sec)
                    raw, sr = backend.compose(part_caption, bpm,
                                              max(C.MUSIC_MIN_SECONDS, float(np.ceil(length)) + 9.0),
                                              cues["key"])
                    pieces.append(media.to_stereo_48k(raw, sr))
                music = mixing.join_sections(pieces, cues["sections"], duration, bpm)
                music = mixing.shape_music(music, C.SAMPLE_RATE, duration)
            elif cues and C.MUSIC_SCORING == "repaint":
                keep, last = float(C.REPAINT_KEEP_SECONDS), len(cues["sections"]) - 1
                pieces = []
                for i, sec in enumerate(cues["sections"]):
                    a = sec["start"] + (keep if i > 0 else 0.0)
                    b = sec["end"] - (keep if i < last else 0.0)
                    if b - a >= 1.0:   # too short to rewrite once the joins are kept: leave as composed
                        pieces.append({"start": round(a, 3), "end": round(b, 3),
                                      "caption": planning.section_caption(music_plan, music_flavour, sec)})
                raw, sr = backend.compose_scored(
                    caption, bpm, max(C.MUSIC_MIN_SECONDS, float(np.ceil(duration)) + 1), cues["key"], pieces)
                music = mixing.shape_music(raw, sr, duration)
            else:
                raw, sr = backend.compose(caption, bpm, max(C.MUSIC_MIN_SECONDS, float(np.ceil(duration)) + 1))
                music = mixing.shape_music(raw, sr, duration)
            if cues:
                music = mixing.shape_to_cues(music, cues)
            if parts != "effects":
                music = mixing.set_loudness(music, under_original or (C.MUSIC_LUFS if voice_wav else C.MUSIC_LUFS_NO_VOICE))
            return music, cues, caption, bpm, miss, music_plan

        def music_ahead():
            _held.lines = []
            try:
                return compose_music(), _held.lines
            finally:
                _held.lines = None

        ahead = None
        if C.RUN_PARALLEL and parts != "effects":
            ahead = ThreadPoolExecutor(1, thread_name_prefix="music")
            ahead_job = ahead.submit(music_ahead)

        say("3/6 Generating sound effects" if parts != "music" else "3/6 Skipping sound effects (music only)")
        from_text = C.SFX_ENGINE == "stable_audio" and parts != "music"
        # Every sound to make is first put on a list, then made side by side on up to
        # SFX_PARALLEL graphics computers. Sounds in the same group are made one after
        # another (stretches of one sound source share the recording chosen for the first).
        waiting: list[tuple] = []

        def queue(target: dict, clip, prompt: str, group=None, then=None) -> None:
            """clip: a function that cuts this sound's piece of the video (run by the worker)."""
            waiting.append((target, clip, prompt, group, then))

        def make_waiting() -> None:
            def one(batch: list[tuple]) -> None:
                for target, clip, prompt, _, _ in batch:
                    audio, sr = backend.effects(clip(), prompt, target)
                    target["audio"] = media.to_stereo_48k(audio, sr)

            batches: dict = {}
            for n, item in enumerate(waiting):
                batches.setdefault(item[3] if item[3] is not None else n, []).append(item)
            workers = max(1, int(C.SFX_PARALLEL), int(C.SFX_WORKERS)) if C.RUN_PARALLEL else 1
            if workers == 1 or len(batches) < 2:
                for batch in batches.values():
                    one(batch)
            else:
                with ThreadPoolExecutor(workers, thread_name_prefix="effects") as pool:
                    list(pool.map(one, batches.values()))
            for _, _, _, _, then in waiting:    # in the order they were asked for
                if then:
                    then()
        apart = C.SFX_SOURCES == "separate" and parts != "music" and not from_text
        segments = media.plan_segments(shots, duration) if parts != "music" and not from_text and not apart else []
        if apart:   # one sound at a time: each source made alone, from where it is heard
            segments = planning.source_spans(plan, duration)
            for seg in segments:
                seg["separate"], seg["gain_db"] = True, 0.0
                clip = lambda seg=seg: media.cut_clip(video, seg, work / f"source{seg['index']:03d}.mp4")
                say(f"    {seg['start']:.1f}-{seg['end']:.1f}s {seg['source']}: {seg['prompt']}")
                queue(seg, clip, seg["prompt"], group=("source", seg["source"]) if seg.get("source") else None)
        spots = []
        layered = C.SFX_LAYERING == "layered"
        text_ambience = None
        if from_text:
            segments, spots, text_ambience = text_effects(plan, duration, backend, layered)
        for seg in segments if not from_text and not apart else []:
            seg["prompt"], seg["gain_db"] = planning.segment_prompt(plan, seg)
            clip = lambda seg=seg: media.cut_clip(video, seg, work / f"clip{seg['index']:03d}.mp4")
            say(f"    {seg['start']:.1f}-{seg['end']:.1f}s: {seg['prompt']}")
            queue(seg, clip, seg["prompt"])
        if C.PLAN_DETAIL == "close" and parts != "music" and not from_text:
            for i, m in enumerate(plan["moments"]):
                spot = {"index": 1000 + i, "time": m["time"], "prompt": m["description"],
                        "gain_db": C.LOUDNESS_DB[m["loudness"]], "position": m.get("position"),
                        "clip_start": round(max(0.0, m["time"] - C.SPOT_BEFORE_SECONDS), 3),
                        "clip_end": round(min(duration, m["time"] + C.SPOT_AFTER_SECONDS), 3)}
                if spot["clip_end"] - spot["clip_start"] < 0.6:
                    continue
                clip = lambda spot=spot, i=i: media.cut_clip(video, spot, work / f"spot{i:03d}.mp4")
                say(f"    moment at {m['time']:.1f}s: {m['description']}")
                spot["takes"] = max(int(C.FOLEY_CANDIDATES), int(C.SPOT_CANDIDATES))
                snap = m.get("pinned") or C.MOMENT_SNAP
                queue(spot, clip, spot["prompt"], then=(
                    lambda spot=spot, pinned=bool(m.get("pinned")): _snap(video, spot, duration, pinned=pinned))
                    if snap else None)
                spots.append(spot)
                # Film-style layering: the supporting sounds of this moment, each made
                # on its own so it is its own clip (a whoosh leading in, a detail after).
                for j, layer in enumerate(m.get("layers") or [] if layered else []):
                    before = float(C.WHOOSH_BEFORE_SECONDS) if layer["role"] == "whoosh" else 0.1
                    after = 0.5 if layer["role"] == "whoosh" else 1.4
                    extra = {"index": 2000 + 10 * i + j, "time": m["time"], "moment": i,
                             "role": layer["role"], "prompt": layer["description"],
                             "gain_db": C.LOUDNESS_DB[m["loudness"]] + float(C.MOMENT_LAYER_DB[layer["role"]]),
                             "clip_start": round(max(0.0, m["time"] - before), 3),
                             "clip_end": round(min(duration, m["time"] + after), 3)}
                    if extra["clip_end"] - extra["clip_start"] < 0.8:
                        continue
                    clip = lambda extra=extra, i=i, j=j: media.cut_clip(video, extra, work / f"spot{i:03d}_{j}.mp4")
                    say(f"      {layer['role']}: {layer['description']}")
                    queue(extra, clip, extra["prompt"])
                    spots.append(extra)
        # The steady background of the place, one stretch made from a held frame of the
        # longest shot (nothing moves, so nothing happens in it) and run under everything.
        ambience = text_ambience
        if layered and parts != "music" and plan.get("ambience") and not from_text:
            longest = max(shots, key=lambda s: s["end"] - s["start"])
            ambience = {"index": 3000, "ambience": True, "prompt": plan["ambience"],
                        "clip_start": 0.0, "clip_end": float(C.AMBIENCE_SECONDS)}
            clip = lambda: media.still_clip(video, (longest["start"] + longest["end"]) / 2,
                                            float(C.AMBIENCE_SECONDS), work / "ambience.mp4")
            say(f"    background of the place: {plan['ambience']}")
            queue(ambience, clip, ambience["prompt"])
        make_waiting()
        # Every effect is kept as its own layer (they add up to the effects track), so
        # the editor can move, mute or remove one effect afterwards.
        layers, sfx = [], silence
        if parts != "music":
            shot_levels = None
            if C.SFX_LEVELING == "by_shot":
                rank = {"quiet": 0, "medium": 1, "loud": 2}
                by_shot = {s["shot"]: max((fx["loudness"] for fx in s["effects"]), key=rank.get, default="medium")
                           for s in plan["shots"]}
                shot_levels = [{"start": s["start"], "end": s["end"],
                                "loudness": by_shot.get(s["index"], "medium")} for s in shots]
            layers = mixing.effects_layers(segments, spots, shot_levels, plan["room"], duration,
                                           under_original or (C.SFX_LUFS if voice_wav else C.SFX_LUFS_NO_VOICE),
                                           ambience=ambience)
            sfx = silence.copy()
            for layer in layers:
                sfx += layer

        say("4/6 Composing the music" if parts != "effects" else "4/6 Skipping the music (effects only)")
        if ahead is not None:
            try:
                (music, cues, caption, bpm, miss, music_plan), lines = ahead_job.result()
            finally:
                ahead.shutdown(wait=False)
            for line in lines:
                say(line)
        else:
            music, cues, caption, bpm, miss, music_plan = compose_music()

        say("5/6 Mixing")
        mixing.save_stem(out_dir / "sfx.wav", sfx)
        mixing.save_stem(out_dir / "music.wav", music)
        # Built first: the editor keeps the layers as they are before the final mix.
        editor.build(out_dir, segments=segments, spots=spots, ambience=ambience, layers=layers, music=music,
                     voice_wav=voice_wav, duck_key=duck_key, kept=kept, shots=shots, duration=duration, plan=plan,
                     caption=caption, parts=parts, work=work)
        mixed = mixing.mix(video, out_dir / "sfx.wav", out_dir / "music.wav", voice_wav,
                           out_dir / "final.mp4", work, duration, duck_key=duck_key,
                           deliver=(out_dir / "sfx.wav", out_dir / "music.wav"))
        if mixed.get("dialogue"):
            say(f"    talking sits {mixed['dialogue']['typical_db']:.0f} dB above the music and effects "
                f"(tightest moment {mixed['dialogue']['tightest_db']:.0f} dB)")

        say("6/6 Checking the result")
        result = check_outputs(video, out_dir, parts, mixed)
        for check in result["checks"]:
            say(f"    {'PASS' if check['passed'] else 'FAIL'}  {check['name']}: {check['detail']}")

        gpu_seconds = sum(c["gpu_seconds"] + c["load_seconds"] for c in backend.gpu_calls)
        gpu_dollars = sum((c["gpu_seconds"] + c["load_seconds"]) * C.GPU_DOLLARS_PER_SECOND[c["gpu"]]
                          for c in backend.gpu_calls)
        api_calls = list(getattr(backend, "api_calls", []))
        api_dollars = sum(c["dollars"] for c in api_calls)
        report = {
            "video": str(video), "output_dir": str(out_dir), "duration": duration,
            "test_tones_only": isinstance(getattr(backend, "backend", backend), TestTones),
            "vibe": vibe, "music_flavour": music_flavour, "parts": parts, "sound_detail": detail,
            "settings": C.current(),
            "passed": result["passed"], "checks": result["checks"],
            "shots": shots, "speech": speech, "kept_audio": kept,
            "segments": [{k: v for k, v in s.items() if k != "audio"} for s in segments],
            "moments": [{k: v for k, v in s.items() if k != "audio"} for s in spots],
            "ambience": {k: v for k, v in ambience.items() if k != "audio"} if ambience else None,
            "mix": {k: v for k, v in mixed.items() if k in ("style", "gain_db", "dialogue", "lufs", "true_peak_db")},
            "library_recordings": library.used([*segments, *spots, ambience]),
            "room": plan["room"], "closer_look": closer,
            "cues": cues,
            "music": {"provider": C.MUSIC_PROVIDER if parts != "effects" else None,
                      "caption": caption, "bpm": bpm, "claude_bpm": music_plan["tempo_bpm"],
                      "average_beat_to_cut_seconds": miss},
            "cost": {"gpu_seconds": round(gpu_seconds, 1), "gpu_dollars": round(gpu_dollars, 4),
                     "claude_dollars": backend.claude.get("dollars", 0.0),
                     "api_dollars": round(api_dollars, 4),
                     "total_dollars": round(gpu_dollars + api_dollars + backend.claude.get("dollars", 0.0), 4),
                     "gpu_calls": list(backend.gpu_calls), "api_calls": api_calls, "claude": dict(backend.claude)},
            "wall_seconds": round(time.time() - started, 1),
        }
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        say(f"Done in {report['wall_seconds']:.0f}s. GPU: {report['cost']['gpu_seconds']:.0f}s, "
            f"cost ${report['cost']['total_dollars']:.3f}. Files are in {out_dir}/")
        return report
    finally:
        say = outer
        backend.gpu_calls = []
        backend.api_calls = []
        if keep_work:
            say(f"Working files kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def make_versions(video: Path, out_dir: Path, backend, vibe: str | None = None) -> list[dict]:
    """Several different takes of the same video, so someone can pick a favourite.

    Each take (see "version_takes" in settings.json) uses its own seed and its own
    twist on the music. The sound plan is made once and shared.
    """
    backend = backend if isinstance(backend, Cached) else Cached(backend, ROOT / "cache")
    base_seed, reports = C.SEED, []
    try:
        for n, take in enumerate(C.VERSION_TAKES, start=1):
            C.override({"seed": base_seed + int(take.get("seed_offset", 0))})
            say(f"\n--- {take.get('name', f'Take {n}')} ---")
            report = make_sound(video, out_dir / f"take_{n}", backend, vibe=vibe,
                                music_flavour=take.get("music_flavour", ""))
            report["take"] = take.get("name", f"Take {n}")
            reports.append(report)
    finally:
        C.override({"seed": base_seed})
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Add AI sound effects and music to a video.")
    parser.add_argument("video", nargs="?", help="path to a video file")
    parser.add_argument("--all", action="store_true", help="run every video in test_videos/")
    parser.add_argument("--out", help="output folder (default: outputs/<video name>/)")
    parser.add_argument("--deploy", action="store_true", help="install the cloud side on Modal")
    parser.add_argument("--dry-run", action="store_true",
                        help="test tones instead of AI sound (free, for testing only)")
    parser.add_argument("--keep-work", action="store_true", help="keep temporary files")
    parser.add_argument("--versions", action="store_true",
                        help="make three different takes to pick from")
    parser.add_argument("--only", choices=["effects", "music"],
                        help="make only the sound effects or only the music")
    parser.add_argument("--vibe", help="the feeling the music should go for, e.g. suspenseful")
    args = parser.parse_args()

    backend = TestTones() if args.dry_run else Cloud()
    if args.deploy:
        Cloud().deploy()
        if not args.video and not args.all:
            return 0
    if args.all:
        videos = sorted(p for p in (ROOT / "test_videos").iterdir() if p.suffix.lower() in VIDEO_TYPES)
        if not videos:
            say("No videos found in test_videos/.")
            return 1
    elif args.video:
        videos = [Path(args.video)]
        if not videos[0].exists():
            say(f"Video not found: {videos[0]}")
            return 1
    else:
        parser.print_help()
        return 1

    reports, failed = [], []
    for video in videos:
        out_dir = Path(args.out) if args.out and len(videos) == 1 else ROOT / "outputs" / video.stem
        try:
            if args.versions:
                made = make_versions(video, out_dir, backend, vibe=args.vibe)
            else:
                made = [make_sound(video, out_dir, backend, keep_work=args.keep_work,
                                   vibe=args.vibe, parts=args.only or "both")]
            reports.extend(made)
            if not all(r["passed"] for r in made):
                failed.append(video.name)
        except Exception as exc:  # keep going so one bad video doesn't hide the others
            say(f"ERROR on {video.name}: {exc}")
            failed.append(video.name)
    if len(reports) > 1:
        say("\n=== Summary ===")
        for r in reports:
            say(f"{'PASS' if r['passed'] else 'FAIL'}  {Path(r['video']).name}: "
                f"{r['cost']['gpu_seconds']:.0f} GPU s, ${r['cost']['total_dollars']:.3f}  "
                f"-> {r['output_dir']}/final.mp4")
    if failed:
        say(f"\nProblems with: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
