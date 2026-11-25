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
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from soundlib import config as C
from soundlib import media, mixing, planning, proxyfix
from soundlib.checks import check_outputs

ROOT = Path(__file__).parent
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}


def say(text: str) -> None:
    print(text, flush=True)


# --- Cloud backend (Modal) -----------------------------------------------------

class Cloud:
    """Talks to the deployed Modal app."""

    def __init__(self) -> None:
        proxyfix.apply()
        import modal

        self.modal = modal
        self.gpu_calls: list[dict] = []
        self.claude: dict = {}

    def deploy(self) -> None:
        import modal_app

        say("Installing the cloud side on Modal (the first time builds the model images, "
            "which can take 10 to 20 minutes)...")
        with self.modal.enable_output():
            modal_app.app.deploy()

    def _lookup(self):
        from modal_app import APP_NAME

        try:
            plan_fn = self.modal.Function.from_name(APP_NAME, "plan_sound")
            plan_fn.hydrate()
        except self.modal.exception.NotFoundError:
            self.deploy()
            plan_fn = self.modal.Function.from_name(APP_NAME, "plan_sound")
        foley = self.modal.Cls.from_name(APP_NAME, "Foley")()
        music = self.modal.Cls.from_name(APP_NAME, "Music")()
        return plan_fn, foley, music

    def connect(self) -> None:
        self.plan_fn, self.foley, self.music = self._lookup()

    def plan(self, frames: list[dict], shots: list[dict], duration: float, speech: list[dict],
             vibe: str | None = None) -> dict:
        payload = [{"shot": f["shot"], "time": f["time"], "jpeg": Path(f["path"]).read_bytes()}
                   for f in frames]
        result = self.plan_fn.remote(payload, shots, duration, speech, vibe)
        self.claude = result["usage"]
        return result["plan"]

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
                               "load_seconds": round(meta["load_seconds"], 2)})
        data, sr = sf.read(io.BytesIO(b"".join(chunks)), dtype="float32", always_2d=True)
        return data, sr

    def effects(self, clip: Path, prompt: str, seg: dict) -> tuple[np.ndarray, int]:
        return self._collect(
            self.foley.generate.remote_gen(
                clip.read_bytes(), prompt, seed=C.SEED + seg["index"], steps=C.FOLEY_STEPS,
                guidance=C.FOLEY_GUIDANCE, neg_prompt=C.FOLEY_NEG_PROMPT),
            f"effects segment {seg['index']}")

    def compose(self, caption: str, bpm: int, seconds: float) -> tuple[np.ndarray, int]:
        return self._collect(
            self.music.generate.remote_gen(caption, bpm, seconds, seed=C.SEED, steps=C.MUSIC_STEPS),
            "music")


# --- Test-tone backend (no AI, no cost) ------------------------------------------

class TestTones:
    """Stands in for the AI models so the rest of the pipeline can be tested for free."""

    gpu_calls: list[dict] = []
    claude: dict = {}

    def connect(self) -> None:
        pass

    def plan(self, frames, shots, duration, speech, vibe=None) -> dict:
        return {
            "summary": "TEST PLAN (no AI was used)",
            "shots": [{"shot": s["index"], "effects": [
                {"start": s["start"], "end": s["end"], "description": "test noise",
                 "loudness": ("quiet", "medium", "loud")[s["index"] % 3]}]} for s in shots],
            "music": {"mood": vibe or "test", "genre": "test tone", "tempo_bpm": 110,
                      "instruments": ["sine wave"], "cut_times": []},
        }

    def effects(self, clip: Path, prompt: str, seg: dict) -> tuple[np.ndarray, int]:
        sr = C.SAMPLE_RATE
        n = int((seg["clip_end"] - seg["clip_start"]) * sr)
        rng = np.random.default_rng(C.SEED + seg["index"])
        noise = rng.standard_normal(n).astype(np.float32)
        noise = np.convolve(noise, np.ones(24, dtype=np.float32) / 24, mode="same")
        swell = 0.6 + 0.4 * np.sin(np.linspace(0, 6 * np.pi, n, dtype=np.float32))
        return (0.4 * noise * swell)[:, None], sr

    def compose(self, caption: str, bpm: int, seconds: float) -> tuple[np.ndarray, int]:
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
                        C.PLAN_MODEL, hashlib.sha256(planning.SYSTEM.encode()).hexdigest())
        path = self.folder / f"{key}.plan.json"
        if path.exists():
            self.hits += 1
            self.backend.claude = {}
            return json.loads(path.read_text())
        self._connect()
        plan = self.backend.plan(frames, shots, duration, speech, vibe)
        path.write_text(json.dumps(plan, indent=2))
        return plan

    def effects(self, clip: Path, prompt: str, seg: dict) -> tuple[np.ndarray, int]:
        key = self._key("fx", hashlib.sha256(clip.read_bytes()).hexdigest(), prompt,
                        C.SEED + seg["index"], C.FOLEY_STEPS, C.FOLEY_GUIDANCE,
                        C.FOLEY_NEG_PROMPT, C.FOLEY_MODEL_SIZE)
        return self._audio(key, lambda: self.backend.effects(clip, prompt, seg))

    def compose(self, caption: str, bpm: int, seconds: float) -> tuple[np.ndarray, int]:
        key = self._key("music", caption, bpm, seconds, C.SEED, C.MUSIC_STEPS, C.MUSIC_MODEL)
        return self._audio(key, lambda: self.backend.compose(caption, bpm, seconds))


# --- The pipeline -----------------------------------------------------------------

def make_sound(video: Path, out_dir: Path, backend, keep_work: bool = False,
               vibe: str | None = None, music_flavour: str = "") -> dict:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="make_sound_"))
    try:
        say(f"\n=== {video.name} ===")
        info = media.probe(video)
        duration = info["duration"]
        say(f"1/6 Reading the video: {duration:.1f}s, {info['width']}x{info['height']}, "
            f"{'has' if info['has_audio'] else 'no'} existing audio")

        shots = media.detect_shots(video, duration)
        speech, voice_wav = [], None
        if info["has_audio"]:
            original = media.extract_audio(video, work / "original.wav", duration)
            speech = media.detect_speech(original)
            if speech:
                data, sr = media.read_audio(original)
                voice = mixing.set_loudness(media.gate_to_speech(data, speech, sr), C.VOICE_LUFS)
                voice_wav = work / "voice.wav"
                media.write_audio(voice_wav, voice)
        say(f"    {len(shots)} shot(s); speech in {len(speech)} place(s)")

        say("2/6 Planning the sound with Claude")
        backend.connect()
        frames = media.extract_keyframes(video, shots, work / "frames")
        plan = planning.clean_plan(backend.plan(frames, shots, duration, speech, vibe), shots, duration)
        say(f"    {plan['summary']}")

        say("3/6 Generating sound effects")
        segments = media.plan_segments(shots, duration)
        for seg in segments:
            seg["prompt"], seg["gain_db"] = planning.segment_prompt(plan, seg)
            clip = media.cut_clip(video, seg, work / f"clip{seg['index']:03d}.mp4")
            say(f"    {seg['start']:.1f}-{seg['end']:.1f}s: {seg['prompt']}")
            audio, sr = backend.effects(clip, seg["prompt"], seg)
            seg["audio"] = media.to_stereo_48k(audio, sr)
        sfx = mixing.stitch_effects(segments, duration)
        sfx = mixing.set_loudness(sfx, C.SFX_LUFS if speech else C.SFX_LUFS_NO_VOICE)

        say("4/6 Composing the music")
        music_plan = plan["music"]
        bpm, miss = planning.pick_tempo(music_plan["tempo_bpm"], music_plan["cut_times"])
        caption = planning.music_caption(music_plan, music_flavour)
        say(f"    {caption} at {bpm} BPM")
        raw, sr = backend.compose(caption, bpm, max(C.MUSIC_MIN_SECONDS, float(np.ceil(duration)) + 1))
        music = mixing.shape_music(raw, sr, duration)
        music = mixing.set_loudness(music, C.MUSIC_LUFS if speech else C.MUSIC_LUFS_NO_VOICE)

        say("5/6 Mixing")
        mixing.save_stem(out_dir / "sfx.wav", sfx)
        mixing.save_stem(out_dir / "music.wav", music)
        mixing.mix(video, out_dir / "sfx.wav", out_dir / "music.wav", voice_wav,
                   out_dir / "final.mp4", work, duration)

        say("6/6 Checking the result")
        result = check_outputs(video, out_dir)
        for check in result["checks"]:
            say(f"    {'PASS' if check['passed'] else 'FAIL'}  {check['name']}: {check['detail']}")

        gpu_seconds = sum(c["gpu_seconds"] + c["load_seconds"] for c in backend.gpu_calls)
        gpu_dollars = sum((c["gpu_seconds"] + c["load_seconds"]) * C.GPU_DOLLARS_PER_SECOND[c["gpu"]]
                          for c in backend.gpu_calls)
        report = {
            "video": str(video), "output_dir": str(out_dir), "duration": duration,
            "test_tones_only": isinstance(getattr(backend, "backend", backend), TestTones),
            "vibe": vibe, "music_flavour": music_flavour, "settings": C.current(),
            "passed": result["passed"], "checks": result["checks"],
            "shots": shots, "speech": speech,
            "segments": [{k: v for k, v in s.items() if k != "audio"} for s in segments],
            "music": {"caption": caption, "bpm": bpm, "claude_bpm": music_plan["tempo_bpm"],
                      "average_beat_to_cut_seconds": miss},
            "cost": {"gpu_seconds": round(gpu_seconds, 1), "gpu_dollars": round(gpu_dollars, 4),
                     "claude_dollars": backend.claude.get("dollars", 0.0),
                     "total_dollars": round(gpu_dollars + backend.claude.get("dollars", 0.0), 4),
                     "gpu_calls": list(backend.gpu_calls), "claude": dict(backend.claude)},
            "wall_seconds": round(time.time() - started, 1),
        }
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        say(f"Done in {report['wall_seconds']:.0f}s. GPU: {report['cost']['gpu_seconds']:.0f}s, "
            f"cost ${report['cost']['total_dollars']:.3f}. Files are in {out_dir}/")
        return report
    finally:
        backend.gpu_calls = []
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
                                   vibe=args.vibe)]
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
