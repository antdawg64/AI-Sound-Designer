"""Local video and audio work: probing, shots, key frames, speech detection, clips."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config as C

ASSETS = Path(__file__).parent / "assets"


def run(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command and raise a readable error if it fails."""
    proc = subprocess.run([str(c) for c in cmd], capture_output=capture, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1500:]
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd[:6]))} ...\n{tail}")
    return proc


def probe(path: str | Path) -> dict:
    """Duration, size, frame rate and whether the file has an audio track."""
    out = run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path]).stdout
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if video is None:
        raise ValueError(f"{path} has no video track")
    num, den = video.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(video.get("duration") or info["format"]["duration"])
    return {
        "duration": duration,
        "container_duration": float(info["format"]["duration"]),
        "audio_duration": float(audio["duration"]) if audio and audio.get("duration") else None,
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "has_audio": audio is not None,
    }


# --- Shots -----------------------------------------------------------------

def detect_shots(path: str | Path, duration: float) -> list[dict]:
    """Split the video into shots. Always returns at least one shot covering it all."""
    from scenedetect import ContentDetector, detect

    cuts: list[float] = []
    try:
        scenes = detect(str(path), ContentDetector(threshold=C.SHOT_THRESHOLD), show_progress=False)
        cuts = [s[0].get_seconds() for s in scenes][1:]
    except Exception as exc:  # fall back to one shot rather than failing the whole job
        print(f"  shot detection failed ({exc}); treating the video as one shot")
    bounds = [0.0] + [c for c in cuts if 0 < c < duration] + [duration]
    # Merge shots that are too short into the previous one.
    merged = [bounds[0]]
    for b in bounds[1:-1]:
        if b - merged[-1] >= C.MIN_SHOT_SECONDS and duration - b >= C.MIN_SHOT_SECONDS:
            merged.append(b)
    merged.append(duration)
    return [
        {"index": i, "start": round(a, 3), "end": round(b, 3)}
        for i, (a, b) in enumerate(zip(merged[:-1], merged[1:]))
    ]


# --- Key frames ------------------------------------------------------------

def extract_keyframes(path: str | Path, shots: list[dict], out_dir: Path) -> list[dict]:
    """A few small JPEGs per shot for Claude to look at."""
    out_dir.mkdir(parents=True, exist_ok=True)
    close = C.PLAN_DETAIL == "close"
    per_shot = C.CLOSE_FRAMES_PER_SHOT if close else C.FRAMES_PER_SHOT
    max_frames = C.CLOSE_MAX_FRAMES if close else C.MAX_FRAMES
    long_side = C.CLOSE_FRAME_LONG_SIDE if close else C.FRAME_LONG_SIDE
    while per_shot > 1 and per_shot * len(shots) > max_frames:
        per_shot -= 1
    frames, asks = [], []
    scale = f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
    for shot in shots[:max_frames]:
        length = shot["end"] - shot["start"]
        positions = [0.5] if per_shot == 1 else np.linspace(0.08 if close else 0.12, 0.92 if close else 0.88, per_shot)
        for k, pos in enumerate(positions):
            t = shot["start"] + float(pos) * length
            out = out_dir / f"shot{shot['index']:03d}_{k}.jpg"
            asks.append(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                         "-vf", scale, "-q:v", "5", out])
            frames.append({"shot": shot["index"], "time": round(t, 2), "path": str(out)})
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as pool:     # each is its own ffmpeg; side by side saves the wait
        list(pool.map(run, asks))
    return frames


# --- Audio -----------------------------------------------------------------

def extract_audio(path: str | Path, out_wav: Path, duration: float) -> Path:
    """Pull the existing audio out as 48 kHz stereo, padded or trimmed to the video length."""
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", C.SAMPLE_RATE,
         "-af", "apad", "-t", f"{duration:.3f}", "-c:a", "pcm_s16le", out_wav])
    return out_wav


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read audio as float32 with shape (samples, channels)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def write_audio(path: str | Path, data: np.ndarray, sr: int = C.SAMPLE_RATE) -> None:
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype="PCM_24")


def to_stereo_48k(data: np.ndarray, sr: int) -> np.ndarray:
    """Any audio to 48 kHz stereo float32."""
    if data.ndim == 1:
        data = data[:, None]
    if sr != C.SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(C.SAMPLE_RATE, sr)
        data = resample_poly(data, C.SAMPLE_RATE // g, sr // g, axis=0).astype(np.float32)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2].astype(np.float32)


def detect_speech(wav_path: str | Path, threshold: float = 0.5) -> list[dict]:
    """Find where people are speaking, using the Silero VAD model (MIT licence)."""
    import onnxruntime as ort
    from scipy.signal import resample_poly

    data, sr = read_audio(wav_path)
    mono = data.mean(axis=1)
    if not np.any(np.abs(mono) > 1e-4):
        return []
    audio = resample_poly(mono, 16000, sr).astype(np.float32)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(ASSETS / "silero_vad.onnx"), opts,
                                   providers=["CPUExecutionProvider"])
    window, context_size = 512, 64
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, context_size), dtype=np.float32)
    probs = []
    for i in range(0, len(audio) - window + 1, window):
        chunk = audio[i:i + window][None, :]
        out, state = session.run(None, {
            "input": np.concatenate([context, chunk], axis=1),
            "state": state,
            "sr": np.array(16000, dtype=np.int64),
        })
        context = chunk[:, -context_size:]
        probs.append(float(out[0, 0]))
    hop = window / 16000
    # Turn per-window probabilities into segments, bridging short gaps.
    segments: list[list[float]] = []
    for i, p in enumerate(probs):
        if p >= threshold:
            t0, t1 = i * hop, (i + 1) * hop
            if segments and t0 - segments[-1][1] <= 0.4:
                segments[-1][1] = t1
            else:
                segments.append([t0, t1])
    total = len(audio) / 16000
    result = []
    for t0, t1 in segments:
        if t1 - t0 >= 0.25:
            result.append({"start": round(max(0.0, t0 - 0.1), 2), "end": round(min(total, t1 + 0.2), 2)})
    return result


def gate_to_speech(data: np.ndarray, speech: list[dict], sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Keep the original audio only where someone speaks, with short fades."""
    gain = np.zeros(len(data), dtype=np.float32)
    fade = int(0.06 * sr)
    for seg in speech:
        a, b = int(seg["start"] * sr), min(len(data), int(seg["end"] * sr))
        if b <= a:
            continue
        env = np.ones(b - a, dtype=np.float32)
        n = min(fade, (b - a) // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            env[:n], env[-n:] = ramp, ramp[::-1]
        gain[a:b] = np.maximum(gain[a:b], env)
    return data * gain[:, None]


# --- Clips for the effects model --------------------------------------------

def plan_segments(shots: list[dict], duration: float) -> list[dict]:
    """Cut the video into pieces for the effects model.

    Normally each shot is its own piece (long shots are split). With
    FOLEY_GROUP_SECONDS above zero, neighbouring short shots are joined into pieces up
    to that long, so the model sees more of the action at once.
    """
    groups: list[list[dict]] = []
    limit = float(C.FOLEY_GROUP_SECONDS or 0.0)
    for shot in shots:
        if groups and limit > 0 and shot["end"] - groups[-1][0]["start"] <= limit:
            groups[-1].append(shot)
        else:
            groups.append([shot])
    segments = []
    for group in groups:
        g_start, g_end = group[0]["start"], group[-1]["end"]
        length = g_end - g_start
        pieces = max(1, int(np.ceil(length / max(C.FOLEY_MAX_SEGMENT_SECONDS, limit))))
        step = length / pieces
        for p in range(pieces):
            start = g_start + p * step
            end = g_start + (p + 1) * step
            # Extra video on each side gives the crossfade real sound to blend.
            clip_start = max(0.0, start - C.FOLEY_OVERLAP_SECONDS)
            clip_end = min(duration, end + C.FOLEY_OVERLAP_SECONDS)
            segments.append({
                "index": len(segments), "shot": group[0]["index"],
                "shots": [s["index"] for s in group],
                "start": round(start, 3), "end": round(end, 3),
                "clip_start": round(clip_start, 3), "clip_end": round(clip_end, 3),
            })
    return segments


def cut_clip(path: str | Path, seg: dict, out: Path) -> Path:
    """A small, silent copy of one segment, kept under the upload size limit."""
    length = seg["clip_end"] - seg["clip_start"]
    s = C.FOLEY_CLIP_SHORT_SIDE
    scale = f"scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)',fps=25"
    for crf in (24, 28, 32, 36):
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{seg['clip_start']:.3f}", "-i", path,
             "-t", f"{length:.3f}", "-an", "-vf", scale, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", crf, "-pix_fmt", "yuv420p", out])
        if out.stat().st_size <= C.FOLEY_CLIP_MAX_BYTES:
            return out
    raise RuntimeError(f"Could not shrink clip {seg['index']} below the upload limit")


def still_clip(path: str | Path, at: float, seconds: float, out: Path) -> Path:
    """A silent clip that holds one frame of the video for `seconds`. Given to the
    effects model with a description, it yields a steady background sound for the place
    with nothing happening in it."""
    s = C.FOLEY_CLIP_SHORT_SIDE
    scale = f"scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)'"
    frame = out.with_suffix(".png")
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", path, "-frames:v", "1",
         "-vf", scale, frame])
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "25", "-i", frame,
         "-t", f"{seconds:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
         "-pix_fmt", "yuv420p", out])
    return out


def action_instant(path: str | Path, near: float, reach: float) -> float | None:
    """The instant, within `reach` seconds of `near`, where the picture changes most
    from one frame to the next (a hit, a snap, something landing), or None when
    nothing stands out. Used to put a key moment's sound on the exact frame."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    first = max(0, int((near - reach) * fps) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    changes, last = [], None
    for n in range(first, int((near + reach) * fps) + 2):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if last is not None:
            changes.append((n / fps, float(np.mean(np.abs(small - last)))))
        last = small
    cap.release()
    if len(changes) < 4:
        return None
    values = np.array([c[1] for c in changes])
    best = int(np.argmax(values))
    # A cut changes the whole picture at once and is not an action: leave those alone.
    if values[best] < 2.0 * (np.median(values) + 0.3) or values[best] > 45.0:
        return None
    return round(changes[best][0], 3)


def frames_at(path: str | Path, times: list[float], long_side: int) -> list[dict]:
    """Small JPEGs of the frames nearest to the given times, for Claude's closer looks.
    Returns {time, jpeg} with the time of the frame actually used; one pass over the video."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    wanted = sorted({max(0, int(round(t * fps))) for t in times})
    frames, n = [], 0
    if wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
        n = wanted[0]
    for index in wanted:
        frame = None
        while n <= index:           # read forward: seeking frame by frame is not exact
            ok, got = cap.read()
            n += 1
            if not ok:
                break
            frame = got
        if frame is None:
            break
        h, w = frame.shape[:2]
        scale = long_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)),
                               interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            frames.append({"time": round(index / fps, 3), "jpeg": jpeg.tobytes()})
    cap.release()
    return frames


def frame_rate(path: str | Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)
