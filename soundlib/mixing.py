"""Stitching effect clips, shaping the music, and the final ffmpeg mix."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import config as C
from .media import read_audio, run, to_stereo_48k, write_audio


# --- Loudness helpers --------------------------------------------------------

def lufs(data: np.ndarray, sr: int = C.SAMPLE_RATE) -> float:
    """Integrated loudness. Returns -inf for silence or audio too short to measure."""
    import pyloudnorm as pyln

    if len(data) < int(0.5 * sr) or not np.any(np.abs(data) > 1e-5):
        return float("-inf")
    value = pyln.Meter(sr).integrated_loudness(data.astype(np.float64))
    return float(value) if np.isfinite(value) else float("-inf")


def set_loudness(data: np.ndarray, target: float, max_peak_db: float = -1.5) -> np.ndarray:
    """Move audio to a target loudness without letting peaks go over max_peak_db."""
    current = lufs(data)
    if not np.isfinite(current):
        return data
    gain = 10 ** ((target - current) / 20)
    peak = float(np.max(np.abs(data))) * gain
    limit = 10 ** (max_peak_db / 20)
    if peak > limit:
        gain *= limit / peak
    return (data * gain).astype(np.float32)


def measure_file(path: str | Path) -> dict:
    """Loudness and true peak of any media file, measured by ffmpeg (EBU R128)."""
    proc = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:a:0",
                "-af", "ebur128=peak=true", "-f", "null", "-"])
    summary = proc.stderr[proc.stderr.rfind("Summary:"):]
    loud = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    as_float = lambda m: float(m.group(1)) if m and m.group(1) != "-inf" else float("-inf")
    return {"lufs": as_float(loud), "true_peak_db": as_float(peak)}


# --- Effects -----------------------------------------------------------------

def level_clip(audio: np.ndarray) -> np.ndarray:
    """Scale a clip to SFX_CLIP_RMS_DB, with a cap on how much a quiet clip is boosted."""
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    if rms < 1e-6:
        return audio
    gain_db = min(C.SFX_CLIP_RMS_DB - 20 * np.log10(rms), C.SFX_CLIP_MAX_BOOST_DB)
    return (audio * 10 ** (gain_db / 20)).astype(np.float32)


def even_out(data: np.ndarray, amount: float, max_boost_db: float,
             window: float = 2.0) -> np.ndarray:
    """Gently reduce slow swings in level (for example a very quiet intro).

    Measures the level over `window`-second stretches and moves each stretch `amount`
    of the way (0 to 1) towards the overall median level. Fast dynamics are untouched.
    """
    sr = C.SAMPLE_RATE
    hop = int(window * sr / 4)
    if amount <= 0 or len(data) < 4 * hop:
        return data
    mono = data.mean(axis=1).astype(np.float64) ** 2
    centres = np.arange(0, len(data), hop)
    half = int(window * sr / 2)
    level = np.array([10 * np.log10(mono[max(0, c - half): c + half].mean() + 1e-10)
                      for c in centres])
    heard = level[level > level.max() - 50]
    gain_db = np.clip((np.median(heard) - level) * amount, -max_boost_db, max_boost_db)
    gain_db[level < level.max() - 50] = 0.0  # leave true silence alone
    curve = np.interp(np.arange(len(data)), centres, gain_db)
    return (data * (10 ** (curve / 20))[:, None]).astype(np.float32)


def stitch_effects(segments: list[dict], duration: float) -> np.ndarray:
    """Lay the effect clips on one timeline, crossfading where neighbours overlap.

    Each segment dict needs: start, end, clip_start, clip_end, gain_db and
    audio (samples, channels) at 48 kHz stereo covering clip_start..clip_end.
    """
    sr = C.SAMPLE_RATE
    total = int(round(duration * sr))
    out = np.zeros((total, 2), dtype=np.float32)
    for seg in segments:
        # The effects model returns each clip at its own arbitrary level (some 30 dB
        # quieter than others), so bring every clip to one level before the plan's gain.
        audio = level_clip(seg["audio"]) * (10 ** (seg.get("gain_db", 0.0) / 20))
        want = int(round((seg["clip_end"] - seg["clip_start"]) * sr))
        if len(audio) < want:  # the model may return slightly less than asked for
            audio = np.concatenate([audio, np.zeros((want - len(audio), 2), dtype=np.float32)])
        audio = audio[:want].copy()
        # Equal-power fades across the overlap shared with each neighbour. A segment's
        # fade is centred on its real boundary, so the two sides always sum cleanly.
        lead = int(round((seg["start"] - seg["clip_start"]) * 2 * sr))
        tail = int(round((seg["clip_end"] - seg["end"]) * 2 * sr))
        edge = int(0.01 * sr)  # tiny fade at the very start and end of the video
        for n, at_start in ((lead or edge, True), (tail or edge, False)):
            n = min(n, len(audio) // 2)
            if n <= 0:
                continue
            ramp = np.sin(np.linspace(0, np.pi / 2, n, dtype=np.float32))[:, None]
            if at_start:
                audio[:n] *= ramp
            else:
                audio[-n:] *= ramp[::-1]
        a = int(round(seg["clip_start"] * sr))
        b = min(total, a + len(audio))
        out[a:b] += audio[: b - a]
    return out


# --- Music -------------------------------------------------------------------

def shape_music(data: np.ndarray, sr: int, duration: float) -> np.ndarray:
    """Trim the music to the video length, even out slow level swings, fade the end."""
    data = to_stereo_48k(data, sr)
    total = int(round(duration * C.SAMPLE_RATE))
    if len(data) < total:
        data = np.concatenate([data, np.zeros((total - len(data), 2), dtype=np.float32)])
    data = even_out(data[:total].copy(), C.MUSIC_EVEN_OUT, C.MUSIC_EVEN_OUT_MAX_DB)
    fade = min(int(C.MUSIC_FADE_OUT_SECONDS * C.SAMPLE_RATE), total // 4)
    if fade > 0:
        data[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None] ** 2
    intro = min(int(0.03 * C.SAMPLE_RATE), total // 4)
    if intro > 0:
        data[:intro] *= np.linspace(0.0, 1.0, intro, dtype=np.float32)[:, None]
    return data


# --- Final mix ---------------------------------------------------------------

def _loudnorm_measure(wav: Path, target_peak: float) -> dict:
    proc = run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav, "-af",
                f"loudnorm=I={C.TARGET_LUFS}:TP={target_peak}:LRA=11:print_format=json",
                "-f", "null", "-"])
    return json.loads(proc.stderr[proc.stderr.rfind("{"): proc.stderr.rfind("}") + 1])


def mix(video: Path, sfx_wav: Path, music_wav: Path, voice_wav: Path | None,
        out_mp4: Path, work: Path, duration: float) -> dict:
    """Mix voice, music and effects with ffmpeg and write the final MP4.

    Voice on top, music ducked under the voice with sidechain compression, effects
    under the voice. Then normalise to -14 LUFS with true peak at -1 dB.
    """
    premix = work / "premix.wav"
    inputs = ["-i", sfx_wav, "-i", music_wav]
    if voice_wav is not None:
        inputs += ["-i", voice_wav]
        graph = (
            "[2:a]asplit=2[voice][key];"
            f"[1:a][key]sidechaincompress={C.DUCK}[music];"
            "[0:a][music][voice]amix=inputs=3:normalize=0:duration=longest[out]"
        )
    else:
        graph = "[0:a][1:a]amix=inputs=2:normalize=0:duration=longest[out]"
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
         "-ar", C.SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_f32le", premix])

    # Two-pass loudness normalisation. The peak target leaves room for the small
    # overshoot that AAC encoding adds, so the finished file stays at or under -1 dBTP.
    peak_target = C.TARGET_TRUE_PEAK_DB - C.ENCODE_PEAK_MARGIN_DB
    m = _loudnorm_measure(premix, peak_target)
    normalised = work / "mix.wav"
    second_pass = (
        f"loudnorm=I={C.TARGET_LUFS}:TP={peak_target}:LRA=11:linear=true:"
        f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
        f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}"
    )
    run(["ffmpeg", "-y", "-v", "error", "-i", premix, "-af", second_pass,
         "-ar", C.SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_s24le", normalised])

    # Safety net: if loudnorm missed the target (it can on very dynamic material),
    # correct with plain gain and a true-peak limiter.
    for _ in range(3):
        got = measure_file(normalised)
        off = C.TARGET_LUFS - got["lufs"]
        if abs(off) <= 0.3 and got["true_peak_db"] <= peak_target + 0.1:
            break
        fixed = work / "mix_fixed.wav"
        limit = 10 ** (peak_target / 20)
        run(["ffmpeg", "-y", "-v", "error", "-i", normalised, "-af",
             f"volume={off:.2f}dB,alimiter=limit={limit:.4f}:level=false:attack=5:release=80,"
             f"aresample={C.SAMPLE_RATE}",
             "-ac", "2", "-c:a", "pcm_s24le", fixed])
        fixed.replace(normalised)

    # Put the new sound on the untouched picture. The audio is padded or cut to the
    # exact video length so the output is the same length as the input.
    run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", normalised,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-af", f"apad,atrim=0:{duration:.4f}", "-c:a", "aac", "-b:a", C.AAC_BITRATE,
         "-ar", C.SAMPLE_RATE, "-t", f"{duration:.4f}", "-movflags", "+faststart", out_mp4])
    return measure_file(out_mp4)


def save_stem(path: Path, data: np.ndarray) -> None:
    write_audio(path, data)


def load_stereo(path: Path) -> np.ndarray:
    data, sr = read_audio(path)
    return to_stereo_48k(data, sr)
