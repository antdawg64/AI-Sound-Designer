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


def _loud_part_db(mono: np.ndarray) -> float:
    """Level of the loud parts of a sound: the 90th percentile of its 100 ms levels.
    Unlike a plain average it reads a few sharp sounds and a steady roar alike."""
    hop = int(0.1 * C.SAMPLE_RATE)
    if len(mono) < hop:
        return -120.0
    frames = mono[: len(mono) // hop * hop].reshape(-1, hop).astype(np.float64)
    levels = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    return float(np.percentile(levels, 90))


def balance_shots(sfx: np.ndarray, shots: list[dict]) -> np.ndarray:
    """Bring every shot's effects to the level its prominence calls for.

    `shots` items: start, end, loudness (quiet, medium or loud). Each shot's loud parts
    are moved to SHOT_LEVEL_DB[loudness] relative to one reference, by at most
    SHOT_LEVEL_MAX_CHANGE_DB, gliding between shots.
    """
    sr, mono = C.SAMPLE_RATE, sfx.mean(axis=1)
    times, gains = [], []
    for shot in shots:
        level = _loud_part_db(mono[int(shot["start"] * sr):int(shot["end"] * sr)])
        want = -24.0 + C.SHOT_LEVEL_DB[shot["loudness"]]
        limit = float(C.SHOT_LEVEL_MAX_CHANGE_DB)
        gains.append(0.0 if level < -80 else float(np.clip(want - level, -limit, limit)))
        times.append(shot["start"])
    if not gains:
        return sfx
    curve = _smooth_steps(times, gains, len(sfx), 0.3)
    return (sfx * (10 ** (curve / 20))[:, None]).astype(np.float32)


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


def add_spots(sfx: np.ndarray, spots: list[dict]) -> np.ndarray:
    """Lay extra one-off effects on the effects track at their exact moments.

    Each spot needs clip_start (seconds), gain_db and audio (48 kHz stereo).
    """
    sr = C.SAMPLE_RATE
    out = sfx.copy()
    for spot in spots:
        audio = level_clip(spot["audio"])
        if C.SFX_LEVELING == "by_shot":
            audio = audio * 10 ** ((-24.0 - _loud_part_db(audio.mean(axis=1))) / 20)
        audio = audio * (10 ** ((spot["gain_db"] + C.SPOT_GAIN_DB) / 20))
        audio = audio.copy()
        fade_in, fade_out = min(int(0.05 * sr), len(audio) // 2), min(int(0.3 * sr), len(audio) // 2)
        if fade_in:
            audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
        if fade_out:
            audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
        a = int(round(spot["clip_start"] * sr))
        b = min(len(out), a + len(audio))
        if b > a:
            dip = float(C.SPOT_BED_DIP_DB)
            if dip < 0:   # the background makes room for the moment, easing down and back
                ramp = min(int(0.15 * sr), (b - a) // 2)
                shape = np.ones(b - a, dtype=np.float32)
                shape[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                shape[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
                out[a:b] *= (10 ** (dip * shape / 20))[:, None]
            out[a:b] += audio[: b - a]
    return out


REVERB_SECONDS = {"tiny enclosed space": 0.18, "small room": 0.32, "medium room": 0.5,
                  "large room": 0.9, "hall": 1.6, "outdoors": 0.0}


def add_reverb(data: np.ndarray, room: str, amount: float) -> np.ndarray:
    """Add room echo sized to `room`. `amount` is how loud the echo is (0 to 1)."""
    from scipy.signal import fftconvolve

    seconds = REVERB_SECONDS.get(room, 0.5)
    if amount <= 0 or seconds <= 0 or not len(data):
        return data
    sr = C.SAMPLE_RATE
    n = int(seconds * 1.5 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    wet = np.zeros_like(data, dtype=np.float64)
    pre = int(0.012 * sr)
    for ch in range(data.shape[1]):
        ir = rng.standard_normal(n) * np.exp(-6.91 * t / seconds)   # 60 dB down at `seconds`
        ir = np.convolve(ir, np.ones(6) / 6, mode="same")            # soften the top end
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
        tail = fftconvolve(data[:, ch].astype(np.float64), ir)[: len(data)]
        wet[pre:, ch] = tail[: len(data) - pre]
    return (data + amount * wet).astype(np.float32)


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


def _smooth_steps(times: list[float], values: list[float], total: int, ramp: float) -> np.ndarray:
    """A curve that holds each value from its time on, gliding between them over `ramp` seconds."""
    sr = C.SAMPLE_RATE
    xs, ys = [0.0], [values[0]]
    for t, prev, cur in zip(times[1:], values[:-1], values[1:]):
        xs += [max(xs[-1], t - ramp / 2), t + ramp / 2]
        ys += [prev, cur]
    xs.append(total / sr + 1.0)
    ys.append(values[-1])
    return np.interp(np.arange(total) / sr, xs, ys)


def shape_to_cues(music: np.ndarray, cues: dict) -> np.ndarray:
    """Make the music follow the cue sheet: each section at its intensity, and for each
    hit a dip just before it and an accent that lands on it and fades."""
    sr, total = C.SAMPLE_RATE, len(music)
    secs = cues["sections"]
    curve = _smooth_steps([s["start"] for s in secs],
                          [C.CUE_INTENSITY_DB[s["intensity"] - 1] for s in secs], total, float(C.CUE_RAMP_SECONDS))
    t = np.arange(total) / sr
    for hit in cues["hits"]:
        dip, accent = C.CUE_HIT_DIP_DB[hit["strength"]], C.CUE_HIT_ACCENT_DB[hit["strength"]]
        lead = 0.5 if hit["strength"] == "big" else 0.25
        before = (t >= hit["time"] - lead) & (t < hit["time"])
        curve[before] += dip * np.clip((t[before] - (hit["time"] - lead)) / (lead * 0.5), 0.0, 1.0)
        after = (t >= hit["time"]) & (t < hit["time"] + 1.0)
        curve[after] += accent * (1.0 - (t[after] - hit["time"]) / 1.0) ** 2
    return (music * (10 ** (curve / 20))[:, None]).astype(np.float32)


def best_start(part: np.ndarray, need: int, bar_seconds: float) -> int:
    """Where to start reading a generated piece: on a bar line, skipping a weak intro."""
    sr = C.SAMPLE_RATE
    probe, best, best_level = int(1.5 * sr), 0, -1.0
    overall = float(np.sqrt(np.mean(np.square(part, dtype=np.float64)))) + 1e-9
    for k in range(0, 5):
        a = int(round(k * bar_seconds * sr))
        if a + need > len(part):
            break
        level = float(np.sqrt(np.mean(np.square(part[a:a + probe], dtype=np.float64)))) / overall
        if level >= 0.7:          # first bar line where the music is properly under way
            return a
        if level > best_level:
            best, best_level = a, level
    return best


def join_sections(parts: list[np.ndarray], sections: list[dict], duration: float, bpm: int) -> np.ndarray:
    """Lay separately composed sections on one timeline, each with a downbeat on its cut.

    With SECTION_CROSSFADE_SECONDS at 0 a section starts sharply on the cut while the
    one before fades quickly underneath. Above 0, neighbours blend over that many
    seconds, centred on the cut.
    """
    sr = C.SAMPLE_RATE
    total = int(round(duration * sr))
    out = np.zeros((total, 2), dtype=np.float32)
    blend = int(float(C.SECTION_CROSSFADE_SECONDS) * sr)
    tail, head = int(0.3 * sr), int(0.03 * sr)
    for i, (part, sec) in enumerate(zip(parts, sections)):
        a = int(round(sec["start"] * sr))
        length = int(round((sec["end"] - sec["start"]) * sr))
        part = set_loudness(part, -20.0)
        if blend > 0:
            lead = blend // 2 if i > 0 else 0          # starts early, fading in
            lag = blend // 2 if i < len(sections) - 1 else 0
            start = best_start(part, length + lag + lead, 4 * 60.0 / bpm)
            start = max(start, lead)
            piece = part[start - lead:start + length + lag].copy()
            if lead:
                n = min(2 * lead, len(piece))
                piece[:n] *= np.sin(np.linspace(0, np.pi / 2, n, dtype=np.float32))[:, None]
            if lag:
                n = min(2 * lag, len(piece))
                piece[-n:] *= np.cos(np.linspace(0, np.pi / 2, n, dtype=np.float32))[:, None]
            a -= lead
        else:
            start = best_start(part, length + tail, 4 * 60.0 / bpm)
            piece = part[start:start + length + tail].copy()
            if a > 0 and len(piece) > head:
                piece[:head] *= np.linspace(0.0, 1.0, head, dtype=np.float32)[:, None]
            n = min(tail, len(piece))
            piece[-n:] *= np.cos(np.linspace(0, np.pi / 2, n, dtype=np.float32))[:, None]
        b = min(total, a + len(piece))
        out[a:b] += piece[: b - a]
    return out


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
    chain, sfx, music = [], "[0:a]", "[1:a]"
    if C.MUSIC_DUCK_UNDER_SFX:  # the music dips when the effects get loud
        chain.append("[0:a]asplit=2[sfx][sfxkey]")
        chain.append(f"[1:a][sfxkey]sidechaincompress={C.DUCK_UNDER_SFX}[m1]")
        sfx, music = "[sfx]", "[m1]"
    if voice_wav is not None:
        inputs += ["-i", voice_wav]
        chain.append("[2:a]asplit=2[voice][key]")
        chain.append(f"{music}[key]sidechaincompress={C.DUCK}[m2]")
        chain.append(f"{sfx}[m2][voice]amix=inputs=3:normalize=0:duration=longest[out]")
    else:
        chain.append(f"{sfx}{music}amix=inputs=2:normalize=0:duration=longest[out]")
    graph = ";".join(chain)
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
    def encode() -> dict:
        run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", normalised,
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
             "-af", f"apad,atrim=0:{duration:.4f}", "-c:a", "aac", "-b:a", C.AAC_BITRATE,
             "-ar", C.SAMPLE_RATE, "-t", f"{duration:.4f}", "-movflags", "+faststart", out_mp4])
        return measure_file(out_mp4)

    # Heavily limited peaks can overshoot by more than the margin once encoded. If the
    # finished file is over the peak target, limit harder by the excess and encode again.
    result, ceiling = encode(), peak_target
    for _ in range(3):
        if result["true_peak_db"] <= C.TARGET_TRUE_PEAK_DB:
            break
        ceiling -= result["true_peak_db"] - C.TARGET_TRUE_PEAK_DB + 0.3
        fixed = work / "mix_fixed.wav"
        run(["ffmpeg", "-y", "-v", "error", "-i", normalised, "-af",
             f"alimiter=limit={10 ** (ceiling / 20):.4f}:level=false:attack=5:release=80,"
             f"aresample={C.SAMPLE_RATE}",
             "-ac", "2", "-c:a", "pcm_s24le", fixed])
        fixed.replace(normalised)
        result = encode()
    return result


def save_stem(path: Path, data: np.ndarray) -> None:
    write_audio(path, data)


def load_stereo(path: Path) -> np.ndarray:
    data, sr = read_audio(path)
    return to_stereo_48k(data, sr)
