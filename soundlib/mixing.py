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


def loudness_gain(data: np.ndarray, target: float, max_peak_db: float = -1.5) -> float:
    """The plain gain (a multiplier) that moves audio to a target loudness without letting
    peaks go over max_peak_db. 1.0 for silence."""
    current = lufs(data)
    if not np.isfinite(current):
        return 1.0
    gain = 10 ** ((target - current) / 20)
    peak = float(np.max(np.abs(data))) * gain
    limit = 10 ** (max_peak_db / 20)
    if peak > limit:
        gain *= limit / peak
    return float(gain)


def set_loudness(data: np.ndarray, target: float, max_peak_db: float = -1.5) -> np.ndarray:
    """Move audio to a target loudness without letting peaks go over max_peak_db."""
    gain = loudness_gain(data, target, max_peak_db)
    return data if gain == 1.0 else (data * gain).astype(np.float32)


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


def even_out_curve(data: np.ndarray, amount: float, max_boost_db: float,
                   window: float = 2.0) -> np.ndarray | None:
    """The gain (a multiplier per sample) that even_out applies, or None for no change."""
    sr = C.SAMPLE_RATE
    hop = int(window * sr / 4)
    if amount <= 0 or len(data) < 4 * hop:
        return None
    mono = data.mean(axis=1).astype(np.float64) ** 2
    centres = np.arange(0, len(data), hop)
    half = int(window * sr / 2)
    level = np.array([10 * np.log10(mono[max(0, c - half): c + half].mean() + 1e-10)
                      for c in centres])
    heard = level[level > level.max() - 50]
    gain_db = np.clip((np.median(heard) - level) * amount, -max_boost_db, max_boost_db)
    gain_db[level < level.max() - 50] = 0.0  # leave true silence alone
    curve = np.interp(np.arange(len(data)), centres, gain_db)
    return 10 ** (curve / 20)


def even_out(data: np.ndarray, amount: float, max_boost_db: float,
             window: float = 2.0) -> np.ndarray:
    """Gently reduce slow swings in level (for example a very quiet intro).

    Measures the level over `window`-second stretches and moves each stretch `amount`
    of the way (0 to 1) towards the overall median level. Fast dynamics are untouched.
    """
    curve = even_out_curve(data, amount, max_boost_db, window)
    return data if curve is None else (data * curve[:, None]).astype(np.float32)


def _loud_part_db(mono: np.ndarray) -> float:
    """Level of the loud parts of a sound: the 90th percentile of its 100 ms levels.
    Unlike a plain average it reads a few sharp sounds and a steady roar alike."""
    hop = int(0.1 * C.SAMPLE_RATE)
    if len(mono) < hop:
        return -120.0
    frames = mono[: len(mono) // hop * hop].reshape(-1, hop).astype(np.float64)
    levels = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    return float(np.percentile(levels, 90))


def balance_curve(sfx: np.ndarray, shots: list[dict]) -> np.ndarray | None:
    """The gain (a multiplier per sample) that balance_shots applies, or None for no change."""
    sr, mono = C.SAMPLE_RATE, sfx.mean(axis=1)
    times, gains = [], []
    for shot in shots:
        level = _loud_part_db(mono[int(shot["start"] * sr):int(shot["end"] * sr)])
        want = -24.0 + C.SHOT_LEVEL_DB[shot["loudness"]]
        limit = float(C.SHOT_LEVEL_MAX_CHANGE_DB)
        gains.append(0.0 if level < -80 else float(np.clip(want - level, -limit, limit)))
        times.append(shot["start"])
    if not gains:
        return None
    return 10 ** (_smooth_steps(times, gains, len(sfx), 0.3) / 20)


def balance_shots(sfx: np.ndarray, shots: list[dict]) -> np.ndarray:
    """Bring every shot's effects to the level its prominence calls for.

    `shots` items: start, end, loudness (quiet, medium or loud). Each shot's loud parts
    are moved to SHOT_LEVEL_DB[loudness] relative to one reference, by at most
    SHOT_LEVEL_MAX_CHANGE_DB, gliding between shots.
    """
    curve = balance_curve(sfx, shots)
    return sfx if curve is None else (sfx * curve[:, None]).astype(np.float32)


def onset_sample(audio: np.ndarray, below_peak_db: float = 25.0) -> int:
    """Where a sound really starts: the first moment it comes within `below_peak_db` of
    its loudest part (a little earlier, to keep its attack)."""
    sr = C.SAMPLE_RATE
    hop = int(0.005 * sr)
    level = _frame_db(audio.mean(axis=1), hop, int(0.02 * sr))
    if not len(level) or level.max() < -80:
        return 0
    first = int(np.argmax(level > level.max() - below_peak_db))
    return max(0, first * hop - int(0.015 * sr))


def peak_sample(audio: np.ndarray) -> int:
    """Where a sound is at its loudest (over 50 ms stretches)."""
    sr = C.SAMPLE_RATE
    hop = int(0.01 * sr)
    level = _frame_db(audio.mean(axis=1), hop, int(0.05 * sr))
    return int(np.argmax(level)) * hop if len(level) else 0


def text_effect_layer(seg: dict, duration: float) -> np.ndarray:
    """One sound made from its description alone, placed at its planned time as a
    full-length layer, at the level its planned prominence calls for."""
    sr = C.SAMPLE_RATE
    total = int(round(duration * sr))
    out = np.zeros((total, 2), dtype=np.float32)
    audio = level_clip(seg["audio"])
    want = int(round((seg["clip_end"] - seg["clip_start"]) * sr))
    audio = audio[:want].copy()
    level = _loud_part_db(audio.mean(axis=1))
    if level < -80 or not len(audio):
        return out
    audio *= 10 ** ((-24.0 + float(C.SHOT_LEVEL_DB[seg.get("loudness", "medium")]) - level) / 20)
    fade_in, fade_out = min(int(0.08 * sr), len(audio) // 2), min(int(0.3 * sr), len(audio) // 2)
    audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    a = int(round(seg["clip_start"] * sr))
    b = min(total, a + len(audio))
    out[a:b] = audio[: b - a]
    return out


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


def _spot_audio(spot: dict) -> np.ndarray:
    """One exact-moment effect, levelled and faded, ready to lay on the timeline.

    A supporting layer has a role: a "whoosh" leads into the moment, so it swells in
    and is gone just after the moment itself; a "detail" follows it.
    """
    sr = C.SAMPLE_RATE
    audio = level_clip(spot["audio"])
    role = spot.get("role", "main")
    if role == "whoosh":   # keep only the lead-in: the hit itself belongs to the main layer
        keep = int(round((spot["time"] - spot["clip_start"] + 0.1) * sr))
        audio = audio[: max(keep, int(0.3 * sr))]
    if C.SFX_LEVELING == "by_shot":
        audio = audio * 10 ** ((-24.0 - _loud_part_db(audio.mean(axis=1))) / 20)
    audio = audio * (10 ** ((spot["gain_db"] + C.SPOT_GAIN_DB) / 20))
    audio = audio.astype(np.float32).copy()
    seconds_in, seconds_out = {"whoosh": (0.25, 0.15), "detail": (0.08, 0.4)}.get(role, (0.05, 0.3))
    fade_in, fade_out = min(int(seconds_in * sr), len(audio) // 2), min(int(seconds_out * sr), len(audio) // 2)
    if fade_in:
        audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    if fade_out:
        audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    return audio


def ambience_layer(ambience: dict, duration: float) -> np.ndarray:
    """The steady background of the place, as one full-length layer: the generated
    stretch is repeated with long crossfades and set well under the other effects."""
    sr = C.SAMPLE_RATE
    total = int(round(duration * sr))
    out = np.zeros((total, 2), dtype=np.float32)
    clip = level_clip(ambience["audio"])
    # The model's first and last moments can carry a click or a fade: leave them out.
    trim = min(int(0.25 * sr), len(clip) // 8)
    clip = clip[trim: len(clip) - trim]
    if len(clip) < sr:
        return out
    level = _loud_part_db(clip.mean(axis=1))
    if level < -80:
        return out
    clip = clip * 10 ** ((-24.0 + float(C.AMBIENCE_LEVEL_DB) - level) / 20)
    cross = min(int(float(C.AMBIENCE_CROSSFADE_SECONDS) * sr), len(clip) // 3)
    step, at = len(clip) - cross, 0
    up = np.sin(np.linspace(0, np.pi / 2, cross, dtype=np.float32))[:, None]
    while at < total:
        piece = clip.copy()
        if at > 0:
            piece[:cross] *= up
        if at + step < total:
            piece[-cross:] *= up[::-1]
        b = min(total, at + len(piece))
        out[at:b] += piece[: b - at]
        at += step
    edge = min(int(0.4 * sr), total // 4)
    if edge:
        out[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)[:, None]
        out[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)[:, None]
    return out


def _spot_dip(n: int) -> np.ndarray | None:
    """How the sound already on the timeline makes room for an exact-moment effect that
    is n samples long (a multiplier per sample), or None when SPOT_BED_DIP_DB is 0."""
    dip = float(C.SPOT_BED_DIP_DB)
    if dip >= 0:
        return None
    ramp = min(int(0.15 * C.SAMPLE_RATE), n // 2)
    shape = np.ones(n, dtype=np.float32)
    shape[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
    shape[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
    return 10 ** (dip * shape / 20)


def add_spots(sfx: np.ndarray, spots: list[dict]) -> np.ndarray:
    """Lay extra one-off effects on the effects track at their exact moments.

    Each spot needs clip_start (seconds), gain_db and audio (48 kHz stereo).
    """
    sr = C.SAMPLE_RATE
    out = sfx.copy()
    for spot in spots:
        audio = _spot_audio(spot)
        a = int(round(spot["clip_start"] * sr))
        b = min(len(out), a + len(audio))
        if b > a:
            dip = _spot_dip(b - a)
            if dip is not None:   # the background makes room for the moment, easing down and back
                out[a:b] *= dip[:, None]
            out[a:b] += audio[: b - a]
    return out


def effects_layers(segments: list[dict], spots: list[dict], shot_levels: list[dict] | None,
                   room: str, duration: float, target_lufs: float,
                   ambience: dict | None = None) -> list[np.ndarray]:
    """The whole effects track, kept as one full-length layer per effect.

    The layers come back in order (every segment, then every spot, then the steady
    background of the place when there is one) and add up to the
    finished effects track: stitched, evened out, balanced shot by shot (when
    shot_levels is given), with the exact-moment effects, the room echo and the final
    loudness. Keeping them apart is what lets the editor move, mute or remove a single
    effect afterwards.
    """
    sr = C.SAMPLE_RATE
    total = int(round(duration * sr))
    from_text = [seg for seg in segments if seg.get("text")]
    layers = [stitch_effects([seg], duration) for seg in segments if not seg.get("text")]

    def together() -> np.ndarray:
        out = np.zeros((total, 2), dtype=np.float32)
        for layer in layers:
            out += layer
        return out

    def scale(curve: np.ndarray | None) -> None:
        if curve is not None:
            for i, layer in enumerate(layers):
                layers[i] = (layer * curve[:, None]).astype(np.float32)

    scale(even_out_curve(together(), float(C.SFX_EVEN_OUT), float(C.SFX_EVEN_OUT_MAX_DB)))
    if shot_levels is not None and layers:
        scale(balance_curve(together(), shot_levels))
    # Sounds made from text are each levelled on their own (order kept: segments first).
    layers = layers + [text_effect_layer(seg, duration) for seg in from_text]
    for spot in spots:
        audio = _spot_audio(spot)
        a = int(round(spot["clip_start"] * sr))
        b = min(total, a + len(audio))
        layer = np.zeros((total, 2), dtype=np.float32)
        if b > a:
            dip = _spot_dip(b - a)
            if dip is not None:
                for earlier in layers:
                    earlier[a:b] *= dip[:, None]
            layer[a:b] = audio[: b - a]
        layers.append(layer)
    layers = [add_reverb(layer, room, float(C.SFX_REVERB)) for layer in layers]
    if ambience is not None:
        layers.append(ambience_layer(ambience, duration))
    gain = loudness_gain(together(), target_lufs)
    return [(layer * gain).astype(np.float32) for layer in layers]


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


# --- Pro mix: making room for talking ----------------------------------------------

def _frame_db(mono: np.ndarray, hop: int, window: int) -> np.ndarray:
    """Level in dB of `window` samples around every `hop`-th sample."""
    power = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    centres = np.arange(0, len(mono), hop)
    a = np.clip(centres - window // 2, 0, len(mono))
    b = np.clip(centres + window // 2, 0, len(mono))
    return 10 * np.log10((power[b] - power[a]) / np.maximum(b - a, 1) + 1e-12)


def _glide(values: np.ndarray, hop_seconds: float, rise: float, fall: float) -> np.ndarray:
    """Smooth a stepped curve: it climbs over about `rise` seconds and drops over `fall`."""
    up, down = 1 - np.exp(-hop_seconds / max(rise, 1e-4)), 1 - np.exp(-hop_seconds / max(fall, 1e-4))
    out, now = np.empty_like(values, dtype=np.float64), float(values[0])
    for i, v in enumerate(values):
        now += (v - now) * (up if v > now else down)
        out[i] = now
    return out


def talk_presence(key: np.ndarray) -> np.ndarray | None:
    """How much someone is talking at every sample, from 0 (not) to 1 (fully), or None
    when the key sound is silent. It rises a moment BEFORE the words start and lets go
    slowly after they stop, so nothing lurches."""
    sr = C.SAMPLE_RATE
    mono = key.mean(axis=1)
    if not np.any(np.abs(mono) > 1e-4):
        return None
    hop = int(0.01 * sr)
    level = _frame_db(mono, hop, int(0.03 * sr))
    top = float(np.percentile(level[level > -70], 90)) if np.any(level > -70) else -70.0
    on = np.clip((level - (top - 32.0)) / 10.0, 0.0, 1.0)
    # Hold through the short gaps between words, and look ahead so the bed is already
    # down when the first word lands.
    hold, ahead = int(0.25 / 0.01), int(0.2 / 0.01)
    held = np.array([on[max(0, i - hold): i + ahead + 1].max() for i in range(len(on))])
    smooth = _glide(held, 0.01, 0.05, 0.35)
    return np.interp(np.arange(len(mono)), np.arange(len(smooth)) * hop, smooth).astype(np.float32)


def _highpass(data: np.ndarray, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    if hz <= 0 or len(data) < 64:
        return data
    sos = butter(2, hz, btype="highpass", fs=C.SAMPLE_RATE, output="sos")
    return sosfiltfilt(sos, data, axis=0).astype(np.float32)


def _speech_band(data: np.ndarray) -> np.ndarray:
    """The part of a sound in the range where speech is understood (1 to 4 kHz), cut
    out without shifting it in time, so the rest is exactly `data` minus this."""
    from scipy.signal import fftconvolve, firwin

    taps = firwin(511, [1000.0, 4000.0], pass_zero=False, fs=C.SAMPLE_RATE)
    return np.stack([fftconvolve(data[:, ch], taps, mode="same") for ch in range(data.shape[1])],
                    axis=1).astype(np.float32)


def make_room(sfx: np.ndarray, music: np.ndarray, presence: np.ndarray | None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Effects and music with room made for the talking: both drop while someone talks,
    the music further in the speech range."""
    if presence is None:
        return sfx, music
    p = presence[:, None]
    n = min(len(p), len(sfx), len(music))
    sfx, music, p = sfx[:n], music[:n], p[:n]
    band = _speech_band(music)
    wide = 10 ** (-float(C.MUSIC_DUCK_DB) * p / 20)
    narrow = 10 ** (-float(C.MUSIC_VOICE_BAND_DB) * p / 20)
    music = ((music - band) * wide + band * wide * narrow).astype(np.float32)
    sfx = (sfx * 10 ** (-float(C.SFX_DUCK_DB) * p / 20)).astype(np.float32)
    return sfx, music


def dialogue_gap(voice: np.ndarray, bed: np.ndarray, presence: np.ndarray | None) -> dict | None:
    """How far the talking sits above everything else while someone talks, in dB:
    the typical gap and the tightest one (the 5th percentile)."""
    if presence is None:
        return None
    sr = C.SAMPLE_RATE
    hop, window = int(0.05 * sr), int(0.4 * sr)
    n = min(len(voice), len(bed), len(presence))
    v = _frame_db(voice[:n].mean(axis=1), hop, window)
    b = _frame_db(bed[:n].mean(axis=1), hop, window)
    talking = (presence[:n:hop][: len(v)] > 0.9) & (v > -60)
    if not np.any(talking):
        return None
    gap = (v - b)[talking]
    return {"typical_db": round(float(np.median(gap)), 1), "tightest_db": round(float(np.percentile(gap, 5)), 1)}


def keep_dialogue_clear(voice: np.ndarray, sfx: np.ndarray, music: np.ndarray,
                        presence: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """The guarantee: wherever effects plus music still come within DIALOGUE_MIN_GAP_DB
    of the talking, both are eased down by just enough (18 dB at most)."""
    if presence is None:
        return sfx, music
    sr = C.SAMPLE_RATE
    hop, window = int(0.05 * sr), int(0.4 * sr)
    n = min(len(voice), len(sfx), len(music), len(presence))
    v = _frame_db(voice[:n].mean(axis=1), hop, window)
    b = _frame_db((sfx[:n] + music[:n]).mean(axis=1), hop, window)
    talking = (presence[:n:hop][: len(v)] > 0.5) & (v > -60)
    need = np.where(talking, np.clip(b - (v - float(C.DIALOGUE_MIN_GAP_DB)) + 0.5, 0.0, 18.0), 0.0)
    if not np.any(need > 0):
        return sfx, music
    reach = int(0.3 / 0.05)
    spread = np.array([need[max(0, i - reach): i + reach + 1].max() for i in range(len(need))])
    curve = _glide(spread, 0.05, 0.1, 0.3)
    gain = 10 ** (-np.interp(np.arange(n), np.arange(len(curve)) * hop, curve) / 20)[:, None]
    sfx, music = sfx.copy(), music.copy()
    sfx[:n] = sfx[:n] * gain
    music[:n] = music[:n] * gain
    return sfx.astype(np.float32), music.astype(np.float32)


def pro_premix(sfx: np.ndarray, music: np.ndarray, voice: np.ndarray | None,
               key: np.ndarray | None, voice_is_talking_only: bool) -> dict:
    """The pro mix before the final loudness: cleaned layers, room made for talking,
    and the dialogue guarantee. Returns the three finished layers, their sum and the
    measured gap between the talking and everything else."""
    n = min(len(sfx), len(music), *( [len(voice)] if voice is not None else []))
    sfx, music = sfx[:n], music[:n]
    hz = C.RUMBLE_HZ
    sfx, music = _highpass(sfx, float(hz["sfx"])), _highpass(music, float(hz["music"]))
    presence = talk_presence(key[:n]) if key is not None else None
    gap = None
    if voice is not None:
        # A kept voice loses its rumble; a kept whole soundtrack keeps its bass.
        voice = _highpass(voice[:n], float(hz["voice"]) if voice_is_talking_only else float(hz["music"]))
        sfx, music = make_room(sfx, music, presence)
        if key is not None:
            sfx, music = keep_dialogue_clear(key[:n], sfx, music, presence)
            gap = dialogue_gap(key[:n], sfx + music, presence)
    total = sfx + music + (voice if voice is not None else 0.0)
    return {"sfx": sfx, "music": music, "voice": voice, "mix": total.astype(np.float32), "dialogue": gap}


def preview_music(music: np.ndarray, key: np.ndarray) -> np.ndarray:
    """What the editor plays for the music in the pro mix: the music with the room for
    talking already made (that depends only on the talking, so it holds at any volume)."""
    presence = talk_presence(key)
    return make_room(np.zeros_like(music), music, presence)[1]


# --- Final mix ---------------------------------------------------------------

def _loudnorm_measure(wav: Path, target_peak: float) -> dict:
    proc = run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav, "-af",
                f"loudnorm=I={C.TARGET_LUFS}:TP={target_peak}:LRA=11:print_format=json",
                "-f", "null", "-"])
    return json.loads(proc.stderr[proc.stderr.rfind("{"): proc.stderr.rfind("}") + 1])


def _classic_premix(sfx_wav: Path, music_wav: Path, voice_wav: Path | None,
                    duck_key: Path | None | str, premix: Path) -> None:
    """The classic mix before normalisation, made with ffmpeg."""
    inputs = ["-i", sfx_wav, "-i", music_wav]
    chain, sfx, music = [], "[0:a]", "[1:a]"
    if C.MUSIC_DUCK_UNDER_SFX:  # the music dips when the effects get loud
        chain.append("[0:a]asplit=2[sfx][sfxkey]")
        chain.append(f"[1:a][sfxkey]sidechaincompress={C.DUCK_UNDER_SFX}[m1]")
        sfx, music = "[sfx]", "[m1]"
    # duck_key: what the music dips under. "voice" = the voice track itself; a file = that
    # sound (the talking, when the voice track is the video's whole original audio);
    # None = the music does not dip.
    if voice_wav is not None:
        inputs += ["-i", voice_wav]
        if duck_key is None:
            chain.append(f"{sfx}{music}[2:a]amix=inputs=3:normalize=0:duration=longest[out]")
        else:
            if duck_key == "voice" or Path(duck_key) == Path(voice_wav):
                chain.append("[2:a]asplit=2[voice][key]")
                voice, key = "[voice]", "[key]"
            else:
                inputs += ["-i", duck_key]
                voice, key = "[2:a]", "[3:a]"
            chain.append(f"{music}{key}sidechaincompress={C.DUCK}[m2]")
            chain.append(f"{sfx}[m2]{voice}amix=inputs=3:normalize=0:duration=longest[out]")
    else:
        chain.append(f"{sfx}{music}amix=inputs=2:normalize=0:duration=longest[out]")
    graph = ";".join(chain)
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
         "-ar", C.SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_f32le", premix])


def mix(video: Path, sfx_wav: Path, music_wav: Path, voice_wav: Path | None,
        out_mp4: Path, work: Path, duration: float, duck_key: Path | None | str = "voice",
        deliver: tuple[Path, Path] | None = None) -> dict:
    """Mix voice, music and effects and write the final MP4.

    MIX_STYLE "classic": voice on top, music ducked under the voice with sidechain
    compression, effects under the voice, then loudness normalisation.
    MIX_STYLE "pro": see pro_premix; then one steady gain to the target loudness and a
    true-peak limiter. With `deliver` (effects path, music path) the two layers are also
    written as they sit in the finished mix.

    Returns the measured loudness and true peak of the finished file, plus, for pro,
    the gain used and how far the talking sits above the rest.
    """
    premix = work / "premix.wav"
    peak_target = C.TARGET_TRUE_PEAK_DB - C.ENCODE_PEAK_MARGIN_DB
    normalised = work / "mix.wav"
    extra: dict = {"style": C.MIX_STYLE}
    pro = C.MIX_STYLE == "pro"
    if pro:
        same = voice_wav is not None and (duck_key == "voice" or (
            duck_key is not None and Path(duck_key) == Path(voice_wav)))
        voice = load_stereo(voice_wav) if voice_wav is not None else None
        key = voice if same else load_stereo(Path(duck_key)) if duck_key not in (None, "voice") else None
        parts = pro_premix(load_stereo(sfx_wav), load_stereo(music_wav), voice, key, same)
        extra["dialogue"] = parts["dialogue"]
        import soundfile as sf

        sf.write(str(premix), parts["mix"], C.SAMPLE_RATE, subtype="FLOAT")
        got = measure_file(premix)
        gain_db = C.TARGET_LUFS - got["lufs"] if np.isfinite(got["lufs"]) else 0.0
        run(["ffmpeg", "-y", "-v", "error", "-i", premix, "-af", f"volume={gain_db:.3f}dB",
             "-ar", C.SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_f32le", normalised])
        extra["gain_db"] = round(float(gain_db), 2)
    else:
        _classic_premix(sfx_wav, music_wav, voice_wav, duck_key, premix)
        # Two-pass loudness normalisation. The peak target leaves room for the small
        # overshoot that AAC encoding adds, so the finished file stays at or under -1 dBTP.
        m = _loudnorm_measure(premix, peak_target)
        second_pass = (
            f"loudnorm=I={C.TARGET_LUFS}:TP={peak_target}:LRA=11:linear=true:"
            f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
            f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}"
        )
        run(["ffmpeg", "-y", "-v", "error", "-i", premix, "-af", second_pass,
             "-ar", C.SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_s24le", normalised])

    # The limiter. Pro: it looks at the sound four times finer than the file stores it,
    # so it catches the peaks between samples, and it reacts fast and lets go fast, which
    # keeps it inaudible. Classic: the safety net for when loudnorm missed the target.
    def limit(ceiling_db: float, change_db: float = 0.0) -> None:
        fixed = work / "mix_fixed.wav"
        level = 10 ** (ceiling_db / 20)
        chain = (f"volume={change_db:.2f}dB,aresample={4 * int(C.SAMPLE_RATE)},"
                 f"alimiter=limit={level:.4f}:level=false:attack=1:release=50,aresample={C.SAMPLE_RATE}"
                 if pro else
                 f"volume={change_db:.2f}dB,alimiter=limit={level:.4f}:level=false:attack=5:release=80,"
                 f"aresample={C.SAMPLE_RATE}")
        run(["ffmpeg", "-y", "-v", "error", "-i", normalised, "-af", chain,
             "-ac", "2", "-c:a", "pcm_f32le" if pro else "pcm_s24le", fixed])
        fixed.replace(normalised)

    for _ in range(4 if pro else 3):
        got = measure_file(normalised)
        off = C.TARGET_LUFS - got["lufs"]
        if abs(off) <= 0.3 and got["true_peak_db"] <= peak_target + 0.1:
            break
        if pro:
            extra["gain_db"] = round(extra["gain_db"] + float(off), 2)
        limit(peak_target, off)

    if pro and deliver is not None:
        # The layers as they sit in the finished video (before the limiter touches the
        # few highest peaks), so the two downloads played together give the same mix.
        gain = 10 ** (extra["gain_db"] / 20)
        peak = max(float(np.abs(parts["sfx"]).max(initial=0.0)), float(np.abs(parts["music"]).max(initial=0.0))) * gain
        gain *= min(1.0, 0.98 / peak) if peak > 0 else 1.0
        write_audio(deliver[0], parts["sfx"] * gain)
        write_audio(deliver[1], parts["music"] * gain)

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
        limit(ceiling)
        result = encode()
    return {**result, **extra}


def save_stem(path: Path, data: np.ndarray) -> None:
    write_audio(path, data)


def load_stereo(path: Path) -> np.ndarray:
    data, sr = read_audio(path)
    return to_stereo_48k(data, sr)
