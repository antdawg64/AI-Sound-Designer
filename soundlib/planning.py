"""The sound plan: what Claude is asked for, and how the answer is cleaned up and used.

`ask_claude` runs inside Modal (that is where the API key lives). Everything else
runs wherever make_sound.py runs.
"""

from __future__ import annotations

import base64
from typing import Literal

import numpy as np
from pydantic import BaseModel

from . import config as C


class Effect(BaseModel):
    start: float
    end: float
    description: str
    loudness: Literal["quiet", "medium", "loud"]


class ShotPlan(BaseModel):
    shot: int
    effects: list[Effect]


class MusicBrief(BaseModel):
    mood: str
    genre: str
    tempo_bpm: int
    instruments: list[str]
    cut_times: list[float]


class SoundPlan(BaseModel):
    summary: str
    shots: list[ShotPlan]
    music: MusicBrief


SYSTEM = """You are a sound designer planning the audio for a short video. You are shown \
still frames sampled from each shot, with their timestamps. Another model will generate \
sound effects from the video and your descriptions, and a music model will write an \
instrumental track from your brief.

For every shot, list the sound effects that should be heard:
- start and end are seconds from the start of the whole video and must sit inside the shot.
- description is a short, concrete description of the sound itself, the way a foley \
artist would label it: the source, the material, the action ("vacuum cleaner motor \
humming steadily over carpet", "ceramic mug set down on a wooden table"). Describe only \
sounds, never the picture. No music, no speech, no brand names.
- loudness is how prominent it should be in the mix.
- Every shot needs at least one effect. If nothing obvious happens, describe the room \
tone or ambience that fits what is visible. Prefer one to four effects per shot.

Then write one music brief for the whole video:
- mood, genre and instruments for an instrumental background track that suits the \
content and leaves room for effects and any voice.
- tempo_bpm between 60 and 180 that fits the pace of the cuts.
- cut_times: the shot changes (in seconds) the music should hit. Use the shot boundaries \
you were given; leave it empty if there is only one shot.

summary is one sentence saying what the video shows."""


def build_user_content(frames: list[dict], shots: list[dict], duration: float,
                       speech: list[dict]) -> list[dict]:
    """The message for Claude. Each frame dict needs: shot, time, jpeg (bytes)."""
    lines = [f"The video is {duration:.2f} seconds long and has {len(shots)} shot(s):"]
    for s in shots:
        lines.append(f"- shot {s['index']}: {s['start']:.2f}s to {s['end']:.2f}s")
    if speech:
        spans = ", ".join(f"{s['start']:.1f}-{s['end']:.1f}s" for s in speech)
        lines.append(f"People are speaking during: {spans}. Their voice is kept in the final "
                     "mix, so keep effects and music clear of it.")
    else:
        lines.append("Nobody speaks in this video.")
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for f in frames:
        content.append({"type": "text", "text": f"Shot {f['shot']}, frame at {f['time']:.2f}s:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(f["jpeg"]).decode("ascii"),
        }})
    content.append({"type": "text", "text": "Plan the sound for this video."})
    return content


def ask_claude(frames: list[dict], shots: list[dict], duration: float, speech: list[dict]) -> dict:
    """Send the key frames to Claude and get the plan back as a dict."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=C.PLAN_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_user_content(frames, shots, duration, speech)}],
        output_format=SoundPlan,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to plan the sound for this video.")
    if response.parsed_output is None:
        raise RuntimeError(f"Claude returned no usable plan (stop reason: {response.stop_reason}).")
    usage = response.usage
    cost = (usage.input_tokens * C.CLAUDE_DOLLARS_PER_MTOK["input"]
            + usage.output_tokens * C.CLAUDE_DOLLARS_PER_MTOK["output"]) / 1e6
    return {
        "plan": response.parsed_output.model_dump(),
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                  "dollars": round(cost, 4), "model": response.model},
    }


def clean_plan(plan: dict, shots: list[dict], duration: float) -> dict:
    """Validate the plan and repair anything that would break later steps."""
    plan = SoundPlan.model_validate(plan).model_dump()
    by_shot = {s["shot"]: s for s in plan["shots"]}
    cleaned = []
    for shot in shots:
        effects = []
        for fx in by_shot.get(shot["index"], {}).get("effects", []):
            start = min(max(fx["start"], shot["start"]), shot["end"])
            end = min(max(fx["end"], start), shot["end"])
            if end - start < 0.05:
                start, end = shot["start"], shot["end"]
            text = " ".join(fx["description"].split())
            if text:
                effects.append({"start": round(start, 2), "end": round(end, 2),
                                "description": text, "loudness": fx["loudness"]})
        if not effects:
            effects = [{"start": shot["start"], "end": shot["end"],
                        "description": "soft natural room ambience", "loudness": "quiet"}]
        cleaned.append({"shot": shot["index"], "effects": effects})
    plan["shots"] = cleaned
    music = plan["music"]
    lo, hi = C.MUSIC_BPM_RANGE
    music["tempo_bpm"] = int(min(max(music["tempo_bpm"], lo), hi))
    # The real cut points are known exactly, so use them rather than Claude's estimates.
    music["cut_times"] = [s["start"] for s in shots[1:]]
    return plan


# --- Using the plan ----------------------------------------------------------

def segment_prompt(plan: dict, seg: dict) -> tuple[str, float]:
    """Text prompt and mix gain (dB) for one effects segment."""
    shot = next(s for s in plan["shots"] if s["shot"] == seg["shot"])
    hits = [fx for fx in shot["effects"] if fx["end"] > seg["start"] and fx["start"] < seg["end"]]
    if not hits:
        hits = shot["effects"]
    seen, parts = set(), []
    for fx in sorted(hits, key=lambda f: f["start"]):
        if fx["description"].lower() not in seen:
            seen.add(fx["description"].lower())
            parts.append(fx["description"].rstrip("."))
    # The loudest effect in the segment decides how prominent the segment is.
    gain = max(C.LOUDNESS_DB[fx["loudness"]] for fx in hits)
    return ", ".join(parts), gain


def pick_tempo(suggested_bpm: int, cut_times: list[float]) -> tuple[int, float]:
    """Choose a tempo near Claude's suggestion so that beats land close to the cuts.

    Returns (bpm, average distance from a cut to the nearest beat in seconds).
    """
    if not cut_times:
        return int(suggested_bpm), 0.0
    lo, hi = C.MUSIC_BPM_RANGE
    cuts = np.asarray(cut_times, dtype=float)
    best = (float("inf"), int(suggested_bpm))
    low = max(lo, int(round(suggested_bpm * (1 - C.MUSIC_BPM_SEARCH))))
    high = min(hi, int(round(suggested_bpm * (1 + C.MUSIC_BPM_SEARCH))))
    for bpm in range(low, high + 1):
        beat = 60.0 / bpm
        offsets = np.abs(cuts / beat - np.round(cuts / beat)) * beat
        # Small preference for staying close to the suggested tempo.
        score = float(offsets.mean()) + 0.0005 * abs(bpm - suggested_bpm)
        if score < best[0]:
            best = (score, bpm)
    bpm = best[1]
    beat = 60.0 / bpm
    miss = float((np.abs(cuts / beat - np.round(cuts / beat)) * beat).mean())
    return bpm, round(miss, 3)


def music_caption(music: dict) -> str:
    """The text description handed to the music model."""
    instruments = ", ".join(music["instruments"])
    return (f"{music['genre']}, {music['mood']}, instrumental background music, "
            f"{instruments}, clean mix, no vocals")
