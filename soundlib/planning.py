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


class Moment(BaseModel):
    time: float
    description: str
    loudness: Literal["quiet", "medium", "loud"]


Room = Literal["tiny enclosed space", "small room", "medium room", "large room", "hall", "outdoors"]


class SoundPlan(BaseModel):
    summary: str
    shots: list[ShotPlan]
    music: MusicBrief
    moments: list[Moment] = []
    room: Room = "medium room"


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
- mood, genre and instruments for an instrumental background track that leaves room \
for effects and any voice.
- Choose the mood from what is at stake in the story, not from how the picture looks. \
First decide what the viewer should feel: if a character is being chased, trapped, \
threatened or in danger, the music is suspenseful and tense, even when the characters \
are cute, small, animated or colourful. Do not default to playful, whimsical, quirky or \
comedic music because the visuals are stylised; use those only when the action itself \
is clearly a joke with nothing at stake. Say the mood plainly in a few words a music \
model understands ("tense, suspenseful, ominous, building dread").
- If you are given a requested vibe, the music brief must follow it.
- tempo_bpm between 60 and 180 that fits the pace of the cuts.
- cut_times: the shot changes (in seconds) the music should hit. Use the shot boundaries \
you were given; leave it empty if there is only one shot.

Then list the key moments: single instants where something visibly happens and a \
viewer would expect to hear it right then (a grab, an impact, a snap, a door slam, a \
switch flipping, something landing). For each: time in seconds (the instant it happens, \
judged from the frame timestamps), a description of just that one short sound, and its \
loudness. Only list moments you can actually see between or in the frames; do not guess. \
Continuous sounds (a motor, wind, footsteps) belong in the shot effects, not here. \
At most eight; an empty list is fine.

room is the kind of space the sound lives in, from the listener's point of view, used \
to pick the echo: tiny enclosed space, small room, medium room, large room, hall, or \
outdoors.

summary is one sentence saying what the video shows."""


def build_user_content(frames: list[dict], shots: list[dict], duration: float,
                       speech: list[dict], vibe: str | None = None) -> list[dict]:
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
    if vibe:
        lines.append(f"Requested vibe for the music: {vibe}.")
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for f in frames:
        content.append({"type": "text", "text": f"Shot {f['shot']}, frame at {f['time']:.2f}s:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(f["jpeg"]).decode("ascii"),
        }})
    content.append({"type": "text", "text": "Plan the sound for this video."})
    return content


def ask_claude(frames: list[dict], shots: list[dict], duration: float, speech: list[dict],
               vibe: str | None = None) -> dict:
    """Send the key frames to Claude and get the plan back as a dict."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=C.PLAN_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_user_content(frames, shots, duration, speech, vibe)}],
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


# --- Cue sheet: scoring the music to the picture --------------------------------

class CueSection(BaseModel):
    start: float
    end: float
    intensity: Literal[1, 2, 3, 4, 5]
    description: str


class CueHit(BaseModel):
    time: float
    strength: Literal["small", "big"]
    what: str


class CueSheet(BaseModel):
    key: str
    sections: list[CueSection]
    hits: list[CueHit]


CUE_SYSTEM = """You are a film composer spotting a short video: deciding how the music \
should follow the picture. You are shown still frames with timestamps, the shot list, \
and the sound plan already made for it (summary, music brief, key moments).

Write a cue sheet:
- key: one musical key for the whole score that suits the mood, like "D minor".
- sections: split the video into 2 to 5 story beats that cover it from start to end \
with no gaps. Each section starts and ends on a shot boundary from the list, and is at \
least 4 seconds long. For each give intensity from 1 (almost nothing, a held note or \
quiet pulse) to 5 (full climax), following the tension of the story, and a description \
of what the music does there in a few words a music model understands (for example \
"sparse low drone, soft ticking pulse" or "driving staccato strings, pounding drums"). \
Keep the instruments of the music brief so the sections sound like one score.
- hits: the instants the music should mark, at most six, chosen from the key moments \
or the most dramatic cuts. time in seconds; strength "big" for the one or two most \
important (the music drops out just before and lands on it), otherwise "small"; what is \
a few words saying what happens."""


def ask_cues(frames: list[dict], shots: list[dict], duration: float, plan: dict) -> dict:
    """Ask Claude for the cue sheet. Runs inside Modal, like ask_claude."""
    import json

    import anthropic

    content = build_user_content(frames, shots, duration, [])[:-1]
    brief = {"summary": plan.get("summary"), "music": plan.get("music"),
             "key_moments": plan.get("moments", [])}
    content.append({"type": "text", "text": "The sound plan so far:\n" + json.dumps(brief, indent=1)
                    + "\n\nWrite the cue sheet for this video."})
    response = anthropic.Anthropic().messages.parse(
        model=C.PLAN_MODEL, max_tokens=8000, system=CUE_SYSTEM,
        messages=[{"role": "user", "content": content}], output_format=CueSheet)
    if response.parsed_output is None:
        raise RuntimeError(f"Claude returned no cue sheet (stop reason: {response.stop_reason}).")
    usage = response.usage
    cost = (usage.input_tokens * C.CLAUDE_DOLLARS_PER_MTOK["input"]
            + usage.output_tokens * C.CLAUDE_DOLLARS_PER_MTOK["output"]) / 1e6
    return {"cues": response.parsed_output.model_dump(),
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                      "dollars": round(cost, 4), "model": response.model}}


def clean_cues(cues: dict, shots: list[dict], duration: float) -> dict:
    """Make the cue sheet safe to use: sections tile the video and sit on real cuts."""
    cues = CueSheet.model_validate(cues).model_dump()
    bounds = [0.0] + [s["start"] for s in shots[1:]] + [duration]
    snap = lambda t: min(bounds, key=lambda b: abs(b - t))
    sections = []
    for sec in sorted(cues["sections"], key=lambda s: s["start"]):
        start = 0.0 if not sections else sections[-1]["end"]
        end = max(snap(sec["end"]), start)
        if end - start < 3.0 and sections:      # too short to be its own piece of music
            sections[-1]["end"] = max(sections[-1]["end"], end)
            continue
        sections.append({"start": round(start, 3), "end": round(end, 3),
                         "intensity": int(sec["intensity"]),
                         "description": " ".join(sec["description"].split()).rstrip(".")})
    if not sections:
        sections = [{"start": 0.0, "end": duration, "intensity": 3, "description": ""}]
    sections[-1]["end"] = duration
    sections = [s for s in sections if s["end"] - s["start"] > 0.5] or sections[:1]
    sections[0]["start"], sections[-1]["end"] = 0.0, duration
    hits = [{"time": round(float(h["time"]), 2), "strength": h["strength"], "what": h["what"]}
            for h in sorted(cues["hits"], key=lambda h: h["time"]) if 0.3 < h["time"] < duration - 0.2]
    return {"key": cues["key"].strip(), "sections": sections, "hits": hits[:6]}


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
    moments = []
    for m in sorted(plan.get("moments") or [], key=lambda m: m["time"]):
        text = " ".join(m["description"].split()).rstrip(".")
        if text and 0.0 <= m["time"] < duration:
            moments.append({"time": round(float(m["time"]), 2), "description": text,
                            "loudness": m["loudness"]})
    plan["moments"] = moments[: C.MAX_MOMENTS]
    music = plan["music"]
    lo, hi = C.MUSIC_BPM_RANGE
    music["tempo_bpm"] = int(min(max(music["tempo_bpm"], lo), hi))
    # The real cut points are known exactly, so use them rather than Claude's estimates.
    music["cut_times"] = [s["start"] for s in shots[1:]]
    return plan


# --- Using the plan ----------------------------------------------------------

def _shorten(text: str, words: int = 6) -> str:
    return " ".join(text.split(",")[0].split()[:words])


def segment_prompt(plan: dict, seg: dict) -> tuple[str, float]:
    """Text prompt and mix gain (dB) for one effects segment.

    A segment normally covers one shot; with FOLEY_GROUP_SECONDS it can span several.
    SFX_PROMPT_STYLE decides the wording: every planned sound ("detailed"), only the
    most important one ("main"), or every sound cut to a few words ("short").
    """
    wanted = set(seg.get("shots") or [seg["shot"]])
    effects = [fx for s in plan["shots"] if s["shot"] in wanted for fx in s["effects"]]
    hits = [fx for fx in effects if fx["end"] > seg["start"] and fx["start"] < seg["end"]]
    if not hits:
        hits = effects
    hits = sorted(hits, key=lambda f: f["start"])
    # The loudest effect in the segment decides how prominent the segment is.
    gain = max(C.LOUDNESS_DB[fx["loudness"]] for fx in hits)
    style = C.SFX_PROMPT_STYLE
    if style == "main":
        rank = {"loud": 2, "medium": 1, "quiet": 0}
        # One main sound per shot, so a clip that spans several shots still gets each
        # shot's key sound, in order.
        parts = []
        for shot in plan["shots"]:
            mine = [fx for fx in shot["effects"] if any(fx is h for h in hits)]
            if shot["shot"] in wanted and mine:
                best = max(mine, key=lambda f: (rank[f["loudness"]], f["end"] - f["start"]))
                text = best["description"].rstrip(".")
                if text not in parts:
                    parts.append(text)
        return ", ".join(parts), gain
    seen, parts = set(), []
    for fx in hits:
        text = fx["description"].rstrip(".")
        if style == "short":
            text = _shorten(text)
        if text.lower() not in seen:
            seen.add(text.lower())
            parts.append(text)
    return ", ".join(parts), gain


def pick_tempo(suggested_bpm: int, cut_times: list[float]) -> tuple[int, float]:
    """Choose a tempo near Claude's suggestion so that beats land close to the cuts.

    MUSIC_TEMPO_RULE: "match_cuts" (search near the suggestion), "as_planned" (use it
    exactly) or "half_time" (halve it first, for a slower feel, then match cuts).

    Returns (bpm, average distance from a cut to the nearest beat in seconds).
    """
    lo, hi = C.MUSIC_BPM_RANGE
    rule = C.MUSIC_TEMPO_RULE
    if rule == "half_time":
        suggested_bpm = int(max(lo, round(suggested_bpm / 2)))
    if not cut_times:
        return int(suggested_bpm), 0.0
    cuts = np.asarray(cut_times, dtype=float)
    if rule == "as_planned":
        beat = 60.0 / suggested_bpm
        miss = float((np.abs(cuts / beat - np.round(cuts / beat)) * beat).mean())
        return int(suggested_bpm), round(miss, 3)
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


def music_caption(music: dict, flavour: str = "") -> str:
    """The text description handed to the music model, built from MUSIC_PROMPT_TEMPLATE."""
    text = C.MUSIC_PROMPT_TEMPLATE.format(
        genre=music["genre"], mood=music["mood"],
        instruments=", ".join(music["instruments"]), flavour=flavour)
    if flavour and "{flavour}" not in C.MUSIC_PROMPT_TEMPLATE:
        text = f"{text}, {flavour}"
    return ", ".join(part.strip() for part in text.split(",") if part.strip())


SECTION_FEEL = ["very sparse and quiet, barely there", "restrained, light and steady",
                "building, more movement", "driving and intense", "full climax, everything playing"]


def section_caption(music: dict, flavour: str, section: dict) -> str:
    """The description for one section of the score. With SECTION_INSTRUMENTS "shared"
    every section keeps the main brief's instruments and only its energy changes."""
    extra = SECTION_FEEL[section["intensity"] - 1] if C.SECTION_INSTRUMENTS == "shared" \
        else section["description"]
    return music_caption(music, ", ".join(x for x in (flavour, extra) if x))


def settle_sections(sections: list[dict], bpm: int, duration: float) -> list[dict]:
    """Tidy the cue sheet's sections for composing: fold very short ones into the next
    (SECTION_MIN_SECONDS) and move the changes onto bar lines (SECTION_SNAP "bar")."""
    secs = [dict(s) for s in sections]
    least = float(C.SECTION_MIN_SECONDS)
    i = 0
    while least > 0 and len(secs) > 1 and i < len(secs):
        if secs[i]["end"] - secs[i]["start"] >= least:
            i += 1
        elif i + 1 < len(secs):
            secs[i + 1]["start"] = secs[i]["start"]
            del secs[i]
        else:
            secs[i - 1]["end"] = secs[i]["end"]
            del secs[i]
    if C.SECTION_SNAP == "bar":
        bar = 240.0 / bpm
        for a, b in zip(secs, secs[1:]):
            t = round(b["start"] / bar) * bar
            if a["start"] + bar <= t <= b["end"] - bar:
                a["end"] = b["start"] = round(t, 3)
    return secs
