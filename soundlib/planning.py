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


Position = Literal["left", "center", "right"]


class Effect(BaseModel):
    start: float
    end: float
    description: str
    loudness: Literal["quiet", "medium", "loud"]
    source: str = ""
    position: Position = "center"


class ShotPlan(BaseModel):
    shot: int
    effects: list[Effect]


class MusicBrief(BaseModel):
    mood: str
    genre: str
    tempo_bpm: int
    instruments: list[str]
    cut_times: list[float]


class MomentLayer(BaseModel):
    role: Literal["whoosh", "detail"]
    description: str


class Moment(BaseModel):
    time: float
    description: str
    loudness: Literal["quiet", "medium", "loud"]
    layers: list[MomentLayer] = []
    position: Position = "center"


Room = Literal["tiny enclosed space", "small room", "medium room", "large room", "hall", "outdoors"]


class SoundPlan(BaseModel):
    summary: str
    shots: list[ShotPlan]
    music: MusicBrief
    moments: list[Moment] = []
    room: Room = "medium room"
    ambience: str = ""


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
- Write each description as one plain audio caption of that single sound, the way a \
sound library labels a recording. One sound per effect: never join two sounds in one \
description.
- loudness is how prominent it should be in the mix.
- source is a short name for the thing making the sound, one to three words, lower \
case ("vacuum motor", "footsteps", "suction air"). Use EXACTLY the same source name \
every time the same thing is heard again in a later shot, so a sound that carries \
across the cuts can be made as one continuous sound.
- position is where that source is on screen: left, center or right. Use center when \
it fills the frame, is off screen, or is everywhere (air, room).
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
loudness, and its position on screen (left, center or right). Only list moments you \
can actually see between or in the frames; do not guess. \
Continuous sounds (a motor, wind, footsteps) belong in the shot effects, not here. \
At most eight; an empty list is fine.

A film sound editor builds one moment from several sounds. For each key moment, \
description is the main sound (the hit, snap or click itself), and layers lists up to \
two supporting sounds, each a single sound on its own:
- role "whoosh": movement or air just BEFORE the moment that leads into it (a swish, a \
rush of air, a wind-up, cloth moving fast).
- role "detail": a small texture just AFTER it (a rattle, bits settling, a short ring, \
a creak).
Only add a layer the picture supports, and never repeat the main sound in a layer. A \
small moment such as a switch or a tap needs no layers.

ambience is the steady background of the place, heard quietly under everything and \
continuing across the cuts: room tone, air, a distant hum, outdoor air. One short \
description of that sound alone, with no events in it ("quiet living room tone, faint \
air conditioning hum"). Do not put the scene's main sound here.

room is the kind of space the sound lives in, from the listener's point of view, used \
to pick the echo: tiny enclosed space, small room, medium room, large room, hall, or \
outdoors.

summary is one sentence saying what the video shows."""


def build_user_content(frames: list[dict], shots: list[dict], duration: float,
                       speech: list[dict], vibe: str | None = None, hint: str | None = None) -> list[dict]:
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
    if hint:   # for example the sound detail level; it outranks the usual counts
        lines.append(hint + " Where this differs from your usual limits on how many sounds to list, follow this.")
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
               vibe: str | None = None, hint: str | None = None) -> dict:
    """Send the key frames to Claude and get the plan back as a dict."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=C.PLAN_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_user_content(frames, shots, duration, speech, vibe, hint)}],
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
                source = " ".join((fx.get("source") or "").lower().split())[:40]
                effects.append({"start": round(start, 2), "end": round(end, 2),
                                "description": text, "loudness": fx["loudness"],
                                "source": source or _shorten(text, 3).lower(),
                                "position": fx.get("position") or "center"})
        if not effects:
            effects = [{"start": shot["start"], "end": shot["end"],
                        "description": "soft natural room ambience", "loudness": "quiet",
                        "source": "room ambience", "position": "center"}]
        cleaned.append({"shot": shot["index"], "effects": effects})
    plan["shots"] = cleaned
    moments = []
    for m in sorted(plan.get("moments") or [], key=lambda m: m["time"]):
        text = " ".join(m["description"].split()).rstrip(".")
        if text and 0.0 <= m["time"] < duration:
            layers, seen = [], {text.lower()}
            for layer in m.get("layers") or []:
                words = " ".join(layer["description"].split()).rstrip(".")
                if words and words.lower() not in seen and len(layers) < 2:
                    seen.add(words.lower())
                    layers.append({"role": layer["role"], "description": words})
            moments.append({"time": round(float(m["time"]), 2), "description": text,
                            "loudness": m["loudness"], "layers": layers,
                            "position": m.get("position") or "center"})
    plan["moments"] = moments[: C.MAX_MOMENTS]
    plan["ambience"] = " ".join((plan.get("ambience") or "").split()).rstrip(".")
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


def source_spans(plan: dict, duration: float) -> list[dict]:
    """The plan regrouped by what makes each sound, for making one sound at a time.

    Effects with the same source that follow straight on from each other (across cuts
    too) become one span, so a vacuum heard over five shots is one continuous sound.
    Each span is then cut into stretches the effects model can make in one go; each
    stretch keeps the description that was planned for that part of the video.
    """
    rank = {"quiet": 0, "medium": 1, "loud": 2}
    spans: list[dict] = []
    flat = sorted(({**fx, "shot": s["shot"]} for s in plan["shots"] for fx in s["effects"]),
                  key=lambda f: (f["start"], f["end"]))
    for fx in flat:
        span = next((sp for sp in reversed(spans) if sp["source"] == fx["source"]
                     and fx["start"] - sp["end"] < 0.35), None)
        if span is None:
            span = {"source": fx["source"], "start": fx["start"], "end": fx["end"], "parts": []}
            spans.append(span)
        span["end"] = max(span["end"], fx["end"])
        span["parts"].append(fx)
    longest = float(C.SOURCE_MAX_SECONDS)
    pieces = []
    for span in spans:
        if span["end"] - span["start"] < float(C.FOLEY_MIN_SEGMENT_SECONDS):   # too short for the model
            mid = (span["start"] + span["end"]) / 2
            span["start"] = max(0.0, min(mid - 0.5, duration - 1.0))
            span["end"] = min(duration, span["start"] + 1.0)
        length = span["end"] - span["start"]
        count = max(1, int(np.ceil(length / longest)))
        for n in range(count):
            a = span["start"] + n * length / count
            b = span["start"] + (n + 1) * length / count
            inside = [p for p in span["parts"] if p["end"] > a and p["start"] < b] or span["parts"]
            main = max(inside, key=lambda p: min(p["end"], b) - max(p["start"], a))
            over = float(C.FOLEY_OVERLAP_SECONDS)
            pieces.append({
                "source": span["source"], "prompt": main["description"].rstrip("."),
                "start": round(a, 3), "end": round(b, 3),
                "clip_start": round(max(0.0, a - over), 3), "clip_end": round(min(duration, b + over), 3),
                "loudness": max((p["loudness"] for p in inside), key=rank.get),
                "position": main.get("position", "center"),
                "shot": inside[0]["shot"], "shots": sorted({p["shot"] for p in inside})})
    pieces.sort(key=lambda p: (p["start"], p["source"]))
    for i, piece in enumerate(pieces):
        piece["index"] = 500 + i        # its own run of numbers, so its own random seeds
    return pieces


# --- Second, closer look: actions the first plan gave no sound ------------------

class MissedAction(BaseModel):
    time: float
    description: str
    loudness: Literal["quiet", "medium", "loud"]
    position: Position = "center"


class MissedActions(BaseModel):
    actions: list[MissedAction] = []


LOOK_SYSTEM = """You are a film sound editor doing a spotting pass on part of a short \
video. You are shown its frames in order, several per second, each with its timestamp, \
and the list of sounds that are ALREADY planned for this part.

Find every action you can see in these frames that a viewer would expect to hear and \
that has no sound of its own yet: something touching, landing, hitting, grabbing, \
snapping, dropping, bumping, sliding to a stop, opening, closing, switching, a single \
clear step or jump, a sudden start or stop of movement. Small sounds count: they are \
what make a soundtrack feel finished.

For each one give:
- time: the timestamp of the first frame where it happens (the frame of contact or of \
the sudden change), in seconds from the start of the whole video.
- description: one plain audio caption of just that one short sound, the way a sound \
library labels a recording: the source, the material, the action ("small rubber wheel \
bumping a wooden table leg"). Describe the sound, never the picture. No music, no \
speech, no brand names.
- loudness: quiet, medium or loud. Most of these are quiet or medium.
- position: left, center or right, where it is on screen.

Rules:
- Only list what you can actually see happen in the frames. Never guess or invent.
- Do not list anything already covered: a planned key moment within about half a \
second of the same action, or a continuous planned effect that is exactly this sound \
(planned footsteps already cover each step; a planned motor already covers the motor).
- Do not list continuous sounds, camera moves, cuts, text on screen or lighting changes.
- At most {limit} for this part, the clearest ones first. An empty list is right when \
nothing is missing."""


def look_windows(shots: list[dict], duration: float, longest: float = 9.0) -> list[dict]:
    """The video in stretches short enough to show Claude every sampled frame at once,
    cut on shot boundaries where possible."""
    windows, start = [], 0.0
    bounds = [s["end"] for s in shots]
    while start < duration - 0.05:
        inside = [b for b in bounds if start + 2.0 <= b <= start + longest]
        end = inside[-1] if inside else min(duration, start + longest)
        if duration - end < 2.0:
            end = duration if duration - start <= longest * 1.3 else end
        windows.append({"start": round(start, 3), "end": round(min(end, duration), 3)})
        start = end
    return windows


def _planned_in(plan: dict, window: dict) -> list[str]:
    lines = []
    for shot in plan["shots"]:
        for fx in shot["effects"]:
            if fx["end"] > window["start"] and fx["start"] < window["end"]:
                lines.append(f"- {fx['start']:.2f}s to {fx['end']:.2f}s (continuous): {fx['description']}")
    for m in plan.get("moments", []):
        if window["start"] - 0.5 <= m["time"] <= window["end"] + 0.5:
            lines.append(f"- {m['time']:.2f}s (key moment): {m['description']}")
    return lines or ["- nothing yet"]


def _effort(effort: str | None) -> dict:
    return {"output_config": {"effort": effort}} if effort else {}


def look_model() -> str:
    return C.LOOK_MODEL or C.PLAN_MODEL


def _usage(responses: list, model: str | None = None) -> dict:
    tokens_in = sum(r.usage.input_tokens for r in responses)
    tokens_out = sum(r.usage.output_tokens for r in responses)
    price = C.CLAUDE_MODEL_DOLLARS_PER_MTOK.get(model or "", C.CLAUDE_DOLLARS_PER_MTOK)
    cost = (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6
    return {"input_tokens": tokens_in, "output_tokens": tokens_out, "dollars": round(cost, 4),
            "model": responses[0].model if responses else C.PLAN_MODEL, "requests": len(responses)}


def _image(jpeg: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(jpeg).decode("ascii")}}


def ask_missed(frames: list[dict], windows: list[dict], duration: float, plan: dict,
               model: str | None = None, most: int | None = None, effort: str | None = None) -> dict:
    """The second look. `frames` items: {time, jpeg}. Runs inside Modal, like ask_claude."""
    from concurrent.futures import ThreadPoolExecutor

    import anthropic

    client = anthropic.Anthropic()
    model = model or look_model()
    limit = max(2, int(np.ceil(1.5 * int(most or C.SECOND_LOOK_MAX_EXTRA) / max(1, len(windows)))))

    def one(window: dict):
        shown = [f for f in frames if window["start"] - 1e-3 <= f["time"] < window["end"]][:90]
        if len(shown) < 2:
            return None
        content: list[dict] = [{"type": "text", "text": (
            f"The whole video is {duration:.2f} seconds long. This part runs from "
            f"{window['start']:.2f}s to {window['end']:.2f}s.\nSounds already planned here:\n"
            + "\n".join(_planned_in(plan, window)))}]
        for f in shown:
            content.append({"type": "text", "text": f"{f['time']:.2f}s:"})
            content.append(_image(f["jpeg"]))
        content.append({"type": "text", "text": "List the actions here that still need a sound."})
        return client.messages.parse(model=model, max_tokens=4000,
                                     system=LOOK_SYSTEM.format(limit=limit),
                                     messages=[{"role": "user", "content": content}],
                                     output_format=MissedActions, **_effort(effort))

    with ThreadPoolExecutor(4) as pool:
        responses = [r for r in pool.map(one, windows) if r is not None]
    actions = []
    for window, response in zip(windows, responses):
        if response.parsed_output is not None:
            actions += [a.model_dump() for a in response.parsed_output.actions]
    return {"actions": actions, "usage": _usage(responses, model)}


def add_missed(plan: dict, actions: list[dict], duration: float) -> list[dict]:
    """Put the second look's finds into the plan as extra key moments (after the planned
    ones, so those keep their place and their sound). Returns what was added."""
    gap, taken = float(C.SECOND_LOOK_MIN_GAP), [m["time"] for m in plan["moments"]]
    added = []
    for a in sorted(actions, key=lambda a: a["time"]):
        text = " ".join(str(a.get("description", "")).split()).rstrip(".")
        time = round(float(a.get("time", -1.0)), 2)
        if not text or not 0.0 <= time < duration - 0.1 or any(abs(time - t) < gap for t in taken):
            continue
        if len(added) >= int(C.SECOND_LOOK_MAX_EXTRA):
            break
        taken.append(time)
        added.append({"time": time, "description": text, "loudness": a.get("loudness", "medium"),
                      "layers": [], "position": a.get("position") or "center", "found": "second look"})
    plan["moments"] = plan["moments"] + added
    return added


# --- Frame-exact timing of key moments --------------------------------------------

class PinnedFrame(BaseModel):
    visible: bool
    time: float
    what_changes: str


PIN_SYSTEM = """You are a film sound editor syncing one sound effect to picture, frame \
by frame. You are shown every consecutive frame of a short stretch of video, each with \
its timestamp, and told which sound is to be placed there.

Find the single frame where that sound should START: the first frame in which the \
action that makes the sound has happened (first contact of an impact, the first frame \
a thing is broken, released, switched or landed, the first frame of a sudden movement). \
Compare neighbouring frames carefully; the answer is where the picture changes for \
that action, not simply the middle of the stretch.

Answer with visible (true if you can see that action in these frames; false if it is \
not there or you cannot tell), time (the timestamp of that frame, copied exactly from \
the labels), and what_changes (a few words on what is different in that frame)."""


def ask_pins(bursts: list[dict], model: str | None = None, effort: str | None = None) -> dict:
    """Frame-exact times. bursts: {index, description, planned, frames: [{time, jpeg}]}.
    Returns {"pins": {index: {time, visible, what_changes}}, "usage"}. Runs inside Modal."""
    from concurrent.futures import ThreadPoolExecutor

    import anthropic

    client = anthropic.Anthropic()
    model = model or look_model()

    def one(burst: dict):
        content: list[dict] = [{"type": "text", "text": (
            f"Sound to place: {burst['description']}. It was roughly estimated at "
            f"{burst['planned']:.2f}s from stills; that estimate can be off by several frames "
            "either way.")}]
        for f in burst["frames"][:90]:
            content.append({"type": "text", "text": f"{f['time']:.3f}s:"})
            content.append(_image(f["jpeg"]))
        content.append({"type": "text", "text": "Which frame does this sound start on?"})
        return client.messages.parse(model=model, max_tokens=2000, system=PIN_SYSTEM,
                                     messages=[{"role": "user", "content": content}],
                                     output_format=PinnedFrame, **_effort(effort))

    with ThreadPoolExecutor(4) as pool:
        responses = list(pool.map(one, bursts))
    pins = {}
    for burst, response in zip(bursts, responses):
        if response.parsed_output is not None:
            pins[burst["index"]] = response.parsed_output.model_dump()
    return {"pins": pins, "usage": _usage(responses, model)}


def apply_pins(plan: dict, pins: dict, bursts: list[dict]) -> list[dict]:
    """Move each key moment onto the frame Claude picked. Returns what moved, for the report."""
    moved = []
    for burst in bursts:
        pin = pins.get(burst["index"]) or pins.get(str(burst["index"]))
        times = [f["time"] for f in burst["frames"]]
        if not pin or not pin.get("visible") or not times:
            continue
        exact = min(times, key=lambda t: abs(t - float(pin["time"])))   # always a real frame
        if abs(exact - float(pin["time"])) > 0.06:
            continue
        moment = plan["moments"][burst["index"]]
        moved.append({"description": moment["description"], "planned": moment["time"], "frame": exact,
                      "moved_seconds": round(exact - moment["time"], 3), "what_changes": pin.get("what_changes", "")})
        moment["planned_time"], moment["time"], moment["pinned"] = moment["time"], round(exact, 3), True
    return moved


# --- A sound asked for in the editor -------------------------------------------------

class SoundIdea(BaseModel):
    name: str
    prompt: str


class SoundIdeas(BaseModel):
    understood_as: str
    ideas: list[SoundIdea]


ADDED_SYSTEM = """You are a Foley artist. Someone watching their video has stopped on one \
frame and typed, in their own rough words, a sound they want to hear starting there. \
They have a clear sound in their head but are not good at describing sounds. You are \
shown the frames around that instant.

Give them THREE different sounds to choose from, the way Foley really works: the sound \
that sells an action on screen is often not a recording of the real thing. Celery \
snapped for a breaking bone; a leather glove flapped for wings; a popping cork or a \
ripped strip of velcro for something tearing free.

1. The first idea is the literal sound they asked for, made precise and fitted to what \
is on screen (the object, its material and size, what it is doing, the space).
2. and 3. are stand-ins: DIFFERENT real-world sounds, made by something else, that \
would play the part convincingly or more excitingly at that instant. Pick sounds that \
are common enough to exist in a library of real recordings, and that differ clearly \
from each other and from the first (a different source, texture or character), so the \
three are a real choice. Match the feel of the video (realistic, cartoon, toy world, \
slapstick, dramatic).

For each idea give: name, 2 to 5 plain words someone can read at a glance ("Carpet \
tearing", "Velcro rip", "Cork pop"); and prompt, ONE description of the sound itself, 6 \
to 16 words, in the plain concrete words a sound library would use: the thing, the \
material, the action, the character (sharp, soft, hollow, wet, rubbery, deep, tiny), \
and how it starts and ends when that matters. A sound, not a story or a picture. One \
event that fits the length given. No music, no speech and no background ambience \
unless they asked for it. Keep what they asked for as the job the sound has to do.

When you are told which ideas were already tried, give three NEW ones that differ \
clearly from all of those, and if they said how it should change ("punchier", "more of \
a pop than a bang"), follow that in all three. Then the first need not be literal.

Also give understood_as: a few plain words on what you took them to mean."""


def ask_added(frames: list[dict], words: str, seconds: float, summary: str = "",
              previous: list[str] | None = None, tweak: str | None = None, model: str | None = None) -> dict:
    """Three ideas for a sound asked for in the editor. frames: {time, jpeg} around the
    chosen instant. Returns {"ideas": [{name, prompt}], "understood_as", "usage"}. Runs
    inside Modal."""
    import anthropic

    model = model or look_model()
    content: list[dict] = []
    if summary:
        content.append({"type": "text", "text": f"What the video is: {summary}"})
    for f in frames[:6]:
        content.append({"type": "text", "text": f"{f['time']:.2f}s" + (" (the chosen instant)" if f.get("chosen") else "") + ":"})
        content.append(_image(f["jpeg"]))
    ask = f'They typed: "{words}". The sound will be about {seconds:g} seconds long.'
    if previous:
        ask += " Already tried (give three different ones): " + "; ".join(f'"{p}"' for p in previous[:12]) + "."
    if tweak:
        ask += f' They want it changed like this: "{tweak}".'
    content.append({"type": "text", "text": ask})
    response = anthropic.Anthropic().messages.parse(
        model=model, max_tokens=1500, system=ADDED_SYSTEM,
        messages=[{"role": "user", "content": content}], output_format=SoundIdeas)
    made = response.parsed_output
    ideas = [{"name": " ".join(i.name.split())[:40], "prompt": " ".join(i.prompt.split())[:300]}
             for i in (made.ideas if made else []) if i.prompt.strip()][:3]
    return {"ideas": ideas or [{"name": words[:40], "prompt": words}],
            "understood_as": made.understood_as if made else "", "usage": _usage([response], model)}
