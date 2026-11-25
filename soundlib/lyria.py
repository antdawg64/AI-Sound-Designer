"""Google Lyria (on Vertex AI) as a second music model, next to ACE-Step.

Used when settings.json has music_provider "lyria". Lyria composes from a text brief
only: it cannot rewrite one section of a finished track the way ACE-Step's repaint does.
So the story-following is done in the brief itself: the cue sheet's sections are written
out with their times (see story_prompt) and Lyria is asked for one continuous piece.

Sign-in: a service account key kept in the Modal secret named by lyria_secret_name
(GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_CLOUD_PROJECT). Without those, the Google
sign-in on this machine is used (scripts/google_login.py).

Licence notes (read 2026-11-25):
  - Lyria 2 (lyria-002) is generally available, instrumental only, 48 kHz, about 30 s
    per clip. Every clip carries Google's inaudible SynthID watermark.
  - Lyria 3 (lyria-3-clip-preview: 30 s clips; lyria-3-pro-preview: whole pieces of a
    requested length, follows timed sections) is in PREVIEW on Vertex AI: pre-release
    terms, can change or be withdrawn, can sing unless told not to. Returns MP3.
  - Lyria is NOT on Google's list of services covered by its generative-AI copyright
    indemnity (cloud.google.com/terms/generative-ai-indemnified-services, version of
    2026-07-20 lists Codey, Gemini, Imagen, PaLM and Veo). Preview models never are.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request

import numpy as np

from . import config as C
from .planning import SECTION_FEEL


# --- The brief -----------------------------------------------------------------------

def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(round(t % 60)):02d}"


def story_prompt(caption: str, bpm: int, keyscale: str, sections: list[dict], duration: float) -> str:
    """The text handed to Lyria. With lyria_prompt_style "story" the cue sheet's sections
    are spelled out with their times, so the one piece follows the picture.

    Lyria 3 reads "[0:00 - 0:10] ..." time ranges and a total length and keeps to them
    (checked 2026-11-25: asked for 28 s in three parts, it returned 27.6 s with the
    changes on the nearest bar lines). Lyria 2 only takes them as a description."""
    three = not C.LYRIA_MODEL.startswith("lyria-00")
    parts = [caption, f"{bpm} BPM"]
    if keyscale:
        parts.append(keyscale)
    text = ", ".join(parts) + "."
    if three:
        text += f" Instrumental only, no vocals. Total length exactly {duration:.0f} seconds."
    if C.LYRIA_PROMPT_STYLE == "story" and len(sections) > 1:
        lines = []
        for sec in sections:
            feel = SECTION_FEEL[int(sec["intensity"]) - 1]
            what = f" ({sec['description']})" if sec.get("description") else ""
            when = f"[{_clock(sec['start'])} - {_clock(sec['end'])}]" if three \
                else f"from {sec['start']:.0f}s to {sec['end']:.0f}s:"
            lines.append(f"{when} {feel}{what}")
        text += (" One continuous piece with the same instruments throughout, scored to picture. "
                 + ("" if three else "It changes with the story: ") + (" " if three else "; ").join(lines) + ".")
    if three:
        text += " It ends cleanly on the last beat."
    elif duration < clip_seconds(C.LYRIA_MODEL):
        text += f" The piece resolves by {duration:.0f} seconds."
    return text


def clip_seconds(model: str) -> float:
    """How much music one request gives. Lyria 2 always returns about 30 s."""
    return float(C.LYRIA_CLIP_SECONDS.get(model, C.LYRIA_CLIP_SECONDS["default"]))


def clip_dollars(model: str) -> float:
    return float(C.LYRIA_DOLLARS_PER_CLIP.get(model, C.LYRIA_DOLLARS_PER_CLIP["default"]))


# --- Talking to Google ------------------------------------------------------------------

def _credentials() -> tuple[str, str]:
    """(access token, project id)."""
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if key:
        info = json.loads(key)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or info["project_id"]
    else:
        creds, project = google.auth.default(scopes=scopes)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or project
    if not project:
        raise RuntimeError("No Google Cloud project set (GOOGLE_CLOUD_PROJECT)")
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token, project


def _post(url: str, body: dict, token: str, project: str) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "x-goog-user-project": project})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Lyria refused the request ({e.code}): {e.read().decode()[:600]}") from None


def _find_audio(node) -> bytes | None:
    """The first piece of base64 audio anywhere in a response."""
    if isinstance(node, dict):
        for name in ("audioContent", "bytesBase64Encoded"):
            if isinstance(node.get(name), str):
                return base64.b64decode(node[name])
        mime = node.get("mimeType") or node.get("mime_type") or ""
        if str(mime).startswith("audio/") and isinstance(node.get("data"), str):
            return base64.b64decode(node["data"])
        node = list(node.values())
    if isinstance(node, list):
        for item in node:
            found = _find_audio(item)
            if found:
                return found
    return None


def request_clip(prompt: str, negative: str, seed: int, model: str, creds: tuple[str, str]) -> tuple[np.ndarray, int]:
    """One request to Lyria. Returns (samples x channels float32, sample rate)."""
    import soundfile as sf

    token, project = creds
    loc = C.LYRIA_LOCATION
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    base = f"https://{host}/v1/projects/{project}/locations/{loc}"
    if model.startswith("lyria-00"):      # Lyria 2: the "predict" call
        instance = {"prompt": prompt, "seed": int(seed)}
        if negative:
            instance["negative_prompt"] = negative
        reply = _post(f"{base}/publishers/google/models/{model}:predict",
                      {"instances": [instance], "parameters": {}}, token, project)
    else:                                  # Lyria 3 and later: the "interactions" call
        reply = _post(f"https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/global/interactions",
                      {"model": model, "input": prompt + (f" Avoid: {negative}." if negative else "")},
                      token, project)
    audio = _find_audio(reply)
    if not audio:
        raise RuntimeError(f"Lyria returned no audio: {json.dumps(reply)[:400]}")
    return sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)


def compose(prompt: str, negative: str, seconds: float, seed: int, model: str) -> dict:
    """Music of at least `seconds`. One clip when it fits; otherwise several clips from
    the same brief, blended into each other (Lyria cannot continue a clip)."""
    creds = _credentials()
    length = clip_seconds(model)
    blend = float(C.LYRIA_JOIN_SECONDS)
    count = 1 if seconds <= length else int(np.ceil((seconds - blend) / (length - blend)))
    clips, sr = [], None
    for n in range(count):
        data, sr = request_clip(prompt, negative, seed + n, model, creds)
        clips.append(data)
    out = clips[0]
    fade = int(blend * sr)
    for nxt in clips[1:]:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        mixed = out[-fade:] * np.cos(ramp * np.pi / 2) + nxt[:fade] * np.sin(ramp * np.pi / 2)
        out = np.concatenate([out[:-fade], mixed, nxt[fade:]])
    return {"audio": out, "sample_rate": sr, "model": model, "clips": count,
            "dollars": round(count * clip_dollars(model), 4)}
