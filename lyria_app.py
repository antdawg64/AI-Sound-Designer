"""Google Lyria music, as its own small Modal app (no GPU: Google does the composing).

Kept apart from modal_app.py so it can be deployed without touching the live effects
and music models:   modal deploy lyria_app.py

Needs the Modal secret named in settings.json (lyria_secret_name, "google-cloud") with
GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_CLOUD_PROJECT. See soundlib/lyria.py.
"""

import io

import modal

from soundlib import config as C

APP_NAME = "ai-sound-designer-lyria"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install("google-auth>=2.30", "requests", "numpy", "soundfile", "pydantic>=2")
    .env({"SOUND_SETTINGS": "/root/settings.json"})
    .add_local_python_source("soundlib")
    .add_local_file(C.SETTINGS_FILE, "/root/settings.json")
)


CHUNK = 1_000_000  # audio is streamed back in pieces of this many bytes (as in modal_app.py)


@app.function(image=image, secrets=[modal.Secret.from_name(C.LYRIA_SECRET_NAME)], timeout=900)
def compose(prompt: str, negative: str, seconds: float, seed: int, model: str,
            settings: dict | None = None):
    """Yields a header {"model", "clips", "dollars", "bytes"}, then the music as FLAC
    bytes in small pieces. (One big return value would travel through Modal's file
    storage, which restricted networks cannot reach.)"""
    import soundfile as sf

    from soundlib import lyria

    if settings:
        C.override(settings)
    result = lyria.compose(prompt, negative, seconds, seed, model)
    buf = io.BytesIO()
    sf.write(buf, result.pop("audio"), result.pop("sample_rate"), format="FLAC", subtype="PCM_16")
    data = buf.getvalue()
    yield {**result, "bytes": len(data)}
    for i in range(0, len(data), CHUNK):
        yield data[i:i + CHUNK]
