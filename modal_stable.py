"""Stable Audio Open 1.0 on Modal: sound effects from text alone (no video).

A second way to make the effects (settings.json "sfx_engine": "stable_audio"). Claude
still watches the video and writes the timed sound plan; this model then makes every
planned sound from its description, and the sounds are placed at the planned times.

It is its own Modal app so the main one (modal_app.py) deploys without it:

    modal deploy modal_stable.py        (python make_sound.py --deploy does it too)

Needs the Modal secret "huggingface-secret" (key HF_TOKEN; name in settings "hf_secret_name") from an account that has accepted
the model's terms at https://huggingface.co/stabilityai/stable-audio-open-1.0

Licence (read 2026-11-25, LICENSE.md in that repo): Stability AI Community License.
Commercial use is free worldwide while yearly revenue is under US $1,000,000. Before
commercial use: register at https://stability.ai/community-license, and show "Powered
by Stability AI" on the site.
"""

import io
import os
import time

import modal

from soundlib import config as C

APP_NAME = (os.environ.get("SOUND_APP") or "ai-sound-designer") + "-stable"
app = modal.App(APP_NAME)

weights = modal.Volume.from_name("ai-sound-designer-weights", create_if_missing=True)
WEIGHTS = "/weights"
MODEL = "stabilityai/stable-audio-open-1.0"
MAX_SECONDS = 47.0
CHUNK = 900_000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("diffusers==0.32.2", "transformers==4.47.1", "accelerate", "torchsde",
                 "sentencepiece", "protobuf", "soundfile", "numpy<2", "huggingface_hub")
    .env({"HF_HOME": f"{WEIGHTS}/hf", "SOUND_SETTINGS": "/root/settings.json"})
    .add_local_python_source("soundlib")
    .add_local_file(C.SETTINGS_FILE, "/root/settings.json")
)


@app.cls(image=image, gpu=C.GPU_TYPE, volumes={WEIGHTS: weights}, timeout=1800,
         scaledown_window=60, max_containers=1,
         secrets=[modal.Secret.from_name(C.HF_SECRET_NAME)])
class StableAudio:
    @modal.enter()
    def load(self):
        started = time.time()
        import torch
        from diffusers import StableAudioPipeline

        self.pipe = StableAudioPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
        weights.commit()   # the first run downloads the model into the shared volume
        self.load_seconds = time.time() - started
        self.reported_load = False

    @modal.method()
    def generate(self, items: list[dict], steps: int = 100, guidance: float = 7.0,
                 neg_prompt: str = "", batch: int = 8):
        """Make one sound per item ({prompt, seconds, seed}). Sounds of similar length
        are made together. Yields a header, then for each sound {"i", "part"} pieces of
        FLAC audio (in the order made, not the order asked)."""
        import numpy as np
        import soundfile as sf
        import torch

        started = time.time()
        rate = int(self.pipe.vae.sampling_rate)
        yield {"count": len(items), "sample_rate": rate}
        # Group by length, rounded up to a few sizes, so each batch shares one length.
        sizes = (2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, MAX_SECONDS)
        groups: dict[float, list[int]] = {}
        for i, item in enumerate(items):
            want = min(float(item["seconds"]), MAX_SECONDS)
            groups.setdefault(next(s for s in sizes if s >= want), []).append(i)
        for size, members in sorted(groups.items()):
            for at in range(0, len(members), int(batch)):
                chunk = members[at:at + int(batch)]
                generator = torch.Generator("cuda").manual_seed(int(items[chunk[0]]["seed"]))
                with torch.no_grad():
                    audios = self.pipe(
                        prompt=[items[i]["prompt"] for i in chunk],
                        negative_prompt=[neg_prompt] * len(chunk) if neg_prompt else None,
                        num_inference_steps=int(steps), guidance_scale=float(guidance),
                        audio_end_in_s=float(size), generator=generator).audios
                for i, audio in zip(chunk, audios):
                    wave = np.clip(audio.T.float().cpu().numpy(), -1.0, 1.0)
                    wave = wave[: int(min(float(items[i]["seconds"]), MAX_SECONDS) * rate)]
                    buf = io.BytesIO()
                    sf.write(buf, wave, rate, format="FLAC", subtype="PCM_16")
                    data = buf.getvalue()
                    for k in range(0, len(data), CHUNK):
                        yield {"i": i, "part": data[k:k + CHUNK]}
        meta = {"done": True, "gpu_seconds": time.time() - started, "load_seconds": 0.0,
                "gpu": C.GPU_TYPE}
        if not self.reported_load:
            meta["load_seconds"], self.reported_load = self.load_seconds, True
        yield meta
