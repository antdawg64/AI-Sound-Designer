"""The cloud side of AI Sound Designer, running on Modal.

Three pieces:
  plan_sound  - asks Claude for the sound plan (CPU; reads the `anthropic` Modal secret)
  Foley       - HunyuanVideo-Foley: synced sound effects from video + text (GPU)
  Music       - ACE-Step 1.5: instrumental music from a text brief (GPU)

Model weights live in a Modal volume, so only the first run downloads them.

Deploy with:  python make_sound.py --deploy     (or: modal deploy modal_app.py)

Licences (read 2026-11-24):
  HunyuanVideo-Foley - Tencent Hunyuan Community License. Not licensed for use in the
                       EU, the UK or South Korea. The web app (Phase 3) must enforce this.
  ACE-Step 1.5       - MIT.
"""

from __future__ import annotations

import io
import os
import time

import modal

from soundlib import config as C

APP_NAME = "ai-sound-designer"
app = modal.App(APP_NAME)

weights = modal.Volume.from_name("ai-sound-designer-weights", create_if_missing=True)
WEIGHTS = "/weights"
CHUNK = 1_000_000  # audio is streamed back in pieces of this many bytes

FOLEY_REPO = "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley"
FOLEY_DIR = "/opt/HunyuanVideo-Foley"
ACE_REPO = "https://github.com/ace-step/ACE-Step-1.5"
ACE_DIR = "/opt/ACE-Step-1.5"

# --- Images ------------------------------------------------------------------

plan_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("anthropic>=2.14", "pydantic>=2", "numpy")
    .add_local_python_source("soundlib")
)

foley_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy==1.26.4", "scipy", "diffusers", "timm", "accelerate",
        "git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2",
        "sentencepiece", "git+https://github.com/descriptinc/audiotools",
        "pillow", "av", "einops", "pyyaml", "omegaconf", "easydict", "loguru", "tqdm",
        "setuptools", "pandas", "pyarrow", "soundfile", "huggingface_hub",
    )
    .run_commands(
        f"git clone --depth 1 {FOLEY_REPO} {FOLEY_DIR}",
        f"cd {FOLEY_DIR} && git rev-parse HEAD",
    )
    .env({"HF_HOME": f"{WEIGHTS}/hf", "PYTHONPATH": FOLEY_DIR})
    .add_local_python_source("soundlib")
)

music_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        f"git clone --depth 1 {ACE_REPO} {ACE_DIR}",
        f"cd {ACE_DIR} && git rev-parse HEAD",
        f"pip install {ACE_DIR}/acestep/third_parts/nano-vllm",
    )
    .pip_install(
        "transformers>=4.51.0,<4.58.0", "diffusers>=0.37.0", "gradio==6.2.0",
        "matplotlib>=3.7.5", "scipy>=1.10.1", "soundfile>=0.13.1", "loguru>=0.7.3",
        "einops>=0.8.1", "accelerate>=1.12.0", "fastapi>=0.110.0", "diskcache",
        "uvicorn[standard]>=0.27.0", "python-multipart>=0.0.18", "numba>=0.63.1",
        "vector-quantize-pytorch>=1.27.15", "torchcodec>=0.9.1", "torchao>=0.16.0,<0.17.0",
        "toml", "peft>=0.18.0", "lycoris-lora", "lightning>=2.0.0", "tensorboard>=2.20.0",
        "modelscope", "typer-slim>=0.21.1", "pytorch-wavelets>=1.3.0", "pywavelets>=1.9.0",
        "setuptools<72", "huggingface_hub",
    )
    .env({"HF_HOME": f"{WEIGHTS}/hf", "PYTHONPATH": ACE_DIR, "ACESTEP_INIT_LLM": "false"})
    .add_local_python_source("soundlib")
)


def _stream_audio(audio, sample_rate: int, meta: dict):
    """Yield a header dict, then the audio as FLAC bytes in small pieces."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC", subtype="PCM_16")
    data = buf.getvalue()
    yield {**meta, "sample_rate": sample_rate, "bytes": len(data)}
    for i in range(0, len(data), CHUNK):
        yield data[i:i + CHUNK]


def _to_numpy(tensor):
    """Torch audio (channels, samples) to float32 numpy (samples, channels)."""
    import numpy as np

    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 1:
        array = array[None, :]
    return np.clip(array.T, -1.0, 1.0)


# --- Sound planning ------------------------------------------------------------

@app.function(image=plan_image, secrets=[modal.Secret.from_name("anthropic")], timeout=600)
def plan_sound(frames: list[dict], shots: list[dict], duration: float, speech: list[dict]) -> dict:
    """Ask Claude for the sound plan. `frames` items: {shot, time, jpeg}."""
    from soundlib.planning import ask_claude

    return ask_claude(frames, shots, duration, speech)


# --- Sound effects -------------------------------------------------------------

@app.cls(image=foley_image, gpu=C.GPU_TYPE, volumes={WEIGHTS: weights}, timeout=1800,
         scaledown_window=60, max_containers=1)
class Foley:
    @modal.enter()
    def load(self):
        started = time.time()
        import torch
        from huggingface_hub import snapshot_download

        model_dir = f"{WEIGHTS}/foley"
        marker = f"{model_dir}/.download-complete"
        if not os.path.exists(marker):
            snapshot_download(
                "tencent/HunyuanVideo-Foley", local_dir=model_dir,
                allow_patterns=["hunyuanvideo_foley.pth", "synchformer_state_dict.pth",
                                "vae_128d_48k.pth", "*.yaml", "LICENSE", "NOTICE"],
            )
            open(marker, "w").close()
        from hunyuanvideo_foley.utils.model_utils import load_model

        config = f"{FOLEY_DIR}/configs/hunyuanvideo-foley-{C.FOLEY_MODEL_SIZE}.yaml"
        self.model_dict, self.cfg = load_model(
            model_dir, config, torch.device("cuda"), model_size=C.FOLEY_MODEL_SIZE)
        weights.commit()  # also saves the SigLIP2 and CLAP helper models for next time
        self.load_seconds = time.time() - started
        self.reported_load = False

    @modal.method()
    def generate(self, clip: bytes, prompt: str, seed: int = C.SEED,
                 steps: int = C.FOLEY_STEPS, guidance: float = C.FOLEY_GUIDANCE,
                 neg_prompt: str = C.FOLEY_NEG_PROMPT):
        """Generate sound for one short silent clip. Streams back FLAC audio."""
        import random
        import tempfile

        import numpy as np
        import torch
        from hunyuanvideo_foley.utils.feature_utils import feature_process
        from hunyuanvideo_foley.utils.model_utils import denoise_process

        started = time.time()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(clip)
            f.flush()
            visual, text, seconds = feature_process(
                f.name, prompt, self.model_dict, self.cfg, neg_prompt=neg_prompt)
            audio, sample_rate = denoise_process(
                visual, text, seconds, self.model_dict, self.cfg,
                guidance_scale=guidance, num_inference_steps=steps)
        meta = {"gpu_seconds": time.time() - started, "load_seconds": 0.0, "gpu": C.GPU_TYPE}
        if not self.reported_load:
            meta["load_seconds"], self.reported_load = self.load_seconds, True
        yield from _stream_audio(_to_numpy(audio[0]), int(sample_rate), meta)


# --- Music ---------------------------------------------------------------------

@app.cls(image=music_image, gpu=C.GPU_TYPE, volumes={WEIGHTS: weights}, timeout=1800,
         scaledown_window=60, max_containers=1)
class Music:
    @modal.enter()
    def load(self):
        started = time.time()
        from huggingface_hub import snapshot_download

        root = f"{WEIGHTS}/acestep"
        marker = f"{root}/.download-complete"
        if not os.path.exists(marker):
            # The language-model planner is not used (music brief comes from Claude),
            # so only the music model itself, its text encoder and its audio codec are needed.
            snapshot_download(
                "ACE-Step/Ace-Step1.5", local_dir=f"{root}/checkpoints",
                allow_patterns=[f"{C.MUSIC_MODEL}/*", "vae/*", "Qwen3-Embedding-0.6B/*",
                                "config.json"],
            )
            open(marker, "w").close()
            weights.commit()
        from acestep.handler import AceStepHandler

        self.dit = AceStepHandler()
        status, ok = self.dit.initialize_service(
            project_root=root, config_path=C.MUSIC_MODEL, device="cuda")
        if not ok:
            raise RuntimeError(f"ACE-Step failed to load: {status}")
        self.load_seconds = time.time() - started
        self.reported_load = False

    @modal.method()
    def generate(self, caption: str, bpm: int, duration: float, seed: int = C.SEED,
                 steps: int = C.MUSIC_STEPS):
        """Generate an instrumental track. Streams back FLAC audio."""
        import tempfile

        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        started = time.time()
        params = GenerationParams(
            task_type="text2music", caption=caption, lyrics="[Instrumental]", instrumental=True,
            bpm=int(bpm), duration=float(duration), inference_steps=steps, seed=seed,
            thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed],
                                  audio_format="flac")
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_music(self.dit, None, params, config, save_dir=tmp)
            if not result.success or not result.audios:
                raise RuntimeError(f"ACE-Step failed: {result.error or result.status_message}")
            item = result.audios[0]
            audio, sample_rate = _to_numpy(item["tensor"]), int(item["sample_rate"])
        meta = {"gpu_seconds": time.time() - started, "load_seconds": 0.0, "gpu": C.GPU_TYPE}
        if not self.reported_load:
            meta["load_seconds"], self.reported_load = self.load_seconds, True
        yield from _stream_audio(audio, sample_rate, meta)
