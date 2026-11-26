"""The cloud side of AI Sound Designer, running on Modal.

Three pieces:
  plan_sound  - asks Claude for the sound plan (CPU; reads the Claude key from a Modal secret)
  Foley       - HunyuanVideo-Foley: synced sound effects from video + text (GPU)
  Music       - ACE-Step 1.5: instrumental music from a text brief (GPU)

Model weights live in a Modal volume, so only the first run downloads them.

Deploy with:  python make_sound.py --deploy     (or: modal deploy modal_app.py)

Licences (read 2026-11-24):
  HunyuanVideo-Foley - Tencent Hunyuan Community License. Not licensed for use in the
                       EU, the UK or South Korea. The web app (Phase 3) must enforce this.
  ACE-Step 1.5       - MIT.
"""


import io
import os
import time

import modal

from soundlib import config as C

APP_NAME = os.environ.get("SOUND_APP") or "ai-sound-designer"   # SOUND_APP: a separate trial copy
app = modal.App(APP_NAME)

weights = modal.Volume.from_name("ai-sound-designer-weights", create_if_missing=True)
WEIGHTS = "/weights"
CHUNK = 1_000_000  # audio is streamed back in pieces of this many bytes

FOLEY_REPO = "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley"
FOLEY_COMMIT = "df7b005b5023df2a9b73e1d66dd51d452799884e"  # pinned 2026-11-24
FOLEY_DIR = "/opt/HunyuanVideo-Foley"
ACE_REPO = "https://github.com/ace-step/ACE-Step-1.5"
ACE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"  # pinned 2026-11-24
ACE_DIR = "/opt/ACE-Step-1.5"

# --- Images ------------------------------------------------------------------

plan_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("anthropic>=2.14", "pydantic>=2", "numpy")
    .env({"SOUND_SETTINGS": "/root/settings.json"})
    .add_local_python_source("soundlib")
    .add_local_file(C.SETTINGS_FILE, "/root/settings.json")
)

foley_image = (
    modal.Image.debian_slim(python_version="3.11")
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
        f"git clone {FOLEY_REPO} {FOLEY_DIR} && cd {FOLEY_DIR} && git checkout {FOLEY_COMMIT}",
    )
    .env({"HF_HOME": f"{WEIGHTS}/hf", "PYTHONPATH": FOLEY_DIR, "SOUND_SETTINGS": "/root/settings.json"})
    .add_local_python_source("soundlib")
    .add_local_file(C.SETTINGS_FILE, "/root/settings.json")
)

music_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        f"git clone {ACE_REPO} {ACE_DIR} && cd {ACE_DIR} && git checkout {ACE_COMMIT}",
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
    .env({"HF_HOME": f"{WEIGHTS}/hf", "PYTHONPATH": ACE_DIR, "ACESTEP_INIT_LLM": "false",
          "SOUND_SETTINGS": "/root/settings.json"})
    .add_local_python_source("soundlib")
    .add_local_file(C.SETTINGS_FILE, "/root/settings.json")
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

@app.function(image=plan_image, secrets=[modal.Secret.from_name(C.CLAUDE_SECRET_NAME)], timeout=600)
def plan_sound(frames: list[dict], shots: list[dict], duration: float, speech: list[dict],
               vibe: str | None = None) -> dict:
    """Ask Claude for the sound plan. `frames` items: {shot, time, jpeg}."""
    from soundlib.planning import ask_claude

    return ask_claude(frames, shots, duration, speech, vibe)


@app.function(image=plan_image, secrets=[modal.Secret.from_name(C.CLAUDE_SECRET_NAME)], timeout=600)
def plan_cues(frames: list[dict], shots: list[dict], duration: float, plan: dict) -> dict:
    """Ask Claude for the music cue sheet (sections, intensity, hit points)."""
    from soundlib.planning import ask_cues

    return ask_cues(frames, shots, duration, plan)


@app.function(image=plan_image, secrets=[modal.Secret.from_name(C.CLAUDE_SECRET_NAME)], timeout=900)
def plan_missed(frames: list[dict], windows: list[dict], duration: float, plan: dict,
                model: str | None = None, most: int | None = None, effort: str | None = None) -> dict:
    """Claude's second, closer look: visible actions that have no sound yet."""
    from soundlib.planning import ask_missed

    return ask_missed(frames, windows, duration, plan, model, most, effort)


@app.function(image=plan_image, secrets=[modal.Secret.from_name(C.CLAUDE_SECRET_NAME)], timeout=900)
def plan_pins(bursts: list[dict], model: str | None = None, effort: str | None = None) -> dict:
    """Claude picks the exact frame each key moment happens on."""
    from soundlib.planning import ask_pins

    return ask_pins(bursts, model, effort)


# --- One-time preparation (CPU, cheap) -------------------------------------------

@app.function(image=foley_image, volumes={WEIGHTS: weights}, timeout=3600)
def prepare_foley() -> str:
    """Download the effects model and check that its code imports."""
    from soundlib import weights as W

    fetched = W.download_foley(WEIGHTS)
    weights.commit()
    from hunyuanvideo_foley.utils.feature_utils import feature_process  # noqa: F401
    from hunyuanvideo_foley.utils.model_utils import denoise_process, load_model  # noqa: F401
    return "downloaded" if fetched else "already there"


@app.function(image=music_image, volumes={WEIGHTS: weights}, timeout=3600)
def prepare_music(model: str = C.MUSIC_MODEL) -> str:
    """Download a music model and check that its code imports."""
    from soundlib import weights as W

    fetched = W.download_music(WEIGHTS, model)
    weights.commit()
    from acestep.handler import AceStepHandler  # noqa: F401
    from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: F401
    return "downloaded" if fetched else "already there"


# --- Sound effects -------------------------------------------------------------

@app.cls(image=foley_image, gpu=C.GPU_TYPE, volumes={WEIGHTS: weights}, timeout=1800,
         scaledown_window=60, max_containers=1)
class Foley:
    @modal.enter()
    def load(self):
        started = time.time()
        import torch
        from soundlib import weights as W

        model_dir = W.foley_dir(WEIGHTS)
        if W.download_foley(WEIGHTS):
            weights.commit()
        from hunyuanvideo_foley.utils.model_utils import load_model

        config = f"{FOLEY_DIR}/configs/hunyuanvideo-foley-{C.FOLEY_MODEL_SIZE}.yaml"
        self.model_dict, self.cfg = load_model(
            model_dir, config, torch.device("cuda"), model_size=C.FOLEY_MODEL_SIZE)
        self.load_seconds = time.time() - started
        self.reported_load = False

    def _match_scores(self, prompt: str, audios: list, sample_rate: int,
                      judge: dict | None = None) -> list[float]:
        """How well each audio matches the words, from the CLAP model (higher is better).

        With a judge ({"reject": [descriptions], "onset": seconds or None}) a take is
        marked down for sounding like any rejected description (speech, music) more
        than a little, and for starting far from the instant the action happens.
        """
        import torch
        import torchaudio
        from transformers import ClapModel, ClapProcessor

        if getattr(self, "clap", None) is None:
            self.clap = ClapModel.from_pretrained("laion/larger_clap_general").to("cuda").eval()
            self.clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        waves = []
        for audio in audios:
            mono = audio.detach().float().cpu()
            mono = mono.mean(dim=0) if mono.ndim > 1 else mono
            if sample_rate != 48000:
                mono = torchaudio.functional.resample(mono, sample_rate, 48000)
            waves.append(mono.numpy())
        with torch.no_grad():
            a = self.clap_processor(audios=waves, sampling_rate=48000, return_tensors="pt")
            reject = list((judge or {}).get("reject") or [])
            t = self.clap_processor(text=[prompt, *reject], return_tensors="pt", padding=True)
            audio_emb = self.clap.get_audio_features(**{k: v.to("cuda") for k, v in a.items()})
            text_emb = self.clap.get_text_features(**{k: v.to("cuda") for k, v in t.items()})
            audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
            text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
            sims = (audio_emb @ text_emb.T).cpu().numpy()
        scores = []
        for wave, row in zip(waves, sims):
            score = float(row[0])
            if reject:      # only a clear likeness counts against a take
                score -= max(0.0, float(row[1:].max()) - 0.15)
            onset = (judge or {}).get("onset")
            if onset is not None and len(wave) > 4800:
                import numpy as np

                frames = wave[: len(wave) // 480 * 480].reshape(-1, 480)
                level = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
                began = float(np.argmax(level > level.max() - 12.0)) / 100.0
                score -= 0.3 * min(abs(began - float(onset)), 0.5)
            scores.append(score)
        return scores

    @modal.method()
    def generate(self, clip: bytes, prompt: str, seed: int = C.SEED,
                 steps: int = C.FOLEY_STEPS, guidance: float = C.FOLEY_GUIDANCE,
                 neg_prompt: str = C.FOLEY_NEG_PROMPT, candidates: int = 1,
                 judge: dict | None = None):
        """Generate sound for one short silent clip. Streams back FLAC audio.

        With candidates > 1, makes that many versions (different seeds) and keeps the
        one whose sound best matches the description.
        """
        import random
        import tempfile

        import numpy as np
        import torch
        from hunyuanvideo_foley.utils.feature_utils import feature_process
        from hunyuanvideo_foley.utils.model_utils import denoise_process

        started = time.time()
        made, sample_rate = [], 48000
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(clip)
            f.flush()
            visual, text, seconds = feature_process(
                f.name, prompt, self.model_dict, self.cfg, neg_prompt=neg_prompt)
            for n in range(max(1, int(candidates))):
                s = seed + 7919 * n
                random.seed(s)
                np.random.seed(s)
                torch.manual_seed(s)
                audio, sample_rate = denoise_process(
                    visual, text, seconds, self.model_dict, self.cfg,
                    guidance_scale=guidance, num_inference_steps=steps)
                made.append(audio[0])
        scores, best = [], 0
        if len(made) > 1:
            scores = self._match_scores(prompt, made, int(sample_rate), judge)
            best = max(range(len(made)), key=lambda i: scores[i])
        meta = {"gpu_seconds": time.time() - started, "load_seconds": 0.0, "gpu": C.GPU_TYPE,
                "scores": scores, "picked": best}
        if not self.reported_load:
            meta["load_seconds"], self.reported_load = self.load_seconds, True
        yield from _stream_audio(_to_numpy(made[best]), int(sample_rate), meta)


# --- Music ---------------------------------------------------------------------

@app.cls(image=music_image, gpu=C.GPU_TYPE, volumes={WEIGHTS: weights}, timeout=1800,
         scaledown_window=60, max_containers=1)
class Music:
    model: str = modal.parameter(default="acestep-v15-turbo")

    @modal.enter()
    def load(self):
        started = time.time()
        from soundlib import weights as W

        root = W.music_root(WEIGHTS)
        if W.download_music(WEIGHTS, self.model):
            weights.commit()
        from acestep.handler import AceStepHandler

        self.dit = AceStepHandler()
        status, ok = self.dit.initialize_service(
            project_root=root, config_path=self.model, device="cuda")
        if not ok:
            raise RuntimeError(f"ACE-Step failed to load: {status}")
        self.load_seconds = time.time() - started
        self.reported_load = False

    @modal.method()
    def generate(self, caption: str, bpm: int, duration: float, seed: int = C.SEED,
                 steps: int = C.MUSIC_STEPS, guidance: float = C.MUSIC_GUIDANCE,
                 keyscale: str = ""):
        """Generate an instrumental track. Streams back FLAC audio."""
        import tempfile

        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        started = time.time()
        params = GenerationParams(
            task_type="text2music", caption=caption, lyrics="[Instrumental]", instrumental=True,
            bpm=int(bpm), duration=float(duration), inference_steps=steps, seed=seed,
            guidance_scale=float(guidance), keyscale=keyscale,
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

    @modal.method()
    def generate_scored(self, caption: str, bpm: int, duration: float, sections: list[dict],
                        seed: int = C.SEED, steps: int = C.MUSIC_STEPS,
                        guidance: float = C.MUSIC_GUIDANCE, keyscale: str = ""):
        """One continuous track scored in sections: compose the whole track, then have
        the model rewrite ("repaint") each section in place from that section's own
        caption. `sections` items: {start, end, caption}. Streams back FLAC audio."""
        import tempfile

        import soundfile as sf
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        started = time.time()
        common = dict(lyrics="[Instrumental]", instrumental=True, bpm=int(bpm),
                      duration=float(duration), inference_steps=steps,
                      guidance_scale=float(guidance), keyscale=keyscale, thinking=False,
                      use_cot_metas=False, use_cot_caption=False, use_cot_language=False)
        with tempfile.TemporaryDirectory() as tmp:
            def run(n: int, **kw):
                config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed + n],
                                          audio_format="flac")
                result = generate_music(self.dit, None, GenerationParams(seed=seed + n, **common, **kw),
                                        config, save_dir=f"{tmp}/out{n}")
                if not result.success or not result.audios:
                    raise RuntimeError(f"ACE-Step failed: {result.error or result.status_message}")
                item = result.audios[0]
                audio, sr = _to_numpy(item["tensor"]), int(item["sample_rate"])
                path = f"{tmp}/step{n}.wav"
                sf.write(path, audio, sr, subtype="FLOAT")
                return audio, sr, path

            os.makedirs(tmp, exist_ok=True)
            audio, sample_rate, path = run(0, task_type="text2music", caption=caption)
            for n, sec in enumerate(sections, start=1):
                end = -1 if sec["end"] >= duration - 0.05 else float(sec["end"])
                audio, sample_rate, path = run(
                    n, task_type="repaint", caption=sec["caption"], src_audio=path,
                    repainting_start=float(sec["start"]), repainting_end=end)
        meta = {"gpu_seconds": time.time() - started, "load_seconds": 0.0, "gpu": C.GPU_TYPE}
        if not self.reported_load:
            meta["load_seconds"], self.reported_load = self.load_seconds, True
        yield from _stream_audio(audio, sample_rate, meta)
