"""Downloading model weights into the shared Modal volume.

Runs inside Modal. Downloading needs no GPU, so `python make_sound.py --deploy` does it
on a cheap CPU machine first; the GPU classes call the same functions as a safety net.
"""

from __future__ import annotations

import os

from . import config as C

FOLEY_HELPERS = ("google/siglip2-base-patch16-512", "laion/larger_clap_general")


def foley_dir(root: str) -> str:
    return f"{root}/foley"


def music_root(root: str) -> str:
    return f"{root}/acestep"


def download_foley(root: str) -> bool:
    """Fetch HunyuanVideo-Foley and its two helper models. Returns True if anything was fetched."""
    from huggingface_hub import snapshot_download

    model_dir = foley_dir(root)
    marker = f"{model_dir}/.download-complete-v2"
    if os.path.exists(marker):
        return False
    snapshot_download(
        "tencent/HunyuanVideo-Foley", local_dir=model_dir,
        allow_patterns=["hunyuanvideo_foley.pth", "synchformer_state_dict.pth",
                        "vae_128d_48k.pth", "*.yaml", "LICENSE", "NOTICE"],
    )
    for repo in FOLEY_HELPERS:  # land in HF_HOME, which is also on the volume
        snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.md"])
    open(marker, "w").close()
    return True


BASE_MUSIC_MODEL = "acestep-v15-turbo"


def download_music(root: str, model: str = BASE_MUSIC_MODEL) -> bool:
    """Fetch ACE-Step 1.5 (music model, text encoder, audio codec). The optional
    language-model planner is skipped: the music brief comes from Claude.

    Other music models (the XL ones) live in their own Hugging Face repos and are
    fetched next to the standard one the first time they are asked for.
    """
    from huggingface_hub import snapshot_download

    base = music_root(root)
    fetched = False
    marker = f"{base}/.download-complete"
    if not os.path.exists(marker):
        snapshot_download(
            "ACE-Step/Ace-Step1.5", local_dir=f"{base}/checkpoints",
            allow_patterns=[f"{BASE_MUSIC_MODEL}/*", "vae/*", "Qwen3-Embedding-0.6B/*", "config.json"],
        )
        open(marker, "w").close()
        fetched = True
    if model != BASE_MUSIC_MODEL:
        marker = f"{base}/.download-complete-{model}"
        if not os.path.exists(marker):
            snapshot_download(f"ACE-Step/{model}", local_dir=f"{base}/checkpoints/{model}")
            open(marker, "w").close()
            fetched = True
    return fetched
