"""The library of real recordings, running on Modal.

  build_fsd50k  - downloads FSD50K from Zenodo, keeps only the CC0 recordings that pass
                  soundlib.library.keep_clip, and stores them with a catalogue (CPU)
  index         - gives every stored recording a CLAP "fingerprint" so it can be found
                  from a description (GPU, a few minutes, once per build)
  Search        - finds recordings for wanted sounds and lets Claude choose (CPU)
  summary       - what is in the library

Everything lives in the Modal volume `ai-sound-designer-library`:
  audio/<source>/<id>.flac     the recordings
  catalog_<source>_<part>.jsonl  one line per recording: source, id, title, uploader,
                               licence, link, labels, tags, description, measurements
  catalog.jsonl + embeddings.npy   the searchable index (same order)

Build it with:   modal run modal_library.py          (both parts, then the index)
Deploy search:   modal deploy modal_library.py

Licences (read 2026-11-25, see licence_research/real_recordings_library_licences.md):
  FSD50K (zenodo.org/records/4060432): each clip has its own Creative Commons licence;
  only the CC0 ones are kept. The collection itself is CC-BY, so the site credits it.
  CLAP (laion/larger_clap_general): Apache 2.0.
"""

import io
import json
import os
import time

import modal

from soundlib import config as C

APP_NAME = (os.environ.get("SOUND_APP") or "ai-sound-designer") + "-library"
app = modal.App(APP_NAME)

library = modal.Volume.from_name("ai-sound-designer-library", create_if_missing=True)
weights = modal.Volume.from_name("ai-sound-designer-weights", create_if_missing=True)
LIB, WEIGHTS = "/library", "/weights"
CLAP = "laion/larger_clap_general"
ZENODO = "https://zenodo.org/records/4060432/files"
FSD_PARTS = {
    "eval": ["FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip"],
    "dev": [f"FSD50K.dev_audio.z0{n}" for n in range(1, 6)] + ["FSD50K.dev_audio.zip"],
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("p7zip-full", "curl", "libsndfile1", "ffmpeg")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy<2", "scipy", "soundfile",
                 "anthropic>=2.14", "pydantic>=2")
    .env({"HF_HOME": f"{WEIGHTS}/hf", "SOUND_SETTINGS": "/root/settings.json"})
    .add_local_python_source("soundlib")
    .add_local_file(C.SETTINGS_FILE, "/root/settings.json")
)


def _fetch(name: str, folder: str) -> None:
    import subprocess

    subprocess.run(["curl", "-sSL", "--fail", "--retry", "8", "--retry-delay", "5", "-C", "-",
                    "-o", f"{folder}/{name}", f"{ZENODO}/{name}?download=1"], check=True)


@app.function(image=image, volumes={LIB: library}, timeout=4 * 3600, cpu=8, memory=8192)
def build_fsd50k(part: str = "eval", limit: int = 0) -> dict:
    """Download one part of FSD50K ("eval" or "dev") and store its usable CC0 recordings."""
    import csv
    import subprocess
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import soundfile as sf

    from soundlib import library as L

    started, tmp = time.time(), "/tmp/fsd"
    os.makedirs(tmp, exist_ok=True)
    for name in ("FSD50K.metadata.zip", "FSD50K.ground_truth.zip"):
        _fetch(name, tmp)
        zipfile.ZipFile(f"{tmp}/{name}").extractall(tmp)
    info = json.load(open(f"{tmp}/FSD50K.metadata/{part}_clips_info_FSD50K.json"))
    labels = {row["fname"]: row["labels"].split(",")
              for row in csv.DictReader(open(f"{tmp}/FSD50K.ground_truth/{part}.csv"))}

    reasons: dict[str, int] = {}
    chosen = []
    for fname, meta in info.items():
        ok, why = L.keep_clip(meta.get("license", ""), labels.get(fname, []), meta.get("title", ""),
                              meta.get("tags", []))
        reasons[why or "kept so far"] = reasons.get(why or "kept so far", 0) + 1
        if ok:
            chosen.append(fname)
    if limit:
        chosen = chosen[:limit]
    print(f"{part}: {len(info)} clips in the records, {len(chosen)} pass the licence and content rules: {reasons}",
          flush=True)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda n: _fetch(n, tmp), FSD_PARTS[part]))
    print(f"downloaded in {time.time() - started:.0f}s", flush=True)
    subprocess.run(["7z", "x", "-y", f"-o{tmp}", f"{tmp}/FSD50K.{part}_audio.zip"], check=True,
                   stdout=subprocess.DEVNULL)
    for name in FSD_PARTS[part]:
        os.remove(f"{tmp}/{name}")
    print(f"unpacked at {time.time() - started:.0f}s", flush=True)

    out_dir = f"{LIB}/audio/fsd50k"
    os.makedirs(out_dir, exist_ok=True)

    def store(fname: str):
        try:
            audio, sr = sf.read(f"{tmp}/FSD50K.{part}_audio/{fname}.wav", dtype="float32", always_2d=True)
        except Exception as exc:
            return None, f"unreadable: {exc}"
        stats = L.clip_stats(audio, sr)
        if stats["silent"] or stats["seconds"] < 0.3 or stats["clipped"] > 0.002:
            return None, "silent, too short or distorted"
        peak = float(np.abs(audio).max())
        audio = audio * (0.89 / peak)          # same headroom for every recording
        sf.write(f"{out_dir}/{fname}.flac", audio, sr, format="FLAC", subtype="PCM_16")
        meta = info[fname]
        return {"source": "fsd50k", "id": fname, "file": f"audio/fsd50k/{fname}.flac",
                "title": meta.get("title", ""), "author": meta.get("uploader", ""),
                "licence": "CC0 1.0", "licence_url": meta.get("license", ""),
                "url": f"https://freesound.org/s/{fname}/",
                "labels": labels.get(fname, []), "tags": meta.get("tags", []),
                "description": " ".join((meta.get("description") or "").split())[:600],
                "collection": "FSD50K (zenodo.org/records/4060432), " + part,
                "added": time.strftime("%Y-%m-%d"), **{k: v for k, v in stats.items() if k != "silent"}}, ""

    rows = []
    with ThreadPoolExecutor(8) as pool:
        for row, why in pool.map(store, chosen):
            if row:
                rows.append(row)
            else:
                reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1
    with open(f"{LIB}/catalog_fsd50k_{part}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    library.commit()
    hours = sum(r["seconds"] for r in rows) / 3600
    return {"part": part, "stored": len(rows), "hours": round(hours, 1), "counts": reasons,
            "seconds": round(time.time() - started)}


@app.function(image=image, gpu="L40S", volumes={LIB: library, WEIGHTS: weights}, timeout=3600)
def index() -> dict:
    """Fingerprint every stored recording with CLAP and write the searchable index."""
    import glob

    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import ClapModel, ClapProcessor

    started = time.time()
    library.reload()
    rows = [json.loads(line) for path in sorted(glob.glob(f"{LIB}/catalog_*_*.jsonl")) for line in open(path)]
    model = ClapModel.from_pretrained(CLAP).to("cuda").eval()
    processor = ClapProcessor.from_pretrained(CLAP)
    weights.commit()
    window = 10 * 48000
    vectors = []
    for at in range(0, len(rows), 48):
        batch = rows[at:at + 48]
        pieces, owner = [], []
        for n, row in enumerate(batch):
            audio, sr = sf.read(f"{LIB}/{row['file']}", dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if sr != 48000:
                mono = resample_poly(mono, 48000, sr).astype(np.float32)
            # Up to three 10-second windows spread over the recording; their average
            # describes the whole of it.
            starts = [0] if len(mono) <= window else np.linspace(0, len(mono) - window, 3).astype(int)
            for s in starts:
                pieces.append(mono[s:s + window])
                owner.append(n)
        with torch.no_grad():
            inputs = processor(audios=pieces, sampling_rate=48000, return_tensors="pt")
            emb = model.get_audio_features(**{k: v.to("cuda") for k, v in inputs.items()})
            emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
        for n in range(len(batch)):
            v = emb[[i for i, o in enumerate(owner) if o == n]].mean(axis=0)
            vectors.append(v / (np.linalg.norm(v) + 1e-9))
        if at % 960 == 0:
            print(f"{at}/{len(rows)} at {time.time() - started:.0f}s", flush=True)
    np.save(f"{LIB}/embeddings.npy", np.asarray(vectors, dtype=np.float32))
    with open(f"{LIB}/catalog.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    library.commit()
    return {"recordings": len(rows), "gpu_seconds": round(time.time() - started)}


def _suits(row: dict, kind: str, seconds: float) -> bool:
    """Whether a recording can do this kind of job at all, from its measurements."""
    if kind in ("ambience", "source"):          # long and even, nothing happening in it
        return row["seconds"] >= max(4.0, min(seconds, 8.0) * 0.6) and row["steady_db"] <= 6.0
    if kind == "whoosh":
        return row["seconds"] <= 8.0 and row["range_db"] >= 12.0
    return row["seconds"] <= 20.0 and row["range_db"] >= 15.0     # a clear event over a quiet background


@app.cls(image=image, volumes={LIB: library, WEIGHTS: weights}, cpu=2, memory=4096, timeout=600,
         scaledown_window=120, secrets=[modal.Secret.from_name(C.CLAUDE_SECRET_NAME)])
class Search:
    @modal.enter()
    def load(self):
        import numpy as np
        from transformers import ClapModel, ClapProcessor

        self.rows = [json.loads(line) for line in open(f"{LIB}/catalog.jsonl")]
        self.vectors = np.load(f"{LIB}/embeddings.npy")
        self.model = ClapModel.from_pretrained(CLAP).eval()
        self.processor = ClapProcessor.from_pretrained(CLAP)

    def _shortlists(self, asks: list[dict], opts: dict) -> list[list[dict]]:
        import numpy as np
        import torch

        with torch.no_grad():
            t = self.processor(text=[a["prompt"] for a in asks], return_tensors="pt", padding=True)
            text = torch.nn.functional.normalize(self.model.get_text_features(**t), dim=-1).numpy()
        scores = text @ self.vectors.T
        sources = set(opts.get("sources") or [])
        lists = []
        for ask, row_scores in zip(asks, scores):
            words = {w for w in ask["prompt"].lower().replace(",", " ").split() if len(w) > 3}
            found = []
            for i in np.argsort(-row_scores)[:400]:
                row = self.rows[i]
                if (sources and row["source"] not in sources) or not _suits(row, ask["kind"], ask["seconds"]):
                    continue
                # A small push for recordings whose own title, tags or labels use the same words.
                theirs = " ".join([row["title"], *row["tags"], *row["labels"]]).lower().replace("_", " ")
                bonus = 0.06 * sum(w in theirs for w in words) / max(1, len(words))
                found.append({**row, "score": round(float(row_scores[i]) + bonus, 4)})
                if len(found) >= 40:
                    break
            found.sort(key=lambda r: -r["score"])
            lists.append([r for r in found if r["score"] >= float(opts.get("min_score", 0.0))]
                         [: int(opts.get("candidates", 6))]
                         or [])
            if not lists[-1] and found:
                lists[-1] = []
                ask["_best"] = found[0]["score"]
        return lists

    @modal.method()
    def match(self, asks: list[dict], opts: dict) -> dict:
        """For each wanted sound {prompt, kind, seconds}: the chosen recording (its
        catalogue line, its audio as FLAC bytes, why) or just why nothing was chosen."""
        from soundlib import library as L

        lists = self._shortlists(asks, opts)
        dollars, model, picks = 0.0, None, None
        if str(opts.get("judge", "")).startswith("claude") and any(lists):
            try:
                judged = L.ask_judge(asks, lists, opts["judge_model"], opts["judge"])
                picks, dollars, model = judged["picks"], judged["dollars"], judged["model"]
            except Exception as exc:        # no judge: fall back to the score alone, but be stricter
                print(f"judge failed: {exc}", flush=True)
        results = []
        for n, (ask, shortlist) in enumerate(zip(asks, lists)):
            best = shortlist[0]["score"] if shortlist else ask.get("_best")
            if not shortlist:
                results.append({"why": "no recording is close enough", "best_score": best})
                continue
            if picks is not None:
                pick, why = picks[n]
                if not 1 <= pick <= len(shortlist):
                    results.append({"why": why or "the judge chose to generate it", "best_score": best,
                                    "considered": [c["title"] for c in shortlist]})
                    continue
                chosen = shortlist[pick - 1]
            else:
                chosen, why = shortlist[0], "closest match by sound"
                if chosen["score"] < float(opts.get("min_score", 0.0)) + 0.08:
                    results.append({"why": "no judge available and the match is not strong", "best_score": best})
                    continue
            results.append({"clip": {k: v for k, v in chosen.items() if k not in ("description",)},
                            "flac": open(f"{LIB}/{chosen['file']}", "rb").read(),
                            "score": chosen["score"], "why": why, "best_score": best})
        return {"results": results, "dollars": dollars, "model": model}

    @modal.method()
    def look(self, prompt: str, kind: str = "moment", seconds: float = 1.5, count: int = 10) -> list[dict]:
        """The closest recordings for a description, without choosing (for checking the library)."""
        asks = [{"prompt": prompt, "kind": kind, "seconds": seconds}]
        return [{k: r[k] for k in ("id", "title", "labels", "seconds", "events", "steady_db", "range_db", "score")}
                for r in self._shortlists(asks, {"candidates": count, "min_score": 0.0})[0]]


@app.function(image=image, volumes={LIB: library})
def summary() -> dict:
    library.reload()
    path = f"{LIB}/catalog.jsonl"
    if not os.path.exists(path):
        return {"recordings": 0}
    rows = [json.loads(line) for line in open(path)]
    labels: dict[str, int] = {}
    for row in rows:
        for label in row["labels"]:
            labels[label] = labels.get(label, 0) + 1
    return {"recordings": len(rows), "hours": round(sum(r["seconds"] for r in rows) / 3600, 1),
            "licences": sorted({r["licence"] for r in rows}),
            "sources": sorted({r["source"] for r in rows}),
            "labels": dict(sorted(labels.items(), key=lambda kv: -kv[1]))}


@app.local_entrypoint()
def main(parts: str = "eval,dev", limit: int = 0, skip_build: bool = False):
    if not skip_build:
        for result in build_fsd50k.starmap([(p, limit) for p in parts.split(",")]):
            print(json.dumps(result, indent=1))
    print(json.dumps(index.remote(), indent=1))
    print(json.dumps(summary.remote(), indent=1)[:3000])
