#!/usr/bin/env python3
"""Make the sound better without guessing.

    python tune.py              Run the current settings plus every variant in
                                variants.json on the test videos, then build the
                                comparison page in tuning/index.html.
    python tune.py --serve      Open the page at http://localhost:8765 . Ratings save
                                to ratings.csv as you click.
    python tune.py --apply      Read ratings.csv, make the winners the new defaults in
                                settings.json and note what changed in PROGRESS.md.

Each variant changes ONE setting. Everything generated is remembered in cache/, so a
variant only pays for the part it actually changes (a mix-level variant is free).

    python tune.py --dry-run    Test tones instead of AI sound (free, for testing).
    python tune.py --page-only  Rebuild the page from what is already in tuning/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from make_sound import ROOT, VIDEO_TYPES, Cached, Cloud, TestTones, make_sound, make_versions, say
from soundlib import config as C
from soundlib import media

TUNING = ROOT / "tuning"
RATINGS = ROOT / "ratings.csv"
TEMPLATE = ROOT / "soundlib" / "assets" / "tuning_page.html"
BASELINE = "current"
COMBO = "(several)"
CSV_FIELDS = ["video", "variant", "setting", "value", "rating", "note", "rated_at"]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "x"


def load_variants(path: Path) -> dict:
    spec = json.loads(path.read_text())
    seen = {BASELINE}
    for v in spec["variants"]:
        if "settings" in v:  # a combination: several settings changed together
            v["setting"], v["value"] = COMBO, v["settings"]
        missing = [k for k in ("id", "label", "setting", "value", "why") if k not in v]
        if missing:
            raise ValueError(f"Variant {v.get('id', '?')} is missing: {missing}")
        for name in (v["value"] if v["setting"] == COMBO else [v["setting"]]):
            C.default(name)  # raises a clear error for an unknown setting
        if v["id"] in seen or v["id"].startswith("take_"):
            raise ValueError(f"Variant id '{v['id']}' is used twice or is reserved")
        seen.add(v["id"])
    return spec


def find_videos(spec: dict, chosen: list[str]) -> list[Path]:
    names = chosen or spec.get("videos") or []
    if names:
        videos = [Path(n) if Path(n).exists() else ROOT / "test_videos" / n for n in names]
    else:
        videos = sorted(p for p in (ROOT / "test_videos").iterdir() if p.suffix.lower() in VIDEO_TYPES)
    missing = [str(v) for v in videos if not v.exists()]
    if missing or not videos:
        raise SystemExit(f"No test videos found ({missing or 'test_videos/ is empty'}).")
    return videos


def preview(final: Path, out: Path) -> None:
    """A small copy for the page: quick to load, full-quality sound."""
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = "scale='if(gt(iw,ih),-2,min(480,ih))':'if(gt(iw,ih),min(480,ih),-2)'"
    media.run(["ffmpeg", "-y", "-v", "error", "-i", final, "-vf", scale, "-c:v", "libx264",
               "-preset", "medium", "-crf", "27", "-maxrate", "1100k", "-bufsize", "2200k", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-b:a", "192k", "-movflags", "+faststart", out])


def show(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= 60 else text[:57] + "..."


# --- Running the variants ------------------------------------------------------------

def run(spec: dict, videos: list[Path], backend, only: list[str]) -> dict:
    cached = Cached(backend, ROOT / "cache")
    dry = isinstance(backend, TestTones)
    manifest = {"run": dt.datetime.now().strftime("%Y%m%d-%H%M%S"), "test_tones_only": dry,
                "videos": [], "cost": {"gpu_seconds": 0.0, "dollars": 0.0}}
    failures = []

    def one(video: Path, variant_id: str, out: Path, **kw) -> dict | None:
        try:
            report = make_sound(video, out, cached, **kw)
        except Exception as exc:  # one broken variant must not lose the others
            say(f"ERROR on {video.name} / {variant_id}: {exc}")
            failures.append(f"{video.name} / {variant_id}: {exc}")
            return None
        manifest["cost"]["gpu_seconds"] += report["cost"]["gpu_seconds"]
        manifest["cost"]["dollars"] += report["cost"]["total_dollars"]
        if not report["passed"]:
            failures.append(f"{video.name} / {variant_id}: failed its automatic checks")
        return report

    for video in videos:
        vid = slug(video.stem)
        info = media.probe(video)
        vibe = spec.get("vibes", {}).get(video.stem)
        entry = {"id": vid, "title": video.name, "aspect": f"{info['width']} / {info['height']}",
                 "groups": []}
        items = []
        todo = [{"id": BASELINE, "label": "Current settings", "setting": None, "value": None,
                 "why": "The sound as it is today. Score this one too: it is the yardstick."}]
        todo += [v for v in spec["variants"] if not only or v["id"] in only]
        for v in todo:
            C.reset()
            if v["setting"]:
                C.override(v["value"] if v["setting"] == COMBO else {v["setting"]: v["value"]})
            say(f"\n##### {video.name}: {v['label']} #####")
            out = TUNING / vid / v["id"]
            report = one(video, v["id"], out, vibe=vibe)
            C.reset()
            if report is None:
                continue
            file = f"media/{vid}__{slug(v['id'])}.mp4"
            preview(out / "final.mp4", TUNING / file)
            change = None
            if v["setting"] == COMBO:
                change = "several settings at once"
            elif v["setting"]:
                change = f"{v['setting']}: {show(C.default(v['setting']))} → {show(v['value'])}"
            items.append({"id": v["id"], "key": f"{vid}__{slug(v['id'])}", "label": v["label"],
                          "why": v["why"], "setting": v["setting"], "value": v["value"],
                          "change": change, "file": file,
                          "music": report["music"]["caption"], "bpm": report["music"]["bpm"]})
        entry["groups"].append({
            "title": "One change at a time",
            "about": "The first tile is the sound as it is now. Every other tile changes one setting.",
            "items": items})

        if spec.get("include_takes") and not only:
            takes = []
            say(f"\n##### {video.name}: three takes #####")
            try:
                reports = make_versions(video, TUNING / vid / "takes", cached, vibe=vibe)
            except Exception as exc:
                say(f"ERROR making takes for {video.name}: {exc}")
                failures.append(f"{video.name} / takes: {exc}")
                reports = []
            for n, report in enumerate(reports, start=1):
                manifest["cost"]["gpu_seconds"] += report["cost"]["gpu_seconds"]
                manifest["cost"]["dollars"] += report["cost"]["total_dollars"]
                file = f"media/{vid}__take-{n}.mp4"
                preview(TUNING / vid / "takes" / f"take_{n}" / "final.mp4", TUNING / file)
                flavour = report["music_flavour"]
                takes.append({"id": f"take_{n}", "key": f"{vid}__take-{n}", "label": report["take"],
                              "why": ("Music: " + flavour + "." if flavour else "The standard music brief.")
                                     + " Its own roll of the dice for effects and music.",
                              "setting": None, "value": None, "change": None, "file": file})
            if takes:
                entry["groups"].append({
                    "title": "Three takes",
                    "about": "What a user gets from the versions option: three different rolls "
                             "of the same video to pick from. Scores here show how much the "
                             "result varies from take to take; they do not change the settings.",
                    "items": takes})
        manifest["videos"].append(entry)

    manifest["cost"] = {k: round(v, 3) for k, v in manifest["cost"].items()}
    manifest["failures"] = failures
    TUNING.mkdir(exist_ok=True)
    (TUNING / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --- The page --------------------------------------------------------------------------

def build_page(manifest: dict) -> Path:
    """Write tuning/index.html (opens anywhere) and tuning/page_fragment.html (the same
    page without the outer html/head tags, for publishing as a shared page)."""
    data = json.dumps(manifest).replace("<", "\\u003c")
    fragment = TEMPLATE.read_text().replace("/*DATA*/", data)
    (TUNING / "page_fragment.html").write_text(fragment)
    page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<style>body{margin:0}[hidden]{display:none!important}</style></head><body>\n"
            + fragment + "\n</body></html>\n")
    (TUNING / "index.html").write_text(page)
    return TUNING / "index.html"


# --- Ratings ---------------------------------------------------------------------------

def read_ratings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f)]


def write_ratings(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def serve(port: int) -> None:
    if not (TUNING / "index.html").exists():
        raise SystemExit("Nothing to show yet. Run: python tune.py")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(TUNING), **kw)

        def log_message(self, *a):
            pass

        def _json(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/ratings":
                stored = {f"{slug(r['video'])}__{slug(r['variant'])}":
                          {"rating": int(r["rating"]) if r.get("rating") else None,
                           "note": r.get("note", ""), "at": r.get("rated_at", "")}
                          for r in read_ratings(RATINGS)}
                return self._json(stored)
            return super().do_GET()

        def send_head(self):
            # Browsers need byte-range requests to jump around inside a video.
            rng = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
            path = Path(self.translate_path(self.path))
            if not rng or not path.is_file():
                return super().send_head()
            size = path.stat().st_size
            start = int(rng.group(1)) if rng.group(1) else max(0, size - int(rng.group(2) or 0))
            end = min(int(rng.group(2)), size - 1) if rng.group(1) and rng.group(2) else size - 1
            if start > end:
                self.send_error(416)
                return None
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return io.BytesIO(path.read_bytes()[start:end + 1])

        def do_POST(self):
            if self.path != "/api/ratings":
                return self._json({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            rows = list(csv.DictReader(io.StringIO(body["csv"])))
            write_ratings(RATINGS, rows)
            say(f"Saved {len(rows)} rating(s) to {RATINGS.name}")
            return self._json({"saved": len(rows)})

    say(f"Comparison page: http://localhost:{port}  (Ctrl+C to stop). Ratings save to {RATINGS}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def decide(rows: list[dict]) -> dict:
    """Work out the winners. A variant wins when its average score beats the current
    settings' average over the videos where both were rated. If two winners change the
    same setting, the higher score takes it."""
    scores: dict[str, dict[str, float]] = {}
    meta: dict[str, dict] = {}
    for r in rows:
        if not r.get("rating") or r["variant"].startswith("take_"):
            continue
        scores.setdefault(r["variant"], {})[r["video"]] = float(r["rating"])
        if r.get("setting"):
            meta[r["variant"]] = {"setting": r["setting"], "value": json.loads(r["value"])}
    base = scores.get(BASELINE, {})
    table, winners = [], {}
    for variant, by_video in sorted(scores.items()):
        if variant == BASELINE:
            continue
        shared = [v for v in by_video if v in base]
        if not shared or variant not in meta:
            table.append({"variant": variant, "verdict": "skipped (current settings not rated on the same video)"})
            continue
        mine = sum(by_video[v] for v in shared) / len(shared)
        theirs = sum(base[v] for v in shared) / len(shared)
        row = {"variant": variant, **meta[variant], "score": round(mine, 2),
               "current_score": round(theirs, 2), "videos": len(shared),
               "verdict": "better" if mine > theirs else ("same" if mine == theirs else "worse")}
        table.append(row)
        if mine > theirs and row["setting"] != COMBO:
            best = winners.get(row["setting"])
            if best is None or mine > best["score"]:
                winners[row["setting"]] = row
    for row in table:
        if row.get("verdict") == "better" and row["setting"] != COMBO \
                and winners.get(row["setting"]) is not row:
            row["verdict"] = "better, but another change to the same setting scored higher"
    # A combination wins outright when it beats the current settings and scores at least
    # as high as every single change; then all of its settings become the defaults.
    combos = [r for r in table if r.get("setting") == COMBO and r["verdict"] == "better"]
    if combos:
        top = max(combos, key=lambda r: r["score"])
        if all(top["score"] >= w["score"] for w in winners.values()):
            winners = {name: {**top, "setting": name, "value": value}
                       for name, value in top["value"].items()}
            top["verdict"] = "better (best overall, all its settings applied)"
        else:
            top["verdict"] = "better, but a single change scored higher"
    return {"table": table, "winners": winners,
            "baseline": round(sum(base.values()) / len(base), 2) if base else None,
            "takes": [r for r in rows if r["variant"].startswith("take_") and r.get("rating")]}


def apply(ratings: Path, write: bool = True) -> dict:
    rows = read_ratings(ratings)
    if not rows:
        raise SystemExit(f"No ratings found in {ratings}. Rate the variants first.")
    result = decide(rows)
    if result["baseline"] is None:
        raise SystemExit("The 'Current settings' tile has no score, so there is nothing to compare with.")
    changes = {s: w["value"] for s, w in result["winners"].items()}
    old = {s: C.default(s) for s in changes}
    say(f"Current settings scored {result['baseline']}.")
    for row in result["table"]:
        say(f"  {row['variant']}: {row.get('score', '-')} ({row['verdict']})")
    if write:
        if changes:
            C.save_defaults(changes)
        lines = [f"\n## Tuning round, {dt.date.today().isoformat()}\n",
                 f"From Anthony's ratings in `{ratings.name}` (current settings scored "
                 f"{result['baseline']} out of 5):\n"]
        for row in result["table"]:
            lines.append(f"- `{row['variant']}`: {row.get('score', '-')} ({row['verdict']})")
        lines.append("")
        if changes:
            lines.append("New defaults written to `settings.json`:\n")
            lines += [f"- `{s}`: `{show(old[s])}` to `{show(v)}`" for s, v in changes.items()]
        else:
            lines.append("No variant beat the current settings, so the defaults are unchanged.")
        notes = [r for r in rows if r.get("note")]
        if notes:
            lines.append("\nAnthony's notes:\n")
            lines += [f"- `{r['variant']}`: {r['note']}" for r in notes]
        with (ROOT / "PROGRESS.md").open("a") as f:
            f.write("\n".join(lines) + "\n")
        say(f"{len(changes)} default(s) updated; details added to PROGRESS.md." if changes
            else "Nothing beat the current settings; defaults unchanged.")
    return {**result, "changes": changes, "old": old}


def main() -> int:
    parser = argparse.ArgumentParser(description="Try sound settings side by side and rate them.")
    parser.add_argument("videos", nargs="*", help="videos to use (default: all of test_videos/)")
    parser.add_argument("--variants", default=str(ROOT / "variants.json"))
    parser.add_argument("--only", nargs="*", default=[], help="run just these variant ids")
    parser.add_argument("--dry-run", action="store_true", help="test tones, no Modal, no cost")
    parser.add_argument("--page-only", action="store_true", help="rebuild the page only")
    parser.add_argument("--serve", action="store_true", help="open the page with saving ratings")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--apply", action="store_true", help="make the winners the new defaults")
    parser.add_argument("--ratings", default=str(RATINGS))
    args = parser.parse_args()

    if args.serve:
        serve(args.port)
        return 0
    if args.apply:
        apply(Path(args.ratings))
        return 0
    if args.page_only:
        say(f"Page: {build_page(json.loads((TUNING / 'manifest.json').read_text()))}")
        return 0

    spec = load_variants(Path(args.variants))
    videos = find_videos(spec, args.videos)
    manifest = run(spec, videos, TestTones() if args.dry_run else Cloud(), args.only)
    page = build_page(manifest)
    say(f"\nNew GPU time: {manifest['cost']['gpu_seconds']:.0f}s, cost about "
        f"${manifest['cost']['dollars']:.2f}.")
    say(f"Comparison page: {page}\nTo rate with saving: python tune.py --serve")
    if manifest["failures"]:
        say("Problems:\n  " + "\n  ".join(manifest["failures"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
