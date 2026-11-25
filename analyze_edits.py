#!/usr/bin/env python3
"""Learn better defaults from what people change in the editor.

Every time someone fine-tunes a video in the web editor, a small record is kept: which
sounds they removed, muted or moved, and how they changed the volumes (no video, no
audio, nothing about who they are). This script gathers those records and shows what
people keep correcting, so the automatic sound can start closer to what they want.

    python analyze_edits.py             download the records from Modal and report
    python analyze_edits.py --local     use the records already in tuning_data/raw/
    python analyze_edits.py --apply     also write the suggested settings into settings.json
    python analyze_edits.py --include-tests    count test-tone jobs too (for testing)

It writes, into tuning_data/:
    clips.csv    one row per sound per edited video (open it in any spreadsheet)
    jobs.csv     one row per edited video
    summary.md   the findings and the suggested setting changes, in plain language

A suggestion is only made when at least --min-jobs videos (default 10) agree, and
--apply only writes those. After --apply, redeploy: python make_sound.py --deploy
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from soundlib import config as C

ROOT = Path(__file__).parent
DATA = ROOT / "tuning_data"
VOLUME = "ai-sound-designer-tuning-data"


def download(raw: Path) -> int:
    """Copy every record from the Modal volume into tuning_data/raw/."""
    from soundlib import proxyfix

    proxyfix.apply()
    import modal

    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    count = 0
    try:
        entries = volume.listdir("/edits", recursive=True)
    except Exception:   # nothing has been saved yet
        entries = []
    for entry in entries:
        if not entry.path.endswith(".json"):
            continue
        target = raw / entry.path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"".join(volume.read_file(entry.path)))
        count += 1
    return count


def load(raw: Path, include_tests: bool = False) -> list[dict]:
    """One record per video: the latest editing session, with every session's events."""
    by_job: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(raw.glob("edits/*/*.json")):
        try:
            record = json.loads(path.read_text())
        except ValueError:
            continue
        if record.get("test_tones") and not include_tests:
            continue
        by_job[record.get("job_id") or path.parent.name].append(record)
    jobs = []
    for sessions in by_job.values():
        sessions.sort(key=lambda r: r.get("saved", 0))
        last = dict(sessions[-1])
        last["sessions"] = len(sessions)
        last["events"] = [e for s in sessions for e in s.get("events", [])]
        last["renders"] = sum(int(s.get("renders", 0)) for s in sessions)
        jobs.append(last)
    return jobs


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole} ({100 * part / whole:.0f}%)" if whole else "none yet"


def analyze(jobs: list[dict], min_jobs: int) -> tuple[list[dict], list[dict], list[str], dict]:
    """Returns (clip rows, job rows, summary lines, suggested setting changes)."""
    clip_rows, job_rows = [], []
    for job in jobs:
        kept = [c for c in job["clips"] if not c["deleted"] and not c["muted"]]
        mix = job["mix"]
        # What the person did to the balance: music level compared with effects level.
        fx_change = mix["effects_db"] + (_median([c["gain_db"] for c in kept]) or 0.0)
        job_rows.append({
            "job_id": job["job_id"], "saved": job.get("saved"), "duration": job["duration"],
            "parts": job.get("parts"), "has_voice": job.get("has_voice"), "vibe": job.get("vibe"),
            "test_tones": job.get("test_tones"), "sessions": job["sessions"],
            "renders": job["renders"], "events": len(job["events"]),
            "effects_db": mix["effects_db"], "music_db": mix["music_db"],
            "effects_muted": mix["effects_muted"], "music_muted": mix["music_muted"],
            "music_vs_effects_db": round(mix["music_db"] - fx_change, 2),
            "clips": len(job["clips"]),
            "deleted": sum(c["deleted"] for c in job["clips"]),
            "muted": sum(c["muted"] for c in job["clips"]),
            "moved": sum(abs(c["moved_seconds"]) > 0.02 for c in job["clips"]),
            "volume_changed": sum(bool(c["gain_db"]) for c in job["clips"]),
            "music": (job.get("music") or {}).get("label"),
        })
        for c in job["clips"]:
            clip_rows.append({"job_id": job["job_id"], **{k: c.get(k) for k in (
                "id", "kind", "label", "loudness", "start", "duration", "level_db",
                "moved_seconds", "gain_db", "muted", "deleted")},
                "removed": c["deleted"] or c["muted"],
                "has_voice": job.get("has_voice"), "effects_db": mix["effects_db"]})

    n = len(jobs)
    lines = ["# What people change in the editor", "",
             f"Edited videos counted: **{n}**. Suggestions need at least {min_jobs}.", ""]
    suggestions: dict = {}
    if not n:
        lines.append("No edits have been saved yet.")
        return clip_rows, job_rows, lines, suggestions
    enough = n >= min_jobs

    # 1. Balance between music and effects.
    lines += ["## Music against effects", ""]
    with_both = [j for j in job_rows if j["parts"] in (None, "both")]
    balance = _median([j["music_vs_effects_db"] for j in with_both])
    music_off = sum(j["music_muted"] for j in with_both)
    fx_off = sum(j["effects_muted"] for j in with_both)
    if balance is not None:
        lines.append(f"- Typical change to the music compared with the effects: **{balance:+.1f} dB** "
                     f"(median over {len(with_both)} videos; minus means people turn the music down).")
        lines.append(f"- Music muted completely: {_share(music_off, len(with_both))}. "
                     f"Effects muted completely: {_share(fx_off, len(with_both))}.")
        if enough and abs(balance) >= 1.0:
            for voice, name in ((False, "music_lufs_no_voice"), (True, "music_lufs")):
                mine = [j["music_vs_effects_db"] for j in with_both if bool(j["has_voice"]) == voice]
                if len(mine) >= min_jobs:
                    shift = _median(mine)
                    if abs(shift) >= 1.0:
                        suggestions[name] = round(float(C.get(name)) + shift, 1)
    lines.append("")

    # 2. Sounds people take out or change, by kind.
    lines += ["## Which sounds get removed or changed", ""]
    for kind, title in (("moment", "Key moments"), ("background", "Background sounds")):
        mine = [c for c in clip_rows if c["kind"] == kind]
        if not mine:
            continue
        gone = [c for c in mine if c["removed"]]
        stayed = [c for c in mine if not c["removed"]]
        moved = [c["moved_seconds"] for c in stayed if abs(c["moved_seconds"]) > 0.02]
        gains = [c["gain_db"] for c in stayed]
        lines.append(f"### {title}")
        lines.append(f"- Removed or muted: {_share(len(gone), len(mine))}.")
        lines.append(f"- Moved: {_share(len(moved), len(stayed))}"
                     + (f", typically by {_median(moved):+.2f} s (plus means later)." if moved else "."))
        lines.append(f"- Volume changed: {_share(sum(bool(g) for g in gains), len(stayed))}; "
                     f"typical change over all kept sounds {(_median(gains) or 0.0):+.1f} dB.")
        for level in ("quiet", "medium", "loud"):
            lv = [c for c in mine if c["loudness"] == level]
            if lv:
                lv_gain = _median([c["gain_db"] for c in lv if not c["removed"]]) or 0.0
                lines.append(f"  - planned as {level}: removed {_share(sum(c['removed'] for c in lv), len(lv))}, "
                             f"typical volume change {lv_gain:+.1f} dB")
        lines.append("")
        if enough and kind == "moment" and len(stayed) >= 2 * min_jobs:
            gain = _median(gains) or 0.0
            if abs(gain) >= 1.0:
                suggestions["spot_gain_db"] = round(float(C.SPOT_GAIN_DB) + gain, 1)
            if len(moved) >= len(stayed) / 3 and abs(_median(moved)) >= 0.08:
                lines.append(f"Key moments are usually dragged {_median(moved):+.2f} s: the planner's "
                             "timing of moments is off by about that much (see planning.py).")
                lines.append("")
        if enough and kind == "background" and C.SFX_LEVELING == "by_shot":
            levels = dict(C.SHOT_LEVEL_DB)
            for level in levels:
                lv = [c["gain_db"] for c in stayed if c["loudness"] == level]
                if len(lv) >= 2 * min_jobs and abs(_median(lv)) >= 1.0:
                    levels[level] = round(levels[level] + _median(lv), 1)
            if levels != dict(C.SHOT_LEVEL_DB):
                suggestions["shot_level_db"] = levels

    # 3. Words that keep showing up in removed sounds.
    removed_words, all_words = Counter(), Counter()
    for c in clip_rows:
        for phrase in {p.strip().lower() for p in str(c["label"]).split(",") if p.strip()}:
            all_words[phrase] += 1
            if c["removed"]:
                removed_words[phrase] += 1
    often = [(p, k, all_words[p]) for p, k in removed_words.most_common(40) if all_words[p] >= 3 and k / all_words[p] >= 0.5]
    if often:
        lines += ["## Sounds people usually don't want", ""]
        lines += [f"- \"{p}\": removed {k} of {total} times" for p, k, total in often[:15]]
        lines.append("")

    # 4. How much editing happens at all.
    acts = Counter(e.get("type") for j in jobs for e in j["events"])
    lines += ["## How people use the editor", "",
              f"- Videos where a new version was made: {_share(sum(j['renders'] > 0 for j in job_rows), n)}.",
              "- Actions: " + (", ".join(f"{k} {v}" for k, v in acts.most_common() if k not in ("open",)) or "none"),
              ""]

    lines += ["## Suggested setting changes", ""]
    if suggestions:
        for name, value in suggestions.items():
            lines.append(f"- `{name}`: {json.dumps(C.get(name))} -> {json.dumps(value)}")
        lines += ["", "Run `python analyze_edits.py --apply` to make these the defaults, then "
                      "`python make_sound.py --deploy`."]
    elif not enough:
        lines.append(f"Not enough edited videos yet ({n} of {min_jobs}).")
    else:
        lines.append("None: people's changes are small or don't agree with each other.")
    return clip_rows, job_rows, lines, suggestions


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn better defaults from editor changes.")
    parser.add_argument("--local", action="store_true", help="don't download, use tuning_data/raw/")
    parser.add_argument("--apply", action="store_true", help="write the suggestions into settings.json")
    parser.add_argument("--min-jobs", type=int, default=10, help="videos needed before suggesting (default 10)")
    parser.add_argument("--include-tests", action="store_true", help="count test-tone jobs too")
    args = parser.parse_args()

    raw = DATA / "raw"
    if not args.local:
        print(f"Downloaded {download(raw)} record(s) from Modal.")
    jobs = load(raw, args.include_tests)
    clip_rows, job_rows, lines, suggestions = analyze(jobs, args.min_jobs)
    DATA.mkdir(exist_ok=True)
    write_csv(DATA / "clips.csv", clip_rows)
    write_csv(DATA / "jobs.csv", job_rows)
    (DATA / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nFiles: {DATA / 'summary.md'}, {DATA / 'jobs.csv'}, {DATA / 'clips.csv'}")
    if args.apply:
        if suggestions:
            C.save_defaults(suggestions)
            print(f"Applied to settings.json: {', '.join(suggestions)}. Now run: python make_sound.py --deploy")
        else:
            print("Nothing to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
