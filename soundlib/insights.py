"""What people change in the editor, turned into findings and suggested defaults.

Every editing session in the web editor leaves one small record (soundlib/editor.py,
tuning_record): which sounds were removed, muted, moved or turned up or down, and the
volume changes. No video, no audio, nothing about the person.

  read_records / group   load the records, one per edited video
  analyze                findings in plain language + suggested setting changes
  bundle                 one zip with the raw records, two spreadsheets and the summary
  report_html            the same findings as a web page (the /insights page)

Used by analyze_edits.py on a computer and by web_app.py for the /insights page.
"""

from __future__ import annotations

import csv
import html
import io
import json
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from . import config as C


def read_records(raw: Path) -> list[dict]:
    """Every saved editing-session record under raw/edits/."""
    records = []
    for path in sorted(Path(raw).glob("edits/*/*.json")):
        try:
            record = json.loads(path.read_text())
        except ValueError:
            continue
        record.setdefault("job_id", path.parent.name)
        records.append(record)
    return records


def group(records: list[dict], include_tests: bool = False) -> list[dict]:
    """One record per video: the latest editing session, with every session's events."""
    by_job: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("test_tones") and not include_tests:
            continue
        by_job[record.get("job_id") or "unknown"].append(record)
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
            "kept_audio": job.get("kept_audio"), "original_db": mix.get("voice_db", 0.0),
            "original_muted": mix.get("voice_muted", False),
            "music_vs_effects_db": round(mix["music_db"] - fx_change, 2),
            "clips": len(job["clips"]),
            "cut_by_person": sum(c.get("origin") == "cut" for c in job["clips"]),
            "copies": sum(c.get("origin") == "copy" for c in job["clips"]),
            "asked_for": sum(c.get("origin") == "asked" for c in job["clips"]),
            "deleted": sum(c["deleted"] for c in job["clips"]),
            "muted": sum(c["muted"] for c in job["clips"]),
            "moved": sum(abs(c["moved_seconds"]) > 0.02 for c in job["clips"]),
            "volume_changed": sum(bool(c["gain_db"]) for c in job["clips"]),
            "music": (job.get("music") or {}).get("label"),
        })
        for c in job["clips"]:
            clip_rows.append({"job_id": job["job_id"], **{k: c.get(k) for k in (
                "id", "src", "origin", "kind", "role", "label", "loudness", "start", "duration", "level_db",
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
    early: dict = {}      # what the edits lean towards before there are enough of them

    def suggest(name: str, value, count: int, needed: int) -> None:
        (suggestions if enough and count >= needed else early)[name] = value

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
        for voice, name in ((False, "music_lufs_no_voice"), (True, "music_lufs")):
            mine = [j["music_vs_effects_db"] for j in with_both if bool(j["has_voice"]) == voice]
            if mine and abs(_median(mine)) >= 1.0:
                suggest(name, round(float(C.get(name)) + _median(mine), 1), len(mine), min_jobs)
    lines.append("")

    # 2. Sounds people take out or change, by kind.
    lines += ["## Which sounds get removed or changed", ""]
    for c in clip_rows:   # a supporting layer of a key moment is judged apart from the moment itself
        c["group"] = (c["role"] if c["kind"] == "moment" and c.get("role") in ("whoosh", "detail") else c["kind"])
    for kind, title in (("moment", "Key moments"), ("whoosh", "Lead-in layers of key moments"),
                        ("detail", "Detail layers of key moments"), ("background", "Background sounds"),
                        ("ambience", "The steady background of the place")):
        mine = [c for c in clip_rows if c["group"] == kind and c.get("origin") not in ("copy", "asked")]
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
        gain = _median(gains) or 0.0
        if kind == "moment" and abs(gain) >= 1.0:
            suggest("spot_gain_db", round(float(C.SPOT_GAIN_DB) + gain, 1), len(stayed), 2 * min_jobs)
        if kind in ("whoosh", "detail") and abs(gain) >= 1.0:
            levels = dict((suggestions if "moment_layer_db" in suggestions else early).get(
                "moment_layer_db", C.MOMENT_LAYER_DB))
            levels[kind] = round(float(C.MOMENT_LAYER_DB[kind]) + gain, 1)
            suggest("moment_layer_db", levels, len(stayed), 2 * min_jobs)
        if kind == "ambience" and abs(gain) >= 1.0:
            suggest("ambience_level_db", round(float(C.AMBIENCE_LEVEL_DB) + gain, 1), len(stayed), min_jobs)
        if kind in ("whoosh", "detail", "ambience") and len(mine) >= 4 and len(gone) >= 0.6 * len(mine):
            lines.append(f"Most people take these out ({_share(len(gone), len(mine))}): consider making fewer of them.")
            lines.append("")
        if kind == "moment" and len(stayed) >= (2 * min_jobs if enough else 3):
            if moved and len(moved) >= len(stayed) / 3 and abs(_median(moved)) >= 0.08:
                lines.append(f"Key moments are usually dragged {_median(moved):+.2f} s: the planner's "
                             "timing of moments is off by about that much (see planning.py).")
                lines.append("")
        if kind == "background" and C.SFX_LEVELING == "by_shot":
            for sure in (True, False):
                levels = dict(C.SHOT_LEVEL_DB)
                for level in levels:
                    lv = [c["gain_db"] for c in stayed if c["loudness"] == level]
                    if len(lv) >= (2 * min_jobs if sure else 2) and abs(_median(lv)) >= 1.0:
                        levels[level] = round(levels[level] + _median(lv), 1)
                if levels != dict(C.SHOT_LEVEL_DB):
                    if sure and enough:
                        suggestions["shot_level_db"] = levels
                    elif not sure and "shot_level_db" not in suggestions:
                        early["shot_level_db"] = levels

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
    copies = sum(j["copies"] for j in job_rows)
    cuts = sum(j["cut_by_person"] for j in job_rows)
    acts = Counter(e.get("type") for j in jobs for e in j["events"])
    lines += ["## How people use the editor", "",
              f"- Videos where a new version was made: {_share(sum(j['renders'] > 0 for j in job_rows), n)}.",
              f"- Sounds people duplicated: {copies}. Pieces people cut themselves: {cuts}.",
              "- Actions: " + (", ".join(f"{k} {v}" for k, v in acts.most_common() if k not in ("open",)) or "none"),
              ""]

    lines += ["## Suggested setting changes", ""]
    if suggestions:
        for name, value in suggestions.items():
            lines.append(f"- `{name}`: {json.dumps(C.get(name))} -> {json.dumps(value)}")
        lines += ["", "Run `python analyze_edits.py --apply` to make these the defaults, then "
                      "`python make_sound.py --deploy`."]
    elif not enough:
        lines.append(f"Not enough edited videos yet ({n} of {min_jobs}) to change anything automatically.")
    else:
        lines.append("None: people's changes are small or don't agree with each other.")
    early = {k: v for k, v in early.items() if k not in suggestions}
    if early:
        lines += ["", "## Early signals (not applied)", "",
                  "Where the edits lean so far. Too few videos to trust yet, so these are never "
                  "applied automatically; they are worth trying as a tile in a tuning round.", ""]
        lines += [f"- `{name}`: {json.dumps(C.get(name))} -> {json.dumps(value)}" for name, value in early.items()]
    for c in clip_rows:
        c.pop("group", None)
    return clip_rows, job_rows, lines, suggestions


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --- Sharing the findings -------------------------------------------------------------

def csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def study(records: list[dict], min_jobs: int = 10, include_tests: bool = False) -> dict:
    """Everything worked out from a set of records, in one dict."""
    jobs = group(records, include_tests)
    clip_rows, job_rows, lines, suggestions = analyze(jobs, min_jobs)
    return {"records": len(records), "videos": len(jobs), "clip_rows": clip_rows, "job_rows": job_rows,
            "lines": lines, "suggestions": suggestions, "min_jobs": min_jobs,
            "tests_left_out": sum(bool(r.get("test_tones")) for r in records) if not include_tests else 0}


def bundle(records: list[dict], result: dict) -> bytes:
    """One zip: every raw record, clips.csv, jobs.csv and summary.md."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for record in records:
            name = f"edits/{record.get('job_id', 'unknown')}/{record.get('session', 'none')}.json"
            z.writestr(name, json.dumps(record, indent=1))
        z.writestr("clips.csv", csv_text(result["clip_rows"]))
        z.writestr("jobs.csv", csv_text(result["job_rows"]))
        z.writestr("summary.md", "\n".join(result["lines"]) + "\n")
    return out.getvalue()


def report_html(result: dict) -> str:
    """The findings as a simple page."""
    parts, in_list = [], False

    def text(line: str) -> str:
        safe = html.escape(line)
        for mark, tag in (("**", "strong"), ("`", "code")):
            pieces = safe.split(mark)
            if len(pieces) % 2:
                safe = "".join(p if i % 2 == 0 else f"<{tag}>{p}</{tag}>" for i, p in enumerate(pieces))
        return safe

    for line in result["lines"]:
        item = line.lstrip().startswith("- ")
        if in_list and not item:
            parts.append("</ul>")
            in_list = False
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            parts.append(f"<h2>{text(line[3:])}</h2>")
        elif line.startswith("### "):
            parts.append(f"<h3>{text(line[4:])}</h3>")
        elif item:
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{text(line.lstrip()[2:])}</li>")
        elif line.strip():
            parts.append(f"<p>{text(line)}</p>")
    if in_list:
        parts.append("</ul>")
    made = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    tests = (f" {result['tests_left_out']} test-tone record(s) are left out."
             if result.get("tests_left_out") else "")
    return PAGE.replace("{{BODY}}", "\n".join(parts)).replace("{{MADE}}", made).replace(
        "{{COUNT}}", f"{result['records']} editing session(s) on {result['videos']} video(s).{tests}")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>What people change · AI Sound Designer</title>
<style>
  :root { --bg:#f6f5f1; --card:#fff; --ink:#1b1b1f; --muted:#62636b; --line:#e2e0d8; --accent:#6741d9; }
  @media (prefers-color-scheme: dark) { :root { --bg:#111114; --card:#1a1a1f; --ink:#ececf1; --muted:#9a9ba6; --line:#2b2b33; --accent:#9775fa; } }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:760px; margin:0 auto; padding:28px 16px 60px; }
  h1 { font-size:28px; margin:0 0 4px; } h2 { font-size:19px; margin:30px 0 8px; } h3 { font-size:16px; margin:18px 0 4px; }
  .muted { color:var(--muted); font-size:14px; } ul { padding-left:20px; } li { margin:4px 0; }
  code { background:var(--line); border-radius:5px; padding:1px 5px; font-size:14px; }
  .box { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:6px 20px 16px; margin-top:18px; }
  .buttons { display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 0; }
  a.button { background:var(--accent); color:#fff; text-decoration:none; font-weight:600; padding:10px 16px; border-radius:10px; }
  a.button.quiet { background:transparent; color:var(--ink); border:1px solid var(--line); }
</style></head><body><main>
<a class="muted" href="/">&larr; AI Sound Designer</a>
<h1>What people change in the editor</h1>
<p class="muted">{{COUNT}} Worked out {{MADE}}. No videos, no audio and nothing about who edited are kept in these records.</p>
<div class="buttons">
  <a class="button" href="/api/insights/download">Download everything (zip)</a>
  <a class="button quiet" href="/api/insights/clips.csv">Sounds spreadsheet</a>
  <a class="button quiet" href="/api/insights/jobs.csv">Videos spreadsheet</a>
</div>
<div class="box">
{{BODY}}
</div>
</main></body></html>
"""
