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
    report.html  the same as a page; everything.zip holds all of it plus the raw records

The same report is on the web page at /insights for the owner's account.

A suggestion is only made when at least --min-jobs videos (default 10) agree, and
--apply only writes those. After --apply, redeploy: python make_sound.py --deploy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from soundlib import config as C
from soundlib import insights

ROOT = Path(__file__).parent
DATA = ROOT / "tuning_data"
WEB_APP = "ai-sound-designer-web"


def download(raw: Path) -> int:
    """Copy every record from Modal into tuning_data/raw/. The web app hands them over
    itself (its export_edits function), so this works from anywhere that can reach Modal."""
    from soundlib import proxyfix

    proxyfix.apply()
    import modal

    records = modal.Function.from_name(WEB_APP, "export_edits").remote()
    for record in records:
        target = raw / "edits" / str(record.get("job_id", "unknown")) / f"{record.get('session', 'none')}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record))
    return len(records)


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
    records = insights.read_records(raw)
    result = insights.study(records, args.min_jobs, args.include_tests)
    DATA.mkdir(exist_ok=True)
    (DATA / "clips.csv").write_text(insights.csv_text(result["clip_rows"]))
    (DATA / "jobs.csv").write_text(insights.csv_text(result["job_rows"]))
    (DATA / "summary.md").write_text("\n".join(result["lines"]) + "\n")
    (DATA / "report.html").write_text(insights.report_html(result))
    (DATA / "everything.zip").write_bytes(insights.bundle(records, result))
    print("\n".join(result["lines"]))
    print(f"\nFiles in {DATA}/: summary.md, report.html, jobs.csv, clips.csv, everything.zip")
    if args.apply:
        if result["suggestions"]:
            C.save_defaults(result["suggestions"])
            print(f"Applied to settings.json: {', '.join(result['suggestions'])}. Now run: python make_sound.py --deploy")
        else:
            print("Nothing to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
