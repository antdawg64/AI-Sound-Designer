"""Automatic checks from the Phase 1 'done when' list."""

from __future__ import annotations

from pathlib import Path

from . import config as C
from .media import probe, read_audio
from .mixing import lufs, measure_file


def check_outputs(source: Path, out_dir: Path) -> dict:
    """Check one finished job. Returns {'passed': bool, 'checks': [...]}."""
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    final, music, sfx = out_dir / "final.mp4", out_dir / "music.wav", out_dir / "sfx.wav"
    missing = [p.name for p in (final, music, sfx) if not p.exists() or p.stat().st_size == 0]
    add("files exist", not missing, "all three files written" if not missing else f"missing: {missing}")
    if missing:
        return {"passed": False, "checks": checks}

    src, out = probe(source), probe(final)
    # Both the picture and the sound of the output must match the input length.
    lengths = [out["duration"], out["audio_duration"] or 0.0]
    worst = max(abs(x - src["duration"]) for x in lengths)
    add("length matches input", worst <= C.CHECK_DURATION_TOLERANCE,
        f"input {src['duration']:.3f}s, output video {lengths[0]:.3f}s, audio {lengths[1]:.3f}s")

    m = measure_file(final)
    add("loudness is -14 LUFS (+/-1)", abs(m["lufs"] - C.TARGET_LUFS) <= C.CHECK_LUFS_TOLERANCE,
        f"{m['lufs']:.1f} LUFS")
    add("no clipping", m["true_peak_db"] <= C.CHECK_MAX_TRUE_PEAK_DB,
        f"true peak {m['true_peak_db']:.1f} dBTP")

    for name, path in (("music stem is not silent", music), ("effects stem is not silent", sfx)):
        data, sr = read_audio(path)
        level = lufs(data, sr)
        length_ok = abs(len(data) / sr - src["duration"]) <= C.CHECK_DURATION_TOLERANCE
        add(name, level > C.CHECK_SILENT_LUFS and length_ok,
            f"{level:.1f} LUFS, {len(data) / sr:.2f}s long")

    return {"passed": all(c["passed"] for c in checks), "checks": checks, "final": m}
