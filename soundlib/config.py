"""Settings. The values live in settings.json at the top of the project; this module
loads them and exposes each one as an upper-case name (settings "music_lufs" becomes
C.MUSIC_LUFS), which is how the rest of the code reads them.

tune.py changes settings for one run with `override(...)` and puts them back with
`reset()`. `save_defaults(...)` writes new winning values into settings.json.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [Path(os.environ["SOUND_SETTINGS"])] if os.environ.get("SOUND_SETTINGS") else []
_CANDIDATES += [_HERE.parent / "settings.json", _HERE / "settings.json"]
SETTINGS_FILE = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[-2])

_defaults: dict = {}      # setting name -> value, as in settings.json
_sections: dict = {}      # setting name -> section it lives in


def _load() -> None:
    if not SETTINGS_FILE.exists():
        raise FileNotFoundError(f"settings.json not found (looked in {SETTINGS_FILE})")
    raw = json.loads(SETTINGS_FILE.read_text())
    _defaults.clear()
    _sections.clear()
    for section, values in raw.items():
        if section.startswith("_") or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key.startswith("_"):
                continue
            _defaults[key] = value
            _sections[key] = section
    reset()


def reset() -> None:
    """Put every setting back to what settings.json says."""
    for key, value in _defaults.items():
        globals()[key.upper()] = json.loads(json.dumps(value))


def names() -> list[str]:
    return sorted(_defaults)


def get(name: str):
    return globals()[_check(name).upper()]


def default(name: str):
    return _defaults[_check(name)]


def section(name: str) -> str:
    return _sections[_check(name)]


def _check(name: str) -> str:
    name = name.lower()
    if name not in _defaults:
        raise KeyError(f"Unknown setting '{name}'. Known settings: {', '.join(names())}")
    return name


def override(changes: dict) -> None:
    """Change settings for this process only (settings.json is not touched)."""
    for name, value in changes.items():
        globals()[_check(name).upper()] = value


def current() -> dict:
    """Every setting with its value right now."""
    return {name: globals()[name.upper()] for name in names()}


def save_defaults(changes: dict) -> None:
    """Write new default values into settings.json, keeping its layout and notes."""
    text = SETTINGS_FILE.read_text()
    for name, value in changes.items():
        name = _check(name)
        if isinstance(value, (dict, list)) and not _is_flat(value):
            raise ValueError(f"'{name}' is too complex to rewrite automatically")
        pattern = re.compile(r'("' + re.escape(name) + r'"\s*:\s*)(\[[^\]]*\]|\{[^}]*\}|"(?:[^"\\]|\\.)*"|[^,\n]+)')
        text, count = pattern.subn(lambda m: m.group(1) + json.dumps(value), text, count=1)
        if count != 1:
            raise ValueError(f"Could not find '{name}' in {SETTINGS_FILE}")
    json.loads(text)  # refuse to write a broken file
    SETTINGS_FILE.write_text(text)
    _load()


def _is_flat(value) -> bool:
    items = value.values() if isinstance(value, dict) else value
    return all(not isinstance(v, (dict, list)) for v in items)


_load()
