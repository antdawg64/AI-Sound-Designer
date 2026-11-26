"""The library of real recordings: what may be stored, and how a recording is cut to fit."""

import io

import numpy as np
import soundfile as sf

from soundlib import config as C
from soundlib import library as L

SR = 44100
CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"


def knock(at: float, seconds: float = 3.0, level: float = 0.8) -> np.ndarray:
    """A recording with one short decaying hit at `at` seconds over a faint hiss."""
    rng = np.random.default_rng(1)
    audio = 0.0005 * rng.standard_normal(int(seconds * SR)).astype(np.float32)
    n = int(0.25 * SR)
    hit = level * rng.standard_normal(n).astype(np.float32) * np.exp(-np.linspace(0, 9, n)).astype(np.float32)
    start = int(at * SR)
    audio[start:start + n] += hit
    return audio[:, None]


def hum(seconds: float = 6.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)[:, None]


def flac(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="FLAC")
    return buf.getvalue()


def test_only_cc0_without_voices_or_music_is_kept():
    assert L.keep_clip(CC0, ["Door", "Slam"], "door slam", ["door"])[0]
    assert not L.keep_clip("http://creativecommons.org/licenses/by/3.0/", ["Door"])[0]
    assert not L.keep_clip("http://creativecommons.org/licenses/by-nc/3.0/", ["Door"])[0]
    assert not L.keep_clip(CC0, ["Human_voice", "Speech"])[0]
    assert not L.keep_clip(CC0, ["Music", "Guitar", "Musical_instrument"])[0]
    assert L.keep_clip(CC0, ["Music", "Bell", "Doorbell"])[0]
    assert not L.keep_clip(CC0, ["Door"], "door from a movie scene")[0]
    assert not L.keep_clip("", ["Door"])[0]


def test_measurements_tell_a_hit_from_a_hum():
    hit, steady = L.clip_stats(knock(1.0), SR), L.clip_stats(hum(), SR)
    assert hit["range_db"] > 15 and hit["events"] == 1
    assert steady["steady_db"] < 2 and steady["seconds"] == 6.0


def test_the_loudest_of_several_events_is_taken_alone():
    audio = knock(0.5, level=0.3) + knock(1.5, level=0.9) + knock(2.4, level=0.4)
    start, peak, end = L.main_event(audio, SR, longest=2.0)
    assert 1.4 < start / SR < 1.51 and 1.5 <= peak / SR < 1.6
    assert end / SR < 2.4          # stops before the next knock


def test_a_moment_starts_on_the_action():
    seg = {"index": 1000, "time": 5.0, "clip_start": 4.75, "clip_end": 6.25}
    out = L.fit(knock(1.2), SR, seg, "moment")
    assert len(out) == int(round(1.5 * SR))
    loud = np.abs(out[:, 0]) > 0.05
    assert abs(np.argmax(loud) / SR - 0.25) < 0.03     # 0.25 s into the clip = the moment


def test_a_whoosh_peaks_on_the_moment_and_a_detail_follows_it():
    seg = {"index": 2000, "role": "whoosh", "time": 5.0, "clip_start": 4.0, "clip_end": 5.5}
    out = L.fit(knock(1.2), SR, seg, "whoosh")
    assert abs(np.abs(out[:, 0]).argmax() / SR - 1.0) < 0.02
    seg = {"index": 2001, "role": "detail", "time": 5.0, "clip_start": 4.9, "clip_end": 6.4}
    out = L.fit(knock(1.2), SR, seg, "detail")
    assert 0.1 < np.argmax(np.abs(out[:, 0]) > 0.05) / SR < 0.2


def test_a_background_fills_the_whole_length_even_from_a_short_recording():
    seg = {"index": 3000, "ambience": True, "clip_start": 0.0, "clip_end": 8.0}
    air = (0.1 * np.random.default_rng(2).standard_normal(3 * SR)).astype(np.float32)[:, None]
    out = L.fit(air, SR, seg, "ambience")
    assert len(out) == 8 * SR
    env = L.envelope_db(out[int(0.2 * SR):-int(0.2 * SR)], SR, 0.1)
    assert env.max() - env.min() < 4               # no holes or bumps at the joins


def test_kinds_follow_the_numbering_in_make_sound():
    assert L.kind_of({"index": 3}) is None
    assert L.kind_of({"index": 500, "separate": True}) == "source"
    assert L.kind_of({"index": 1002}) == "moment"
    assert L.kind_of({"index": 2010, "role": "whoosh"}) == "whoosh"
    assert L.kind_of({"index": 2011, "role": "detail"}) == "detail"
    assert L.kind_of({"index": 3000, "ambience": True}) == "ambience"


class Backend:
    def __init__(self):
        self.api_calls = []


def test_switched_off_changes_nothing():
    seg = {"index": 1000, "time": 1.0, "clip_start": 0.75, "clip_end": 2.25}
    C.override({"sfx_library": "off"})
    try:
        assert L.cache_tag(seg) == []
        assert L.find(Backend(), "door slam", seg, search=lambda *a: 1 / 0) is None
        assert "library" not in seg
    finally:
        C.reset()


def test_find_uses_a_chosen_recording_and_records_where_it_came_from():
    clip = {"id": "123", "source": "fsd50k", "title": "door slam", "author": "someone",
            "licence": "CC0 1.0", "url": "https://freesound.org/s/123/", "seconds": 3.0, "labels": ["Door"]}
    search = lambda asks, opts: {"dollars": 0.01, "model": "m", "results": [
        {"clip": clip, "flac": flac(knock(1.0)), "score": 0.5, "why": "same sound"}]}
    C.override({"sfx_library": "on"})
    try:
        backend, seg = Backend(), {"index": 1000, "time": 1.0, "clip_start": 0.75, "clip_end": 2.25}
        audio, sr = L.find(backend, "door slam", seg, search=search)
        assert sr == SR and len(audio) == int(round(1.5 * SR))
        assert seg["library"]["used"] and seg["library"]["licence"] == "CC0 1.0"
        assert backend.api_calls[0]["dollars"] == 0.01
        assert L.used([seg, None, dict(seg)]) == [{k: clip[k] for k in ("id", "source", "title", "author", "licence", "url")}]
        assert L.cache_tag(seg) and L.cache_tag({"index": 2, "clip_start": 0}) == []

        miss = {"index": 1001, "time": 1.0, "clip_start": 0.75, "clip_end": 2.25}
        nothing = lambda asks, opts: {"results": [{"why": "nothing fits", "best_score": 0.1}]}
        assert L.find(Backend(), "alien gargle", miss, search=nothing) is None
        assert miss["library"] == {"used": False, "why": "nothing fits", "best_score": 0.1}

        broken = {"index": 1002, "time": 1.0, "clip_start": 0.75, "clip_end": 2.25}
        assert L.find(Backend(), "door slam", broken, search=lambda *a: 1 / 0) is None
        assert not broken["library"]["used"]
    finally:
        C.reset()
