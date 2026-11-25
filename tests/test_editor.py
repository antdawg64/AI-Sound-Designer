"""Checks for the editor's sound handling. Run: python -m pytest tests/ -q  (free, no Modal)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soundlib import config as C  # noqa: E402
from soundlib import editor, mixing  # noqa: E402

SR = C.SAMPLE_RATE
DURATION = 6.0


def _noise(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal((int(seconds * SR), 2))).astype(np.float32)


def _pieces():
    segments = [
        {"index": 0, "shot": 0, "shots": [0], "start": 0.0, "end": 3.0, "clip_start": 0.0,
         "clip_end": 3.3, "gain_db": 0.0, "prompt": "hum", "audio": _noise(3.3, 1)},
        {"index": 1, "shot": 1, "shots": [1], "start": 3.0, "end": 6.0, "clip_start": 2.7,
         "clip_end": 6.0, "gain_db": -6.0, "prompt": "steps", "audio": _noise(3.3, 2)},
    ]
    spots = [{"index": 1000, "time": 2.0, "prompt": "click", "gain_db": 0.0, "clip_start": 1.75,
              "clip_end": 3.25, "audio": _noise(1.5, 3)}]
    levels = [{"start": 0.0, "end": 3.0, "loudness": "loud"}, {"start": 3.0, "end": 6.0, "loudness": "quiet"}]
    return segments, spots, levels


def _old_way(segments, spots, levels, target):
    sfx = mixing.stitch_effects(segments, DURATION)
    sfx = mixing.even_out(sfx, float(C.SFX_EVEN_OUT), float(C.SFX_EVEN_OUT_MAX_DB))
    sfx = mixing.balance_shots(sfx, levels)
    sfx = mixing.add_spots(sfx, spots)
    sfx = mixing.add_reverb(sfx, "small room", float(C.SFX_REVERB))
    return mixing.set_loudness(sfx, target)


def test_layers_add_up_to_the_effects_track():
    for changes in ({}, {"sfx_even_out": 0.6, "spot_bed_dip_db": -6.0, "sfx_reverb": 0.3}):
        before = {k: C.get(k) for k in changes}
        C.override(changes)
        try:
            segments, spots, levels = _pieces()
            layers = mixing.effects_layers(segments, spots, levels, "small room", DURATION, -30.0)
            assert len(layers) == 3
            assert np.abs(sum(layers) - _old_way(segments, spots, levels, -30.0)).max() < 1e-5
        finally:
            C.override(before)


def test_build_edit_and_tuning_record(tmp_path):
    segments, spots, levels = _pieces()
    layers = mixing.effects_layers(segments, spots, levels, "small room", DURATION, -30.0)
    music = _noise(DURATION, 9) * 0.1
    plan = {"summary": "s", "shots": [{"shot": 0, "effects": [{"loudness": "loud"}]},
                                      {"shot": 1, "effects": [{"loudness": "quiet"}]}],
            "moments": [{"time": 2.0, "loudness": "loud"}]}
    manifest = editor.build(tmp_path, segments=segments, spots=spots, layers=layers, music=music,
                            voice_wav=None, shots=[{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 6.0}],
                            duration=DURATION, plan=plan, caption="tense", parts="both", work=tmp_path)
    assert [c["id"] for c in manifest["clips"]] == ["bg0", "bg1", "m0"]
    assert [c["loudness"] for c in manifest["clips"]] == ["loud", "quiet", "loud"]

    untouched = editor.clean_state({}, manifest)
    assert not editor.is_changed(untouched)
    track = editor.effects_track(tmp_path, manifest, untouched)
    assert np.abs(track - sum(layers)).max() < 1e-3          # 16-bit files

    state = editor.clean_state({"music_db": 99, "effects_db": "x", "clips": {
        "m0": {"start": 4.0, "gain_db": -3}, "bg1": {"deleted": True}, "nope": {"start": 1}}}, manifest)
    assert state["music_db"] == 12.0 and state["effects_db"] == 0.0
    assert set(state["clips"]) == {"m0", "bg1"}
    moved = editor.effects_track(tmp_path, manifest, state)
    assert np.abs(moved[int(5.9 * SR):]).max() == 0.0        # bg1 is gone
    assert np.abs(moved[int(4.1 * SR):int(5.4 * SR)]).max() > 0   # m0 now plays here

    record = editor.tuning_record(manifest, state, job={"id": "j"}, report=None, session="s",
                                  events=[{"t": 1, "type": "move", "clip": "m0"}, "junk"], renders=1)
    by_id = {c["id"]: c for c in record["clips"]}
    assert by_id["m0"]["moved_seconds"] == round(4.0 - manifest["clips"][2]["start"], 3)
    assert by_id["bg1"]["deleted"] and record["changed"] and len(record["events"]) == 1
