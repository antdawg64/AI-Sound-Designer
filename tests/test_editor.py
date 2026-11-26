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

    first = editor.initial_pieces(manifest)
    assert [p["id"] for p in first] == ["bg0", "bg1", "m0"]
    untouched = editor.clean_state({}, manifest)
    assert not editor.is_changed(untouched, manifest)
    whole = sum(layers)
    assert np.abs(editor.effects_track(tmp_path, manifest, untouched) - whole).max() < 1e-3   # 16-bit files

    # Cutting a piece in two and leaving both halves in place changes nothing you can hear.
    cut = [dict(p) for p in first]
    left, right = dict(cut[0], id="a", out=1.3), dict(cut[0], id="b", start=cut[0]["start"] + 1.3)
    right["in"] = 1.3
    state = editor.clean_state({"pieces": [left, right, *cut[1:]]}, manifest)
    assert editor.is_changed(state, manifest)
    assert np.abs(editor.effects_track(tmp_path, manifest, state) - whole).max() < 1e-3

    # Move the moment, remove bg1, duplicate the moment, and send some nonsense.
    m0 = dict(cut[2], start=4.0, gain_db=-3)
    copy = dict(cut[2], id="copy1", start=0.5)
    state = editor.clean_state({"music_db": 99, "effects_db": "x", "pieces": [
        cut[0], dict(cut[1], deleted=True), m0, copy, {"src": "nope"}, "junk",
        dict(cut[0], id="bad", **{"in": 5.0, "out": 1.0})]}, manifest)
    assert state["music_db"] == 12.0 and state["effects_db"] == 0.0
    assert [p["id"] for p in state["pieces"]][:4] == ["bg0", "bg1", "m0", "copy1"]
    moved = editor.effects_track(tmp_path, manifest, state)
    assert np.abs(moved[int(5.9 * SR):]).max() == 0.0             # bg1 is gone
    assert np.abs(moved[int(4.1 * SR):int(5.4 * SR)]).max() > 0   # m0 now plays here

    record = editor.tuning_record(manifest, state, job={"id": "j"}, report=None, session="s",
                                  events=[{"t": 1, "type": "move", "clip": "m0"}, "junk"], renders=1)
    by_id = {c["id"]: c for c in record["clips"]}
    assert by_id["m0"]["moved_seconds"] == round(4.0 - manifest["clips"][2]["start"], 3)
    assert by_id["m0"]["origin"] == "ai" and by_id["copy1"]["origin"] == "copy"
    assert by_id["bg1"]["deleted"] and record["changed"] and len(record["events"]) == 1


def test_background_clips_are_cut_per_shot():
    manifest = {"duration": 10.0, "shots": [{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 7.0},
                                            {"start": 7.0, "end": 10.0}],
                "shot_labels": ["hum", "steps", "door"],
                "clips": [{"id": "bg0", "kind": "background", "label": "hum, steps, door", "start": 0.0,
                           "duration": 10.0, "shots": [0, 1, 2], "file": "bg0.flac"},
                          {"id": "m0", "kind": "moment", "label": "click", "start": 2.0, "duration": 1.5,
                           "file": "m0.flac"}]}
    pieces = editor.initial_pieces(manifest)
    assert [(p["id"], p["in"], p["out"], p["label"]) for p in pieces] == [
        ("bg0.0", 0.0, 3.0, "hum"), ("bg0.1", 3.0, 7.0, "steps"), ("bg0.2", 7.0, 10.0, "door"),
        ("m0", 0.0, 1.5, "click")]


# --- Pro mix ---------------------------------------------------------------------------

def _talking(seconds: float, spans) -> np.ndarray:
    """A stand-in for a voice: a buzzy tone that is only there during `spans`."""
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((150, 300, 1200, 2400), start=1))
    gate = np.zeros_like(t)
    for a, b in spans:
        gate[int(a * SR):int(b * SR)] = 1.0
    return (0.1 * tone * gate)[:, None].repeat(2, axis=1).astype(np.float32)


def test_pro_mix_makes_room_only_while_someone_talks():
    voice = mixing.set_loudness(_talking(DURATION, [(2.0, 4.0)]), C.VOICE_LUFS)
    music, sfx = _noise(DURATION, 5), _noise(DURATION, 6)
    presence = mixing.talk_presence(voice)
    assert presence[: int(1.5 * SR)].max() < 0.05 and presence[int(5.6 * SR):].max() < 0.05
    assert presence[int(2.0 * SR):int(4.0 * SR)].min() > 0.95      # already down when the words start
    out_sfx, out_music = mixing.make_room(sfx, music, presence)
    assert np.allclose(out_music[:SR], music[:SR], atol=1e-4)      # untouched away from the talking
    level = lambda x: 20 * np.log10(np.sqrt(np.mean(x[int(2.5 * SR):int(3.5 * SR)] ** 2)))
    assert level(music) - level(out_music) > float(C.MUSIC_DUCK_DB) - 0.5
    assert abs((level(sfx) - level(out_sfx)) - float(C.SFX_DUCK_DB)) < 0.5


def test_pro_mix_guarantees_the_gap_to_the_talking():
    voice = mixing.set_loudness(_talking(DURATION, [(1.0, 5.0)]), C.VOICE_LUFS)
    loud_music, loud_sfx = _noise(DURATION, 5), _noise(DURATION, 6)   # far too loud
    parts = mixing.pro_premix(loud_sfx, loud_music, voice, voice, True)
    assert parts["dialogue"]["tightest_db"] >= float(C.DIALOGUE_MIN_GAP_DB) - 1.0
    quiet = mixing.pro_premix(loud_sfx * 0.005, loud_music * 0.005, voice, voice, True)
    assert quiet["dialogue"]["typical_db"] > 30                     # nothing extra taken off


def test_layered_effects_are_separate_clips(tmp_path):
    segments, spots, levels = _pieces()
    spots.append({"index": 2000, "time": 2.0, "moment": 0, "role": "whoosh", "prompt": "air rush",
                  "gain_db": -6.0, "clip_start": 1.0, "clip_end": 2.5, "audio": _noise(1.5, 4)})
    ambience = {"index": 3000, "ambience": True, "prompt": "room tone", "clip_start": 0.0,
                "clip_end": 4.0, "audio": _noise(4.0, 8)}
    layers = mixing.effects_layers(segments, spots, levels, "small room", DURATION, -30.0, ambience=ambience)
    assert len(layers) == 5
    whoosh = layers[3]
    assert not np.any(whoosh[int(2.2 * SR):])                       # gone just after the moment
    bed = layers[4]
    assert np.abs(bed[int(1 * SR):int(5 * SR)]).max() > 0           # runs the whole way
    plan = {"summary": "s", "shots": [], "moments": [{"time": 2.0, "description": "click", "loudness": "loud"}]}
    manifest = editor.build(tmp_path, segments=segments, spots=spots, ambience=ambience, layers=layers,
                            music=_noise(DURATION, 9) * 0.1, voice_wav=None, shots=[
                                {"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 6.0}],
                            duration=DURATION, plan=plan, caption="music", parts="both", work=tmp_path)
    kinds = {c["id"]: (c["kind"], c.get("role")) for c in manifest["clips"]}
    assert kinds["m0"] == ("moment", "main") and kinds["m0w0"] == ("moment", "whoosh")
    assert kinds["amb"] == ("ambience", None)
    state = editor.clean_state(None, manifest)
    assert np.abs(editor.effects_track(tmp_path, manifest, state) - sum(layers)).max() < 2e-4


def test_one_sharp_sound_does_not_quieten_the_rest():
    """A layer with a few very tall peaks used to drag the whole effects track down."""
    from soundlib import config as C, mixing
    sr = C.SAMPLE_RATE
    rng = np.random.default_rng(3)
    steady = (0.05 * rng.standard_normal((6 * sr, 2))).astype(np.float32)
    snaps = np.zeros((2 * sr, 2), dtype=np.float32)
    burst = rng.standard_normal((int(0.03 * sr), 2)) * np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr))[:, None]
    for at in (0.5, 0.9, 1.3):          # three sharp snaps over a quiet rustle
        snaps[int(at * sr): int(at * sr) + len(burst)] = burst
    snaps += (0.004 * rng.standard_normal(snaps.shape)).astype(np.float32)
    segs = [{"index": 500, "separate": True, "source": "air", "prompt": "air", "loudness": "loud", "position": "center",
             "start": 0.0, "end": 6.0, "clip_start": 0.0, "clip_end": 6.0, "audio": steady, "gain_db": 0.0}]
    quiet = mixing.effects_layers(segs, [], None, "outdoors", 6.0, -30.0)
    spot = {"index": 1000, "time": 3.0, "prompt": "snap", "gain_db": 0.0, "clip_start": 2.5, "clip_end": 4.5, "audio": snaps}
    both = mixing.effects_layers(segs, [spot], None, "outdoors", 6.0, -30.0)
    def level(x): return 10 * np.log10(np.mean(x[: 2 * sr] ** 2) + 1e-12)
    assert abs(level(both[0]) - level(quiet[0])) < 3.0          # the steady sound keeps its level
    assert np.abs(sum(both)).max() <= 10 ** (-1.4 / 20)
