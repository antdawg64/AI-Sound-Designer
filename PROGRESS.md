# Progress

_Last updated: 2026-11-24_

## Where things stand

Phase 1 (the one-command prototype) is **done and working for real**, as of
2026-11-24. `python make_sound.py video.mp4` plans the sound with Claude, generates
synced effects with HunyuanVideo-Foley and music with ACE-Step 1.5 on Modal, mixes
them and checks the result. Waiting for Anthony to listen and say whether to start
Phase 2.

## What works (tested)

- Real end-to-end run on `mitey_vacuum_short_living_room_silent.mp4` (28 s, 13 shots,
  no speech). All automatic checks pass: length 28.000 s in and out, loudness
  -13.8 LUFS, true peak -1.7 dBTP, neither stem silent.
- Outputs for Anthony: `/mnt/project-files/prompt1_output/`
  (`mitey_vacuum_with_sound.mp4`, `music.wav`, `sfx.wav`). In the repo they are in
  `outputs/<video name>/` (git-ignored) with `plan.json` and `report.json`.
- A run takes about 3.5 minutes when the GPU machines are cold, about 150 GPU seconds.
- `--dry-run` still runs everything with test tones, no Modal, no cost.
- Modal app `ai-sound-designer` is deployed in Anthony's workspace `anthony-56415`:
  `plan_sound` (Claude Opus 5.5), `Foley` and `Music` (L40S), plus `prepare_foley` /
  `prepare_music`, which download weights on a CPU machine. Weights (18 GB) are in the
  Modal volume `ai-sound-designer-weights`. Both model repos are pinned to a commit.

## Things learned while getting it running

- The Claude key is in the Modal secret **`anthropic-secret`** (not `anthropic`).
- Modal sign-in from a cloud session: `modal token new` loses its connection while
  waiting. Use `python scripts/modal_login.py` instead (run it in the background; it
  prints the link, then polls with short waits and retries). The token is stored in
  `~/.modal.toml`, which does not survive into a new thread, so a new thread needs a
  new sign-in link from Anthony (or the keys as environment variables).
- The cloud environment must allow `input-plane.us-east.modal.com` (added, along with
  us-west). Modal sends function calls there, not to `api.modal.com`.
- Modal needs a payment method on the account before it will create GPU functions.
- Build both model images from one process. Two processes building the same image at
  once get "terminated due to external shut-down".
- The effects model returns each clip at its own level (some 30 dB quieter), so every
  clip is levelled before mixing (`SFX_CLIP_RMS_DB`). The music model started with an
  8 s near-silent intro, so slow level swings are evened out (`MUSIC_EVEN_OUT`).

## Anthony's feedback on the first result (2026-11-24)

Effects mostly match and mostly land on time, "but not everything" (no specifics yet).
Music was wrong (whimsical; it should have been suspenseful) and too loud. Fixed and
rerun: the planner now picks the mood from what is at stake in the story, not from how
the picture looks, and the music sits 8 dB under the effects (`MUSIC_LUFS_NO_VOICE`
-26). Run 3 gave a "cinematic orchestral thriller score, tense, suspenseful". Output:
`/mnt/project-files/prompt1_output/mitey_vacuum_with_sound_v2.mp4` (plus `music_v2.wav`,
`sfx_v2.wav`). Anthony has not heard v2 yet.

## Not done / known limits

- Only one test video exists, with no speech. Voice ducking has only been tested with
  test tones on a made-up talking clip. Four more videos are wanted for Phase 2.
- Nobody has listened to the result yet. Claude cannot hear; only levels, lengths and
  loudness were checked. Sound quality and sync need Anthony's ears.
- Modal prices in `soundlib/config.py` are list prices from memory, not re-checked
  (modal.com is not reachable from the cloud session). The cost shown per run counts
  only model loading and generation. Machine start-up and the 60 s idle time before a
  GPU machine shuts down are billed too, so the real figure is higher. Check the
  Modal usage page for the true number.

## Needed from Anthony

- [ ] Watch `prompt1_output/mitey_vacuum_with_sound.mp4` and say what sounds right
      and wrong.
- [ ] Four more test videos (some with talking, some with fast cuts) before Phase 2.
- [ ] Say "start Prompt 2" when ready.
- Later (Phase 5): Stripe account.

## Licence check (2026-11-24)

Read each model's own LICENSE and README. Both match CLAUDE.md.

- HunyuanVideo-Foley: Tencent Hunyuan Community License. Commercial use allowed.
  Not licensed in the EU, UK or South Korea, and outputs may not be used there
  either. Needs a licence from Tencent above 100 million monthly users. Its helper
  models are downloaded separately: `google/siglip2-base-patch16-512` and
  `laion/larger_clap_general` (both listed as Apache 2.0 on Hugging Face; re-check
  when they are first downloaded).
- ACE-Step 1.5: MIT. Only the music model is used, not its optional lyric planner.
- No Hugging Face licence pages need accepting for Phase 1 (neither model is gated).

## Design notes

- The test video has 13 shots in 28 s. Each shot is sent to the effects model as its
  own short clip (1 to 5 s). Very short clips may sound thin; if so, Phase 2 should
  try generating over longer windows that span several shots.
- Original audio is kept only where speech is detected, and is treated as the voice.
- Cost per video will be the sum of GPU seconds (including model loading on a cold
  start) times the L40S price, plus the Claude call.

## Cost log

| Date | What | GPU seconds | Cost |
|------|------|-------------|------|
| 2026-11-24 | Phase 0 setup | 0 | $0 |
| 2026-11-24 | Phase 1 code and test-tone runs (no Modal use) | 0 | $0 |
| 2026-11-24 | Model downloads and image builds (CPU only) | 0 | not measured, cents |
| 2026-11-24 | Real run 1 on the vacuum video | 216 | $0.18 est. (GPU $0.12 + Claude $0.07) |
| 2026-11-24 | Real run 2 after level fixes | 147 | $0.14 est. (GPU $0.08 + Claude $0.07) |
| 2026-11-24 | Two Claude plan tests | 0 | $0.13 |
| 2026-11-24 | Real run 3 after Anthony's feedback | 161 | $0.15 est. |
