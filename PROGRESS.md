# Progress

_Last updated: 2026-11-24_

## Where things stand

Phase 1 (the one-command prototype) is **written but not finished**. All the code
exists. Everything that runs without a GPU has been tested and passes. The AI parts
(Claude planning, HunyuanVideo-Foley effects, ACE-Step music) have **never been run**,
because the cloud session it was built in could not connect to Modal. See "Blocked".

## What works (tested)

- Phase 0: CLAUDE.md, this file, `requirements.txt`, `scripts/setup.sh`, `.env.example`.
- `python make_sound.py video.mp4 --dry-run` runs the whole pipeline end to end with
  test tones standing in for the AI models, and writes `final.mp4`, `music.wav`,
  `sfx.wav`, `plan.json` and `report.json` into `outputs/<video name>/`.
- Tested this way on `mitey_vacuum_short_living_room_silent.mp4` (28 s, 13 shots, no
  audio) and on a made-up talking clip. On both, every automatic check passed:
  length within 0.1 s (28.000 s in, 28.000 s out), loudness -14.1 LUFS, true peak
  -1.8 dBTP, neither stem silent.
- Shot detection (PySceneDetect), speech detection (Silero VAD, MIT, bundled in
  `soundlib/assets/`), key frame extraction, cutting shots into small clips for the
  effects model, crossfaded stitching, tempo picking (beats land about 0.02 s from the
  cuts on the test video), the ffmpeg mix with voice ducking, two-pass loudness
  normalisation, and the automatic checks.

## Written but NOT yet run

- `modal_app.py`: the Modal side. `plan_sound` (Claude Opus 5.5 via the `anthropic`
  Modal secret), `Foley` (HunyuanVideo-Foley XXL on an L40S), `Music` (ACE-Step 1.5
  turbo on an L40S). Weights cache in the Modal volume `ai-sound-designer-weights`.
- Expect to debug the two model images on first deploy: package versions for each
  model were taken from their own requirement files but have not been built.
- After the first successful build, pin both model repos to the commit that the image
  build prints (they are cloned at the latest version right now).

## Blocked

Modal cannot be reached from a cloud session on the `Default` environment. The three
keys were saved there as "Network secrets", which forces traffic through an
inspecting proxy that only speaks HTTP/1.1. Modal needs HTTP/2, so every call fails
with "Connection lost". The keys are also not visible as environment variables.

A separate cloud environment, `ai-sound-designer`, was created to fix this. It lets
Modal through untouched. It still needs the keys, and the project has to be switched
to it (see "Needed from Anthony").

## Needed from Anthony

- [ ] In Project settings > Cloud environment, pick **ai-sound-designer**.
- [ ] In that environment (gear icon), add these as **Environment variables**, not
      Network secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`.
- [ ] Then ask Claude to "finish Prompt 1". It has to be a new thread, because a
      running thread cannot change environment.
- [x] Modal account with a spending limit.
- [x] Claude API key saved as the Modal secret `anthropic` (not yet tested).
- [x] Test video (one provided instead of five; its sound was removed).
- Later (Phase 5): Stripe account.

## Next steps for Claude (in the new thread)

1. Check `env | grep MODAL` and that `modal profile current` connects. Copy the test
   video into place (it is not in git):
   `cp /mnt/project-files/mitey_vacuum_short_living_room_silent.mp4 test_videos/`
2. `python make_sound.py --deploy`, fix any image build problems.
3. `python make_sound.py --all`. Uploads over about 2 MB go to Modal's storage hosts
   (`*.r2.cloudflarestorage.com`, `*.amazonaws.com`), which the environment does not
   allow. The code avoids this by sending small clips and streaming audio back in
   pieces, so it should not come up. If it does, ask Anthony to allow those hosts.
4. Fill in the cost table below with real GPU seconds and dollars, check the Modal
   prices in `soundlib/config.py` against modal.com/pricing, pin the model repos.
5. Copy `final.mp4`, `music.wav` and `sfx.wav` to the project files for Anthony.

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
