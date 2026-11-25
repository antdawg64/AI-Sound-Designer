# Progress

_Last updated: 2026-11-24_

## Where things stand

Phase 1 (the one-command prototype) is done. Phase 2 (quality tuning) is **built and
run on the vacuum video**, and **done** as of 2026-11-25: Anthony rated the variants
and the four winners are the new defaults (see the tuning round at the bottom).
Nothing scored above 3 out of 5, so the sound still needs work. Do not start Phase 3
until Anthony asks.

## Phase 2: quality tuning (2026-11-25)

- **`settings.json`** now holds every setting (effect prompt style, guidance, steps,
  seed, music prompt template, tempo rule, mix levels and the rest).
  `soundlib/config.py` only loads it. The cloud side gets a copy on every deploy.
- **`python tune.py`** runs the current settings plus each variant in `variants.json`
  (each changes ONE setting) on every video in `test_videos/`, and builds the rating
  page in `tuning/` (`index.html`, small preview videos in `tuning/media/`).
- **`python tune.py --serve`** opens the page at http://localhost:8765 and saves
  ratings to `ratings.csv`. The same page is also published as a private claude.ai page
  for Anthony, where ratings save to the page's own store (collection `ratings`); Claude
  reads them back and writes `ratings.csv`.
- **`python tune.py --apply`** reads `ratings.csv`. A variant wins if its average score
  beats "Current settings" on the same videos. Winners are written to `settings.json`
  and logged at the bottom of this file.
- **`python make_sound.py video.mp4 --versions`** makes three takes (different seeds
  and music twists, listed under `version_takes` in `settings.json`).
  `--vibe suspenseful` tells the music what feeling to go for.
- **`cache/`** remembers every plan, effects clip and music track, so a variant only
  pays for what it changes. Mix-level variants are free.
- New settings added for tuning: `sfx_prompt_style` (detailed / main / short),
  `foley_group_seconds` (join quick shots into longer effects clips),
  `music_prompt_template`, `music_tempo_rule` (match_cuts / as_planned / half_time).

The 6 starting variants, chosen from Anthony's notes on the Phase 1 result (effects
mostly right, timing mostly right, music was the wrong mood and too loud):

1. Music further back: 4 dB quieter again.
2. Music allowed to build: stop flattening the music's swells.
3. Sparser music: ask for a thin underscore instead of a full track.
4. One clear effect per shot: give the effects model only the main sound.
5. Effects follow the description harder: guidance 4.5 to 6.5.
6. Effects made over longer stretches: join quick shots into clips up to 8 s.

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

- [ ] Listen to `/mnt/project-files/prompt2_tuning/new_defaults/mitey_vacuum_tuned.mp4`
      (the four winners combined) and say what is still wrong.
- [ ] Say which effects sounded wrong in the Phase 1 video (still open).
- [ ] Four more test videos (some with talking, some with fast cuts) would make the
      tuning much more trustworthy; one video with no speech is thin evidence.
- [ ] Say "start Prompt 3" when ready.
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

## Tuning round, 2026-11-25

From Anthony's ratings in `ratings.csv` (current settings scored 2.0 out of 5):

- `effects_follow_words`: 3.0 (better)
- `effects_longer_clips`: 3.0 (better)
- `effects_main_sound`: 3.0 (better)
- `music_can_build`: 3.0 (better)
- `music_further_back`: 2.0 (same)
- `music_sparse`: 2.0 (same)

New defaults written to `settings.json`:

- `foley_guidance`: `4.5` to `6.5`
- `foley_group_seconds`: `0.0` to `8`
- `sfx_prompt_style`: `detailed` to `main`
- `music_even_out`: `0.6` to `0`

After applying, the four winners were run together once on the vacuum video (all checks
pass, $0.03): `/mnt/project-files/prompt2_tuning/new_defaults/`. The "main" prompt
style was changed to keep one main sound per shot when a clip spans several shots.
Anthony's scores on the three takes: 2, 3 (sparse music), 1 (bigger music).

| 2026-11-25 | Phase 2 tuning run: baseline, 6 variants, 3 takes | 657 | $0.42 est. |
| 2026-11-25 | Phase 2 confirmation run with the new defaults | 63 | $0.03 est. |

## Improvement round 2 (2026-11-25), rated and applied

Anthony approved a round on three ideas before Prompt 3. Built on the round 1 defaults:

- `plan_detail: close`: Claude sees up to 6 frames per shot (78 frames for the vacuum
  video instead of 13) and lists exact moments; each moment gets an extra effect
  generated from a 1.5 s clip around it and laid on the effects track at that time.
  The plan also names the room, used for the echo.
- `foley_candidates: 3`: three versions of each effects clip; the CLAP model (already
  part of the effects model, Apache 2.0) scores sound against description and the best
  is kept. Claude cannot listen to audio, so Claude does not do the picking.
- `sfx_reverb: 0.25`: room echo on the effects, sized from the plan's room.
- `music_duck_under_sfx: true`: music dips under loud effects.
- Variant `all_together` changes all four at once (`tune.py` now accepts `settings`
  with several changes; it wins outright only if it scores at least as high as every
  single change).

Round 1 files were renamed `variants_round1.json` and `ratings_round1.csv`. The round 2
page is the same claude.ai page (Sound Tuning Room); copy in
`/mnt/project-files/prompt2_tuning/round2/`. After Anthony rates: copy the page's
`ratings` rows for run `20261125-014748` into `ratings.csv`, run
`python tune.py --apply`, redeploy, and stop. Do not start Phase 3.

| 2026-11-25 | Round 2 tuning run: baseline and 5 variants | 567 | $0.49 est. |

## Tuning round, 2026-11-25

From Anthony's ratings in `ratings.csv` (current settings scored 2.0 out of 5):

- `all_together`: 2.0 (same)
- `best_of_three`: 2.0 (same)
- `close_planning`: 3.0 (better)
- `music_ducks`: 1.0 (worse)
- `room_reverb`: 1.0 (worse)

New defaults written to `settings.json`:

- `plan_detail`: `standard` to `close`

Anthony's notes:

- `all_together`: this one has the most potential, but the foley is way too loud. and echo is bad dont do echo

Round 2 outcome: only closer planning with exact-moment effects beat the current
settings (3 vs 2), so `plan_detail` is now `close`. Best of three tied (2). Room echo
and music dipping both scored 1 and stay off; Anthony said plainly "don't do echo".
On the all-together version he wrote that it "has the most potential, but the foley is
way too loud". Effects level has not been changed yet because it was not a rated
variant: the next round should test quieter effects (`sfx_lufs_no_voice`, `spot_gain_db`).
The sound with the current defaults is
`/mnt/project-files/prompt2_tuning/new_defaults/mitey_vacuum_tuned_round2.mp4`.
Best score so far is 3 out of 5. Phase 3 has not been started.

## Round 3 (2026-11-25): effects level, waiting for Anthony's ratings

Anthony: "the foley is too loud in general". Default effects level lowered 4 dB
(`sfx_lufs_no_voice` -18 to -22, `sfx_lufs` -21 to -25). A free comparison (mix only,
all sound cached) is on the Sound Tuning Room page, run `20261125-025310`: current
(-22), -26, -30, and the old -18 for reference. After he rates: copy the page's rows
for that run into `ratings.csv`, `python tune.py --apply`, redeploy. Rounds 1 and 2 are
in `variants_round*.json` / `ratings_round*.csv`.

## Tuning round, 2026-11-25

From Anthony's ratings in `ratings.csv` (current settings scored 1.0 out of 5):

- `effects_as_before`: 1.0 (same)
- `effects_much_quieter`: 3.0 (better)
- `effects_quieter_still`: 2.0 (better, but another change to the same setting scored higher)

New defaults written to `settings.json`:

- `sfx_lufs_no_voice`: `-22.0` to `-30`

Round 3 outcome: quieter was better every step (-18: 1, -22: 1, -26: 2, -30: 3), and
Anthony added "better but still loud". `sfx_lufs_no_voice` is now -30 (`sfx_lufs` -33
with voice, set by hand to match).

## Round 4 (2026-11-25): effects still too loud, waiting for Anthony's ratings

Free mix-only comparison on the Sound Tuning Room page, run `20261125-030359`: current
(-30), -34, -38, no +4 dB boost for sounds planned as "loud" (`loudness_db`), and
moment effects 6 dB softer (`spot_gain_db` -8). After he rates: copy that run's rows
into `ratings.csv`, `python tune.py --apply`, redeploy. If quieter keeps winning, the
real problem may be the character of the effects (a constant harsh vacuum roar), not
their level; ask Anthony what exactly sounds too loud.
