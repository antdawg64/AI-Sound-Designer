# Progress

_Last updated: 2026-11-25_


## Real recordings library (2026-11-26, round 16 rated: on by default in settings, not live yet)

Branch `claude/project-thread-aw3m54` (built on the layering branch at 7c8e047). Anthony
asked for real recordings for common sounds ("yes on 6 too"). Round 16 rated 2026-11-26:
current 2, real recordings that sound right 3 (winner, applied: `sfx_library` on,
`library_judge` claude_sounds_alike, `library_min_score` 0.2), key moments only 2.
**Not deployed to the live apps**: until `ai-sound-designer-library` is deployed the
live apps cannot reach the library and simply generate every sound as before.

- **Licences** (read 2026-11-25, quotes in
  `/mnt/project-files/licence_research/real_recordings_library_licences.md`): only CC0
  recordings, because customers download `sfx.wav` and see each sound as an editor
  clip, so the sounds themselves are handed on. Source: FSD50K
  (zenodo.org/records/4060432), CC0 clips only, no voices, no music. The collection is
  CC-BY: **credit FSD50K on the site before this goes live.** Rejected: Sonniss,
  Pixabay, Freesound's API. BigSoundBank is CC0 and the best quality, but its owner asks
  automated tools not to copy the audio; needs an email from Anthony first.
- **The library**: Modal volume `ai-sound-designer-library`: 12,116 recordings, 29 hours.
  `catalog.jsonl` has one line per file with source, Freesound id, title, uploader,
  licence, link, labels and measurements; `embeddings.npy` is the search index (CLAP).
  Of 51,197 clips, 31,324 were not CC0, 6,585 were voice or music, 767 had titles
  suggesting borrowed material, 405 were silent, too short or distorted.
  Rebuild: `python scripts/modal_cmd.py run modal_library.py` (about 75 minutes, $1.50).
- **How a sound is chosen**: `soundlib/library.py`. For every sound made on its own (key
  moment, whoosh, detail, room background, and steady sources with
  `sfx_sources: separate`), `Cloud.effects` asks the library first. The search
  (`modal_library.py`, class `Search`) finds the closest recordings by sound, Claude
  reads their titles and labels and picks one or says "generate" (about $0.007 per
  sound). The recording is cut and placed like a generated clip (a moment starts on the
  action, a whoosh peaks on it, a background is one even stretch), so snapping,
  clean-up, mixing and editor clips are unchanged. Sounds that must follow the picture
  beat by beat (footsteps) stay generated.
- `library_judge`: `claude` (only the same thing making the same sound; on the vacuum
  video nothing qualified) or `claude_sounds_alike` (foley-artist judgement; used a
  body-fall thud for the nozzle landing, a cloth rip for the tuft snap, and a recorded
  room tone).
- Each job's `report.json` lists `library_recordings` (id, title, uploader, licence,
  link) and every sound's `library` note (used or why not). Editor clips from the
  library carry a `recording` field (not shown in the editor page yet).
- Trial copy used for all tests: `ai-sound-designer-dev-library`
  (`SOUND_APP=ai-sound-designer-dev`). The live name will be
  `ai-sound-designer-library`; deploy it with `modal deploy modal_library.py` before
  switching `sfx_library` on for the web app.
- **Round 16** (`variants_library.json`, run `20261126-025902`): Current vs real
  recordings that sound right vs the same for key moments only. Rating page (its own
  page, not the Sound Tuning Room): https://claude.ai/artifact/MhXcXzrDbvwy38CcLTHNSc .
  Full-size files: `/mnt/project-files/real_recordings/`. After Anthony rates: copy the
  page's `ratings` rows into `ratings.csv`, then
  `python tune.py --variants variants_library.json --apply`.
- Known limits: the vacuum video has few everyday sounds, so it is a weak test; a video
  with doors, footsteps or objects being put down would show more. Kenney's CC0 packs
  are not in (no direct download link found).

| 2026-11-25/26 | Library build (download, sort, index) | 2125 GPU + CPU | about $1.50 |
| 2026-11-26 | Round 16: baseline and library versions of the vacuum video | 226 | $0.51 |

## Layered effects, pro mix, Stable Audio, edit insights (2026-11-25, round 14 waiting for ratings)

Branch `claude/project-thread-1d0ddk` (contains the editor branch up to 7eded98). All new
sound behaviour is behind settings whose defaults are today's sound, so nothing changes
until Anthony rates round 14 and the winners are applied. **Not deployed to the live
apps yet** (the live deploy was held back for Anthony's OK).

- **Layered effects** (`sfx_layering: layered`): the planner now gives each key moment
  up to two supporting layers (a lead-in "whoosh", a "detail" just after) and one
  `ambience` line for the place. Each layer is generated on its own and is its own clip
  in the editor (key-moment lane, labelled "Whoosh:" / "Detail:"); the ambience is one
  quiet clip in its own lane, made from a held frame so nothing happens in it, looped
  with long crossfades. Levels: `moment_layer_db`, `ambience_level_db`. Extra cost on
  the vacuum video: $0.03.
- **Pro mix** (`mix_style: pro`): rumble filters per layer, music and effects both drop
  while someone talks (music further between 1 and 4 kHz), a guarantee that talking
  stays `dialogue_min_gap_db` (10 dB) above everything else, then ONE steady gain to
  -14 LUFS and a 4x-oversampled true-peak limiter (no loudnorm, so no pumping). The
  `music.wav` / `sfx.wav` downloads are then at their level in the finished video. A
  new automatic check "talking stays clear" runs when there is talking. The vacuum
  video has no talking, so that part is proven only by tests (`tests/test_editor.py`)
  with a stand-in voice. **Needs a test video with talking** to judge by ear.
- **Stable Audio** (`sfx_engine: stable_audio`): Stable Audio Open 1.0 makes every
  planned sound from its description; each is lined up with its planned time by
  listening for where it starts. Own Modal app (`modal_stable.py`), needs the Modal
  secret `huggingface-secret` (key `HF_TOKEN`; Anthony made it and accepted the model
  terms on 2026-11-25). Licence read 2026-11-25 (LICENSE.md in the model repo): matches
  CLAUDE.md. **Before paid use with it:** register at stability.ai/community-license and
  show "Powered by Stability AI" on the site. Known weakness: it does not watch the
  video, and a continuing sound (the vacuum) is made anew for every shot. Cost on the
  vacuum video: about $0.11 for 30 sounds on first run (includes loading the model).
- **Edit insights**: the analysis moved to `soundlib/insights.py`. The web app gets a
  Modal function `export_edits` (hands every record over, so `python analyze_edits.py`
  works from a cloud session with no extra allowed host) and an owner-only page
  `/insights` with the findings in plain language and downloads (zip of raw records,
  two spreadsheets). Owner = addresses whose SHA-256 is in `web_admin_email_sha256`.
  The report now covers layers and ambience, and shows "early signals" before the
  10-video threshold (never applied automatically). Goes live with the next web deploy.
- Trial copies on Modal, used for the test runs so the live apps were untouched:
  `ai-sound-designer-dev` and `ai-sound-designer-dev-stable` (set `SOUND_APP=ai-sound-designer-dev`).
  They cost nothing while idle and can be deleted any time.
- **Round 14** on the Sound Tuning Room (run `20261125-r14-layers`): Current, Layered,
  Layered with layers further back, Layered with pro mix, Stable Audio, Stable Audio
  layered. Files: `variants_layers.json`, `ratings_round14_layers.csv`. Copies of four finished videos:
  `/mnt/project-files/round14/`. Next: copy the page's `ratings` rows for that run into
  `ratings.csv`, `python tune.py --apply`, then deploy (tell the editor thread first).

| 2026-11-25 | Round 14: fresh baseline, layers, pro mix, Stable Audio | 433 | $0.44 |

## Needed from Anthony

- Rate round 14 on the Sound Tuning Room.
- OK to deploy to the live site (insights page; new sound only after the ratings).
- A short test video with someone talking, to hear the dialogue handling.
- If Stable Audio wins: register at stability.ai/community-license before charging.

## The site is private (2026-11-25)

One shared password in front of every page and API call (`soundlib/gate.py`, sign-in page
`web/login.html`). The password is in the Modal secret `site-password` (key
`SITE_PASSWORD`); change it with `modal secret create site-password SITE_PASSWORD=... --force`
and redeploy. A Lovable sign-in page is being built in another thread; when it is
published, put its address in `web_login_url` (settings.json) and redeploy. Tests
through the live site must sign in first (POST `password` to `/login`, keep the cookie).

## Sound editor (built and live 2026-11-25)

After the sound is made, the results page has a **Fine-tune the sound** button that
opens a timeline editor (`/edit/<job id>`, desktop first).

- Every effect is its own clip: drag to move, click for volume / mute / remove / listen,
  arrow keys to nudge, undo and redo, "Start over". One volume slider and mute for all
  effects and one for the music. Playback is instant in the browser (no server, no cost).
- Since the evening of 2026-11-25 the timeline is made of **pieces**: each background
  clip starts out cut into one piece per shot (labelled with that shot's sounds), and
  any piece can be **split** at the red line (S) or **duplicated** (D). The sounds
  inside one piece are still one recording from the effects model and cannot be pulled
  apart; truly separate sounds would mean generating each one on its own (more GPU
  time, and the sound would need re-tuning). Anthony asked for this after his first try.
- **Make my video** remixes on the server (about 15 s, no GPU) and offers the edited
  video, effects and music as downloads. The original files are kept.
- How it works: `mixing.effects_layers` builds the effects track as one layer per
  effect; `soundlib/editor.py` saves them in `out/editor/` with `out/editor.json`,
  cleans edits, renders, and writes tuning records. `soundlib/editor_api.py` is the
  API; `web/editor.html` is the page. Jobs made before this have no editor files.
- **Tuning data**: each editing session is saved as one small JSON file in the Modal
  volume `ai-sound-designer-tuning-data` (`/edits/<job>/<session>.json`): every clip
  with what was done to it, the volume changes, each action in order, and the settings
  the job was made with. No video, no audio, nothing about the person. It is never
  deleted by the daily clean-up.
- **`python analyze_edits.py`** downloads the records, writes `tuning_data/clips.csv`,
  `jobs.csv` and `summary.md`, and suggests new defaults (music level, key-moment
  level, per-shot effect levels) once 10 edited videos agree. `--apply` writes them to
  `settings.json`. Test-tone jobs are left out unless `--include-tests`.
- Try it free on a laptop: `python make_sound.py v.mp4 --dry-run --out outputs/demo`,
  then `python scripts/editor_dev_server.py outputs/demo v.mp4` and open
  http://localhost:8766/edit/demo. Checks: `python -m pytest tests/ -q`.
- **Live since 2026-11-25** at https://anthony-56415--ai-sound-designer.us-east.modal.run.
  Tested there with test tones and with one real run of the vacuum video (8 clips;
  drag, volume, save, remix and download all worked; tuning record written).
- From a cloud session the tuning records can be listed but not downloaded (Modal's
  storage host is not on the allowlist), so run `analyze_edits.py` on a computer, or
  add the host the error names to the cloud environment first.
- The front page (`web/index.html`) was redesigned the same day as a product-style
  landing page; before/after pictures are in `/mnt/project-files/editor/`.
- Later steps Anthony agreed to: "regenerate this one sound" and "add a sound by
  typing a description".
## Phase 3: web page (2026-11-25), live

- Public page: https://anthony-56415--ai-sound-designer.us-east.modal.run
- `web_app.py` (Modal app `ai-sound-designer-web`): the page (`web/index.html`), upload and
  progress API, `run_job` (runs `make_sound.make_sound()` on a 4-CPU machine, calling the
  GPU app by name), and a daily `cleanup` that deletes jobs after `web_keep_days` (7).
  Files: Modal volume `ai-sound-designer-jobs` (`/jobs/<id>/`); status: Modal Dict of the
  same name. Each job's `report.json` keeps the settings used, for later tuning.
- People choose: effects and music, effects only, or music only (`--only` on the command
  line). The part left out costs nothing and comes back as a silent file.
- Licence rule: no uploads from the EU, UK or South Korea, checked by the country picked
  on the page AND by the visitor's internet address (DB-IP Lite, CC BY 4.0, credited in the
  footer). List: `web_blocked_countries` in settings.json.
- Cost guard until Phase 4: at most `web_daily_job_limit` (10) videos a day for everyone,
  about $3 at most. Videos 3 to 75 s, up to 400 MB.
- Re-tuning: settings.json drives the command line, tune.py and the web page alike. After
  a tuning round, `python make_sound.py --deploy` deploys both Modal apps.
- Free test of the live page: add `?test=tones` to the address (test tones, no GPU).
- Tested live 2026-11-25: vacuum video, effects and music, 243 s, all checks pass
  (-13.9 LUFS, -1.0 dBTP, 28.000 s). Copies in `/mnt/project-files/prompt3_web/`. Also
  tested: GB upload refused (403), effects-only with test tones.
- The cloud session needs the exact host `anthony-56415--ai-sound-designer.us-east.modal.run`
  allowed (added) to test the page; reading the jobs volume from here is blocked
  (`clobber.modal-storage.com`), so check results through the page's own downloads.


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
- [ ] Try the web page with your own videos and say what to change.
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
| 2026-11-25 | Phase 3 live web test (vacuum, effects and music) | about 150 | about $0.30 est. |
| 2026-11-24 | Real run 2 after level fixes | 147 | $0.14 est. (GPU $0.08 + Claude $0.07) |
| 2026-11-24 | Two Claude plan tests | 0 | $0.13 |
| 2026-11-24 | Real run 3 after Anthony's feedback | 161 | $0.15 est. |

## Tuning round, 2026-11-25

From Anthony's ratings in `ratings_round14.csv` (current settings scored 2.0 out of 5):

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

From Anthony's ratings in `ratings_round14.csv` (current settings scored 2.0 out of 5):

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

## Tuning round, 2026-11-25

From Anthony's ratings in `ratings.csv` (current settings scored 1.0 out of 5):

- `effects_minus_4`: 2.0 (better, but another change to the same setting scored higher)
- `effects_minus_8`: 3.0 (better)
- `no_loud_boost`: 3.0 (better)
- `softer_moments`: 4.0 (better)

New defaults written to `settings.json`:

- `sfx_lufs_no_voice`: `-30` to `-38`
- `loudness_db`: `{"quiet": -6.0, "medium": 0.0, "loud": 4.0}` to `{"quiet": -6, "medium": 0, "loud": 0}`
- `spot_gain_db`: `-2.0` to `-8`

Round 4 outcome: softer moment effects scored 4 (best so far), no loud boost 3, effects
at -38 scored 3; all three applied (`spot_gain_db` -8, `loudness_db` loud 0,
`sfx_lufs_no_voice` -38, and `sfx_lufs` -41 with voice set by hand to match).

## Round 5 (2026-11-25): music quality, waiting for Anthony's ratings

Anthony asked how to make the music better quality. Added the ACE-Step 1.5 XL models
(4B, MIT, not gated; licence read on each Hugging Face model card 2026-11-25):
`acestep-v15-xl-turbo` (8 steps) and `acestep-v15-xl-sft` (50 steps, guidance 7).
Both are downloaded to the Modal volume (20 GB each). The `Music` class on Modal now
takes the model as a parameter, so each model runs in its own container. Page run
`20261125-031413`: current (2B turbo), XL turbo, XL sft at 50 steps, and a check tile
with effects at -34. Not tried yet: the model's own LM planner ("thinking" mode).

| 2026-11-25 | Rounds 3 and 4 (mix only, all cached) | 0 | $0 |
| 2026-11-25 | XL music model downloads (CPU) and round 5 run | 62 | $0.03 est. plus cents of CPU |

## Tuning round, 2026-11-25

From Anthony's ratings in `ratings.csv` (current settings scored 1.0 out of 5):

- `effects_up_a_step`: 3.0 (better)
- `music_bigger_model`: 3.0 (better)
- `music_bigger_model_slow`: 4.0 (better (all its settings applied))

New defaults written to `settings.json`:

- `sfx_lufs_no_voice`: `-38` to `-34`
- `music_model`: `acestep-v15-turbo` to `acestep-v15-xl-sft`
- `music_steps`: `8` to `50`

Anthony's notes:

- `music_bigger_model_slow`: now music is too loud though

Round 5 outcome: XL music in slow mode scored 4 (XL turbo 3, old model 1), so
`music_model` is `acestep-v15-xl-sft` with `music_steps` 50. Effects one step louder
scored 3 against 1, so `sfx_lufs_no_voice` is -34. Anthony's note on the XL slow tile:
"now music is too loud though".

## Round 6 (2026-11-25): music level, waiting for Anthony's ratings

Music lowered 3 dB on that note (`music_lufs_no_voice` -26 to -29). Free page run
`20261125-041548`: current (-29), -32, -34 (level with effects), and -26 for
reference. The with-voice levels (`sfx_lufs` -37, `music_lufs` -30) were moved by hand
to keep the same balance and have never been heard on a video with speech.

Worth knowing: the final mix is always brought to -14 LUFS, so with no voice only the
gap between music and effects matters, not the numbers themselves. Round by round the
preferred gap has moved from effects 8 dB above music to effects about 5 dB below it.
`tune.py --apply` now keeps winning single changes alongside a winning combination,
unless they touch the same setting.

Round 6 outcome: Anthony scored only one tile, "music level with the effects", a 4.
With no score on the current tile `--apply` had nothing to compare with, so it was
applied by hand: `music_lufs_no_voice` -34, equal to the effects.

## Where tuning stands after six rounds (2026-11-25)

Defaults now: closer planning (up to 6 frames per shot) with exact-moment effects
6 dB softer, one main sound per shot, effects guidance 6.5, effects made over clips up
to 8 s, no boost for "loud" sounds, music from ACE-Step 1.5 XL sft at 50 steps, music
swells left alone, music and effects at equal level (-34 / -34), no echo, no music
ducking under effects, one effects candidate. Best score so far is 4 out of 5.

The with-voice levels were reset to `sfx_lufs` -27 and `music_lufs` -27 (voice -16):
equal balance like the no-voice case, 11 dB under the voice. They are a guess. Nothing
has been heard on a video with speech; test this as soon as such a video exists.

Finished video with these defaults:
`/mnt/project-files/prompt2_tuning/new_defaults/mitey_vacuum_tuned_round6.mp4` (plus
`music_round6.wav`, `sfx_round6.wav`). All of this rests on ONE video with no speech.
Modal spend for all Phase 2 work: about $1.00. Phase 3 has not been started.

## Round 7 (2026-11-25): music scored to the picture, waiting for Anthony's ratings

Anthony asked whether the music was timed to the picture. It was not: one brief, one
track, only the tempo nudged so beats fall near cuts. He wants film-style scoring.
Built `music_scoring` (settings.json):

- A second Claude call (`plan_cues` on Modal, `planning.ask_cues`) writes a cue sheet:
  a key, 2 to 5 sections on shot boundaries with intensity 1 to 5 and a description,
  and up to six hits (small or big). It is cached like the plan and does not change the
  effects plan. About 9 cents per video.
- `shaped`: the usual single track, with its level following the cue sheet
  (`mixing.shape_to_cues`): section intensity, a dip before each hit and an accent on it.
- `sections`: each section composed separately at the same tempo and key
  (`keyscale`), started on a bar line past any weak intro, joined on the cuts
  (`mixing.join_sections`), then shaped the same way.

Page run `20261125-043518`: current, shaped, sections. Not built: separate stingers or
risers at hits, and tempo changes between sections.

| 2026-11-25 | Round 7: cue sheet and sectioned music | 38 | $0.11 est. |

## Tuning round, 2026-11-25

From Anthony's ratings in `ratings_round14.csv` (current settings scored 2.0 out of 5):

- `music_sections`: 4.0 (better)
- `music_shaped`: 3.0 (better, but another change to the same setting scored higher)

New defaults written to `settings.json`:

- `music_scoring`: `single` to `sections`

Anthony's notes:

- `music_sections`: needs smoother transitions but right thought

## Round 8 (2026-11-25): smoother joins between sections, waiting for Anthony's ratings

`sections` is now the default. Anthony's note: "needs smoother transitions but right
thought". Two approaches built:

- `section_crossfade_seconds` (settings.json): 0 = sharp switch on the cut (current);
  a number = neighbouring sections blend into each other over that many seconds around
  the cut (`mixing.join_sections`).
- `music_scoring: repaint`: one full track is composed, then ACE-Step rewrites each
  section inside it (`Music.generate_scored` on Modal, task type repaint), so it stays
  one continuous piece and the model makes the joins itself.

Page run `20261125-045101`: current, blend 1 s, blend 2 s, one piece.

| 2026-11-25 | Round 8: blended joins and one-piece scoring | 43 | $0.02 est. |

Round 8 result (`ratings_round8.csv`): blend 1 s scored 1, blend 2 s scored 1, one
piece scored 3. The current tile was not rated; sections with a sharp switch scored 4
in round 7, so nothing was applied and the defaults are unchanged
(`music_scoring: sections`, `section_crossfade_seconds: 0`). Crossfading two separately
composed sections is clearly worse than a sharp switch. Smoother joins are still open.

## Round 9 (2026-11-25): even smoother music, waiting for Anthony's ratings

Anthony after round 8: "Better, but not smooth enough still" (read as: the one-piece
version is the right direction). Three new settings:

- `section_instruments: shared`: every section keeps the main brief's instruments and
  only its energy wording changes (`planning.section_caption`), instead of each section
  getting its own instrument list from the cue sheet.
- `repaint_keep_seconds`: with repaint, this long on each side of a section change is
  left as first composed, so the join itself is never rewritten.
- `cue_ramp_seconds`: how long the level takes to glide between sections (was fixed 0.8).

Page run `20261125-050613`: current (sections, own instruments), same band, one piece
same band, one piece joins kept, one piece gentle (3 s glide).

| 2026-11-25 | Round 9: shared instruments, kept joins, slow glide | 75 | $0.04 est. |

Round 9 result (`ratings_round9.csv`): same band 1, one piece same band 3 ("audio
leveling is bad, its not smooth. but i like the same instruments"), joins kept 1,
gentle 1. Current tile not rated, so nothing applied. Leaving the joins unrewritten made
it worse. The roughness he hears is the volume, not the composition.

## Round 10 (2026-11-25): volume handling, waiting for Anthony's ratings

All tiles are repaint + shared instruments; only the level shaping differs (all free,
cached): as round 9, no dips or accents at hits, no dips plus a small intensity range
with a 2 s glide, no added shaping at all, and no shaping plus `music_even_out` 0.7.
Page run `20261125-052304`.

Bug fixed on the way: with effects about 20 dB below full scale the master limiter works
hard, and the AAC encode then overshot to +0.9 dBTP on one tile. `mixing.mix` now checks
the encoded file and limits harder if it is over the peak target. Worth testing later:
that heavy limiting on effect spikes may itself make the music pump.

## Tuning round, 2026-11-25

From Anthony's ratings in `ratings.csv` (current settings scored 1.0 out of 5):

- `as_round9`: 2.0 (better)
- `gentle_levels`: 3.0 (better)
- `hands_off`: 4.0 (better (all its settings applied))
- `no_dips`: 2.0 (better)

New defaults written to `settings.json`:

- `music_scoring`: `sections` to `repaint`
- `section_instruments`: `own` to `shared`
- `cue_hit_dip_db`: `{"small": -5.0, "big": -14.0}` to `{"small": 0.0, "big": 0.0}`
- `cue_hit_accent_db`: `{"small": 1.5, "big": 3.0}` to `{"small": 0.0, "big": 0.0}`
- `cue_intensity_db`: `[-10.0, -6.0, -3.0, -1.0, 0.0]` to `[0.0, 0.0, 0.0, 0.0, 0.0]`

Round 10 outcome: "volume left exactly as the music was played" scored 4 (current 1).
Defaults are now one continuous piece (`music_scoring: repaint`), shared instruments,
and no added level shaping (intensity all 0 dB, no dips or accents at hits). The cue
sheet still decides where sections change and how each is written. "Evened out" was not
rated. Finished video (the rated tile itself):
`/mnt/project-files/prompt2_tuning/new_defaults/mitey_vacuum_tuned_round10.mp4`.
A fresh run of `make_sound.py` with the new defaults passed all checks (180 GPU s plus
planning, $0.30; it does not share the tuning cache). Phase 2 Modal spend about $1.50.

| 2026-11-25 | Round 10 confirm run with new defaults | 180 | $0.30 |

## Round 11 (2026-11-25): sharp change at 0:04 and buried moment effects, waiting for ratings

Anthony: ":04 had a sharp transition" and "we missed the cable shooting out from his
arm". Measured: the 3.2 s opening section was about 10 dB quieter than the next one; the
cable ("grapple line firing", moment at 16.65 s) was planned and generated but mixed
8 dB down (`spot_gain_db`) under the vacuum roar. New settings:

- `section_min_seconds`: sections shorter than this fold into the next one.
- `section_snap: bar`: section changes move to the nearest bar line
  (`planning.settle_sections`).
- `spot_bed_dip_db`: the background effects drop briefly under each moment effect.

Page run `20261125-054755`: current, no short sections, on the bar, both, both evened,
moments louder (`spot_gain_db` 0), moments louder with the background making room.
Still unverified because Claude cannot hear: whether each spot clip really sounds like
its description. If a louder zip is still wrong, next step is a text-only effect for
each moment (Stable Audio Open, see licence notes in CLAUDE.md).

| 2026-11-25 | Round 11: section tidy-up, louder moments | 80 | $0.04 est. |

Round 11 result (`ratings_round11.csv`, current not rated): both 4, both evened 4, on
the bar 3, no short 2, moments louder 3, moments louder with room 2. Applied by hand:
`section_min_seconds: 5`, `section_snap: bar`. Louder moments did not help, so not
applied.

## Round 12 (2026-11-25): effects balance, waiting for Anthony's ratings

Anthony: "Some sound effects are a lot louder than others. It should only be louder when
it makes sense... I can't hear the vacuum... the vacuum is the main thing in this scene."
Cause: every generated clip was set to one average level, so sparse sharp sounds
(footsteps) peaked as high as anything, and inside a clip the vacuum could fade away.

- `sfx_leveling: by_shot` (`mixing.balance_shots`): each shot's loud parts are brought
  to a level set only by the plan's prominence for that shot (`shot_level_db`).
- `sfx_even_out`: lifts stretches where a steady sound drops away.
- `spot_candidates`: best-of-N (CLAP) for the exact-moment effects only.

Page run `20261125-063334`: current, balanced, balanced steady, balanced louder
(effects 3 dB over music), balanced with best-of-4 moments, balanced with every planned
sound in the prompts (`sfx_prompt_style: short`).

| 2026-11-25 | Round 12: best-of-4 moments, every-sound prompts | 197 | $0.11 est. |

Round 12 result (`ratings_round12.csv`; current and plain balanced not rated): balanced
with every planned sound 4, balanced steady 3, balanced louder 3, balanced best-of-4
moments 3. Applied by hand: `sfx_leveling: by_shot`, `sfx_prompt_style: short`.
Finished video (the rated tile): 
`/mnt/project-files/prompt2_tuning/new_defaults/mitey_vacuum_tuned_round12.mp4`.
Phase 2 Modal spend about $1.70. Phase 3 not started.

## Round 13 (2026-11-25): build from scratch, waiting for Anthony's ratings

Anthony asked for a rebuild from nothing with the current defaults, to compare with the
round 12 winner. The old cache was moved to `cache_before_rebuild/` (git-ignored, lost
when this container goes) and `python tune.py` run with no variants: new plan, new cue
sheet, new effects, new music. All checks passed. The new plan differs a little (the
cable moment is now "wire line snaps taut with a twang", and there is a new moment at
9.2 s); the cue sheet has three sections. Page run `20261125-065425`: "Built from
scratch" and "Round 12 version". Copy:
`/mnt/project-files/prompt2_tuning/new_defaults/mitey_vacuum_fresh_build.mp4`.

| 2026-11-25 | Round 13: full fresh build | 143 | $0.28 |

## Original audio is kept for music-only and effects-only runs (Nov 25, 2026)

- Asking for only music or only effects now keeps the whole sound the video came with
  (before, only talking was kept and the rest was dropped). It is in the finished video
  and shows in the editor as an "Original audio" track with its own volume and Mute.
- Asking for both still keeps only the talking, because the new effects and music
  replace the rest. That track ("Voice") now has volume and Mute too.
- If the kept sound has no talking, the new music or effects sit 6 dB under it; with
  talking they sit under the voice as before and the music dips under the talking.
- Tested free with test tones on the vacuum video (music only and effects only).

## Open front page, personal accounts (Nov 25, 2026)

- Anthony's change: the front page shows to everyone with a "Sign in" button top right.
  Sign-in is asked only when someone presses "Add sound" while signed out; nothing is
  generated before that. The chosen video waits in their browser and is sent
  automatically when they come back signed in.
- Accounts are on (settings `web_login_url`, `web_supabase_url`, `web_supabase_anon_key`);
  the shared password no longer opens the site. The sign-in page is the Lovable one;
  people who are not approved see its waiting-list screen.
- Job pages, the editor and every other API call still need a signed-in account.
- The Modal secret `site-password` must stay: it signs the session cookies.


## Tuning round, 2026-11-25

From Anthony's ratings in `ratings_round14_layers.csv` (current settings scored 2.0 out of 5):

- `layered`: 3.0 (better)
- `layered_pro`: 4.0 (better (all its settings applied))
- `layered_soft`: 3.0 (better)
- `stable`: 2.0 (same)
- `stable_layered`: 3.0 (better)

New defaults written to `settings.json`:

- `sfx_layering`: `single` to `layered`
- `mix_style`: `classic` to `pro`
## Google Lyria as a second music model (2026-11-25, thread "Try Google Lyria for music")

- **What works:** `music_provider: lyria` in `settings.json` makes the music with Google
  Lyria on Vertex AI instead of ACE-Step (default stays `acestep`, so the live site is
  unchanged). Code: `soundlib/lyria.py`, `lyria_app.py` (its own Modal app
  `ai-sound-designer-lyria`; `modal deploy lyria_app.py` does not touch the live apps).
  Lyria cannot rewrite a section in place, so the cue sheet's sections go into the brief
  with their times. `lyria_model`: `lyria-002` (Lyria 2, fully released, always 30 s),
  `lyria-3-pro-preview` (preview; keeps to the asked length and the timed sections),
  `lyria-3-clip-preview` (preview; 30 s).
- **Google Cloud:** project `ai-sound-designer-c20200` on Anthony's billing account;
  service account `lyria-music` (Vertex AI User only); key in Modal secret
  `google-cloud`. Sign in from a cloud session with `scripts/google_login.py`.
- **Licence:** Lyria is NOT on Google's list of services with the generative-AI
  copyright indemnity (list of 2026-07-20). Lyria 3 is a preview release. Every Lyria
  track carries an inaudible SynthID watermark. Anthony was told; lawyer to confirm.
- **Round 14 comparison** (`variants_lyria.json`, run `20261125-221443`): current music
  vs Lyria 3 Pro vs Lyria 2, same plan, effects and mix. Rating page (separate from the
  Sound Tuning Room, which the layering thread is using):
  https://claude.ai/artifact/C1LPE89e6bRzenGJqP7jXX . Full-size files in
  `/mnt/project-files/lyria_test/`. After Anthony rates: copy the page's `ratings` rows
  into `ratings.csv`, `python tune.py --variants variants_lyria.json --apply` only if he
  wants Lyria as the default, then `modal deploy` is with the editor thread.
- **Not done:** videos longer than 30 s with Lyria 2 blend several clips (untested by
  ear). Lyria 3 prices in settings are unconfirmed. Two Lyria requests were paid for and
  lost to a download error on the first run (about $0.14).

| 2026-11-25 | Lyria test: fresh baseline + Lyria 2 + Lyria 3 Pro, and trial clips | 146 | about $0.29 Modal + about $0.60 Google (est.) |

**Round 14 result (2026-11-25, Anthony's scores):** Lyria 3 Pro 4, Lyria 2 3, current
music 2 (`ratings_round14.csv`). Anthony chose on a decision card to make Lyria 3 Pro
the music on the live site. Defaults are now `music_provider: lyria`,
`lyria_model: lyria-3-pro-preview`; if Google fails, the run falls back to ACE-Step.
Going live needs the web app redeployed from a branch that contains this one (the
editor thread owns that deploy); `ai-sound-designer-lyria` is already deployed.

Added after the round (2026-11-25 22:35 UTC): `stable_layered_pro` (Stable Audio,
layered, pro mix) scored 3, against 4 for the same with the current effects model. So
the effects model stays HunyuanVideo-Foley (`sfx_engine: foley`). Stable Audio stays in
the code as an option; its known weakness is that a continuing sound is made anew for
every shot. An idea not yet tried: Stable Audio only where the licence blocks the
current model (EU, UK, South Korea), so those visitors are served instead of refused.

## Tuning round, 2026-11-26

From Anthony's ratings in `ratings_round16_library.csv` (current settings scored 2.0 out of 5):

- `library_alike`: 3.0 (better (all its settings applied))
- `library_alike_moments`: 2.0 (same)

New defaults written to `settings.json`:

- `sfx_library`: `off` to `on`
- `library_judge`: `claude` to `claude_sounds_alike`
- `library_min_score`: `0.25` to `0.2`
