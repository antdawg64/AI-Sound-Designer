# AI Sound Designer: project brief

Owner: Anthony. Anthony is not a programmer and is vibe coding. Claude makes every
technical decision, writes all the code, runs it, tests it and fixes it. Explain only
what Anthony must do personally, in short plain steps. Ask Anthony only for things
Claude can't do: creating accounts, API keys, logins, payments, and approving spending.

If you lose track: read this file and PROGRESS.md, then keep going.

## Product

A web app where someone uploads a short video (30 to 60 seconds now, up to 10 minutes
later) and gets it back with sound effects synced to the action and original background
music, mixed under any voice. They can also download the music and the effects as
separate files.

## Stack (already chosen; licences checked Nov 24, 2026)

- Synced sound effects: HunyuanVideo-Foley (Tencent Hunyuan Community License). Paid use
  is allowed except in the EU, UK and South Korea.
- Fallback effects from text cues: Stable Audio Open 1.0 (Stability Community License:
  free under $1M yearly revenue, needs registration and a "Powered by Stability AI"
  notice).
- Music: ACE-Step 1.5 (MIT). Fallback: DiffRhythm (Apache 2.0).
- NEVER use MMAudio, ThinkSound, PrismAudio, YuE2 or MusicGen. Their weights are
  non-commercial.
- Shot detection: PySceneDetect or ffmpeg. Sound planning: the Claude API looking at key
  frames. Mixing: ffmpeg.
- GPU: Modal serverless (L40S or A100). Language: Python.

## Working rules

- Read each model's own README and LICENSE before using it. If a licence differs from
  the above, stop and tell Anthony.
- Keep secrets in a `.env` file (or cloud environment variables) and in Modal secrets.
  Never commit them to git. `.env` is in `.gitignore`.
- Keep PROGRESS.md updated: what works, what's next, and what is needed from Anthony.
- Before anything that would spend more than about $5, tell Anthony the estimate and
  wait for an OK.
- After each phase, run it end to end on every video in `test_videos/` and give Anthony
  the output file paths.
- Commit to git after each working step.

## Build phases (from Anthony's plan)

0. Project brief: this file, PROGRESS.md, tools, account checklist.
1. One-day prototype: `python make_sound.py path/to/video.mp4` outputs `final.mp4`,
   `music.wav`, `sfx.wav`.
2. Quality tuning harness (`tune.py`, config file, side-by-side rating page).
3. Simple web app on a public link (Modal web endpoint, country-based licence rules).
4. Reliability and cost control (queue, limits, admin page, daily spend cap).
5. Accounts and payments (magic link sign-in, Stripe in TEST mode only until Anthony
   says otherwise in writing).

Only start a phase when Anthony asks for it.

## Where things live

- Code: this repo (github.com/antdawg64/ai-sound-designer).
- Test videos: `test_videos/` (git-ignored; 5 videos, 30 to 60 s, some with talking,
  some without, some with fast cuts).
- Secrets the code reads (names only, never values): `MODAL_TOKEN_ID`,
  `MODAL_TOKEN_SECRET`, `HF_TOKEN` come from the cloud environment variables. They must
  be plain environment variables in the `ai-sound-designer` cloud environment, NOT
  "Network secrets": a network secret on a Modal host breaks Modal (see PROGRESS.md).
  `ANTHROPIC_API_KEY` is NOT in the cloud environment (the project settings refuse
  Anthropic secrets): it lives in a Modal secret named `anthropic-secret`, so every Claude API
  call runs inside a Modal function with `secrets=[modal.Secret.from_name("anthropic-secret")]`.
  See `.env.example`.
- Setup: `bash scripts/setup.sh` installs the Python packages.
- Run: `python make_sound.py video.mp4` (or `--all` for everything in `test_videos/`).
  Results go to `outputs/<video name>/`. `--dry-run` uses test tones and no Modal.
  `--deploy` installs `modal_app.py` on Modal. Settings live in `settings.json`.
- Tune: `python tune.py` runs the variants in `variants.json` and builds the rating page in
  `tuning/`; `--serve` opens it with saving to `ratings.csv`; `--apply` makes the winners
  the new defaults. `python make_sound.py video.mp4 --versions` makes three takes.
- Editor: `web/editor.html` + `soundlib/editor.py` + `soundlib/editor_api.py` (timeline
  to fine-tune a finished job). What people change is kept in the Modal volume
  `ai-sound-designer-tuning-data`; `python analyze_edits.py` turns it into suggested
  defaults (`--apply` writes them to `settings.json`).
