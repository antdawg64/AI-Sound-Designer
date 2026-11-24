# Progress

_Last updated: 2026-11-24_

## What works

- Phase 0 done: CLAUDE.md (brief and rules), this file, `requirements.txt`,
  `scripts/setup.sh`, `.env.example`, `.gitignore`.
- Tools checked in the cloud build machine: Python 3.13, ffmpeg 6.1 (has `loudnorm`,
  `sidechaincompress`, `acrossfade`, `ebur128`). These Python packages install cleanly:
  modal, anthropic, scenedetect, opencv, pyloudnorm, soundfile, numpy, python-dotenv.

## What's next

- Phase 1: one-day prototype (`make_sound.py`). Waiting on the items below.

## Needed from Anthony

- [ ] Modal account, with a monthly spending limit set in its settings.
- [ ] Modal API token (token ID and token secret).
- [ ] Hugging Face account and an access token (read access is enough).
- [ ] Claude API key from the Anthropic Console, with a little credit on it.
- [ ] 5 test videos, 30 to 60 seconds each: some with talking, some without, some with
      fast cuts.
- [ ] Add the keys as environment variables in the project's cloud environment, using
      the names in `.env.example`.
- Later (Phase 1): accept the model licence pages on Hugging Face when asked.
- Later (Phase 5): Stripe account.

## Cost log

| Date | What | GPU seconds | Cost |
|------|------|-------------|------|
| 2026-11-24 | Phase 0 setup | 0 | $0 |
