#!/usr/bin/env bash
# Installs everything the project needs. Safe to run more than once.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo "ffmpeg is missing. Install it (apt install ffmpeg / brew install ffmpeg)."; exit 1; }
python3 -m pip install -q -r requirements.txt
python3 - <<'PY'
import modal, anthropic, scenedetect, pyloudnorm, soundfile, numpy, scipy, onnxruntime, dotenv
print("Python packages OK")
PY
# The reference test video lives in the project folder (videos are not in git).
REF=/mnt/project-files/mitey_vacuum_short_polish_full_silent.mp4
if [ -f "$REF" ] && [ ! -f test_videos/$(basename "$REF") ]; then
  cp "$REF" test_videos/ && echo "Copied the reference video into test_videos/."
fi
echo "Setup done."
