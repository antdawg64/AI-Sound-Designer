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
echo "Setup done."
