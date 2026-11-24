"""Every tunable setting in one place. Phase 2 turns this into a config file."""

SAMPLE_RATE = 48000

# --- Shots ---------------------------------------------------------------
SHOT_THRESHOLD = 27.0        # PySceneDetect ContentDetector threshold
MIN_SHOT_SECONDS = 1.0       # shorter shots are merged into a neighbour

# --- Key frames sent to Claude --------------------------------------------
FRAMES_PER_SHOT = 3
MAX_FRAMES = 24
FRAME_LONG_SIDE = 768
PLAN_MODEL = "claude-opus-5-5"
CLAUDE_SECRET_NAME = "anthropic-secret"   # the Modal secret holding ANTHROPIC_API_KEY

# --- Effects (HunyuanVideo-Foley) -----------------------------------------
FOLEY_MAX_SEGMENT_SECONDS = 10.0   # long shots are cut into pieces this long
FOLEY_MIN_SEGMENT_SECONDS = 1.0
FOLEY_OVERLAP_SECONDS = 0.3        # extra video each side, used for the crossfade
FOLEY_CLIP_SHORT_SIDE = 512        # small copy of the shot sent to the GPU
FOLEY_CLIP_MAX_BYTES = 1_400_000   # keeps each upload inside one Modal message
FOLEY_STEPS = 50
FOLEY_GUIDANCE = 4.5
FOLEY_NEG_PROMPT = "noisy, harsh, music, speech, voice, singing"
FOLEY_MODEL_SIZE = "xxl"

# --- Music (ACE-Step 1.5) --------------------------------------------------
MUSIC_MODEL = "acestep-v15-turbo"
MUSIC_STEPS = 8
MUSIC_MIN_SECONDS = 10.0           # the model's shortest supported length
MUSIC_FADE_OUT_SECONDS = 1.5
MUSIC_BPM_SEARCH = 0.12            # search +/- 12% around Claude's tempo
MUSIC_BPM_RANGE = (60, 180)

# --- Mix -------------------------------------------------------------------
TARGET_LUFS = -14.0
TARGET_TRUE_PEAK_DB = -1.0
ENCODE_PEAK_MARGIN_DB = 0.8        # AAC encoding can raise peaks slightly
VOICE_LUFS = -16.0
SFX_LUFS = -21.0                   # effects sit under the voice
MUSIC_LUFS = -23.0
MUSIC_LUFS_NO_VOICE = -20.0        # a bit more music when nobody speaks
SFX_LUFS_NO_VOICE = -18.0
LOUDNESS_DB = {"quiet": -6.0, "medium": 0.0, "loud": 4.0}
DUCK = "threshold=0.02:ratio=6:attack=15:release=450:makeup=1"
AAC_BITRATE = "256k"

# --- Seeds -----------------------------------------------------------------
SEED = 1

# --- Cost (US dollars per second; Modal list prices, re-check on modal.com/pricing)
GPU_TYPE = "L40S"
GPU_DOLLARS_PER_SECOND = {"L40S": 0.000542, "A100": 0.000583, "A100-80GB": 0.000694}
# Claude Opus 5.5 list price per million tokens.
CLAUDE_DOLLARS_PER_MTOK = {"input": 4.00, "output": 20.00}

# --- Automatic checks ------------------------------------------------------
CHECK_DURATION_TOLERANCE = 0.1
CHECK_LUFS_TOLERANCE = 1.0
CHECK_MAX_TRUE_PEAK_DB = -0.1      # anything at or above 0 dBTP would be clipping
CHECK_SILENT_LUFS = -50.0
