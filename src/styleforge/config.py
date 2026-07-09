import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("STYLEFORGE_DATA", str(ROOT / "data")))
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = DATA_DIR / "out"

FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")

# Mock mode: full pipeline runs offline with canned model outputs (container smoke tests).
MOCK = os.getenv("STYLEFORGE_MOCK", "0") == "1"

# VERIFIED 2026-07-09: Fireworks serverless does NOT serve Gemma (7-model catalog;
# only Kimi K2.5/K2.6 have vision). Defaults below are the working serverless models;
# Gemma's placement (tuned in-container vs MI300X endpoint) is a pending team decision
# and swaps in via these env vars.
PERCEPTION_MODEL = os.getenv("PERCEPTION_MODEL", "accounts/fireworks/models/kimi-k2p6")
STYLIZER_MODEL = os.getenv("STYLIZER_MODEL", "accounts/fireworks/models/kimi-k2p6")
# kimi-k2p5 returns 500s as of 2026-07-09 — k2p6 verified working as judge.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "accounts/fireworks/models/kimi-k2p6")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "12"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "512"))

# Style specs are the single source of truth used by both the stylizer and the judge.
# Keys match the harness schema EXACTLY (Participant Guide: underscores, not hyphens).
STYLES: dict[str, str] = {
    "formal": (
        "Precise, professional, neutral register. Complete sentences, no slang, no jokes, "
        "no exclamation marks. Reads like a news wire or corporate video summary."
    ),
    "sarcastic": (
        "Dry, deadpan irony about what happens in the clip. Feigned admiration, "
        "understatement, eye-rolling tone. Never label the sarcasm; let it drip."
    ),
    "humorous_tech": (
        "Genuinely funny, with technology/programming/engineering references (bugs, deploys, "
        "APIs, AI, IT support, version control). The joke must still be about the clip's content."
    ),
    "humorous_non_tech": (
        "Genuinely funny for a general audience using everyday-life humor. "
        "Absolutely no technology, programming, or internet-culture references."
    ),
}

# Absolute last resort (harness NFR: a missing style scores zero, a generic caption may not).
FALLBACK_CAPTIONS: dict[str, str] = {
    "formal": "The clip documents a short everyday scene as it unfolds.",
    "sarcastic": "Ah yes, another clip. Truly the pinnacle of cinema.",
    "humorous_tech": "This clip loaded faster than my CI pipeline, and it shows.",
    "humorous_non_tech": "Somewhere, someone filmed this and thought: yes, the world needs it.",
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, OUT_DIR, CACHE_DIR / "judge"):
        d.mkdir(parents=True, exist_ok=True)
