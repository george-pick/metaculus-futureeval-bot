"""Configuration for the Metaculus FutureEval bot.

Secrets are loaded from environment / .env files. NEVER hardcode keys here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BOT_DIR = Path(__file__).resolve().parent.parent

# Load shared project .env first (ANTHROPIC_API_KEY etc.), then local overrides.
_SHARED_ENV = Path("/workspace-vast/nickj/projects/.env")
if _SHARED_ENV.exists():
    load_dotenv(_SHARED_ENV)
load_dotenv(BOT_DIR / ".env", override=True)

# ---------------------------------------------------------------- Metaculus
METACULUS_API_BASE = "https://www.metaculus.com/api"
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")  # None => dry-run only

# Summer 2026 FutureEval tournament. The API accepts numeric IDs or slugs.
# https://www.metaculus.com/tournament/summer-futureeval-2026/  (ID 33022)
TOURNAMENT_ID: int | str = os.getenv("TOURNAMENT_ID", "summer-futureeval-2026")
MINIBENCH_ID = "minibench"
BOT_TESTING_AREA_ID = "bot-testing-area"

# ---------------------------------------------------------------- Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8000"))
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "high")  # low|medium|high|max
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "10"))

# ---------------------------------------------------------------- Ensemble
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", "5"))
# Fraction trimmed from EACH tail of the sample set before averaging
# (0.2 with N=5 drops the single highest and lowest sample).
TRIM_FRACTION = float(os.getenv("TRIM_FRACTION", "0.2"))

# Calibration guardrails: never submit binary probabilities more extreme than
# this (overconfidence is the classic bot failure mode under log scoring).
BINARY_FLOOR = float(os.getenv("BINARY_FLOOR", "0.015"))
BINARY_CEIL = 1.0 - BINARY_FLOOR
MC_FLOOR = float(os.getenv("MC_FLOOR", "0.005"))  # per-option minimum

# ---------------------------------------------------------------- Research
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
RESEARCH_CACHE_TTL_HOURS = float(os.getenv("RESEARCH_CACHE_TTL_HOURS", "20"))
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "12"))

# ---------------------------------------------------------------- Storage
RUNS_DIR = BOT_DIR / "runs"
CACHE_DIR = RUNS_DIR / "cache" / "research"
LOG_DIR = RUNS_DIR / "logs"
for _d in (CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
