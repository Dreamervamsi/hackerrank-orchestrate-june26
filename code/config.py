"""Paths and runtime configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / "code" / ".cache"

load_dotenv(REPO_ROOT / ".env")

CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
OUTPUT_CSV = DATASET_DIR / "output.csv"

IMAGES_DIR = DATASET_DIR / "images"

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")
USE_LOCAL = os.getenv("USE_LOCAL", "0").strip().lower() in {"1", "true", "yes"}
MAX_JSON_RETRIES = int(os.getenv("MAX_JSON_RETRIES", "2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
