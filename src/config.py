from pathlib import Path
from typing import Literal
import json
from dotenv import load_dotenv
import os

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "out"
SLANTS_TRAIN = DATA_DIR / "slants_train.csv"
MODEL_DIR = ROOT / "models"

load_dotenv(ROOT / ".env")

YT_API_KEYS = json.loads(os.getenv("YT_API_KEYS"))