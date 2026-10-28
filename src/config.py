from pathlib import Path
from typing import Literal

def get_model_name(name : str):
    return name.split("/")[-1].lower()

LOCAL_SERVER = True

CONTEXT_MODEL : Literal["gpt-4o-mini", "Qwen/Qwen3-8B"] = "Qwen/Qwen3-8B"
DECISION_MODEL : Literal["typesafe/jev-1.14", "autotrust/JEV-27B"] = "autotrust/JEV-27B"

if LOCAL_SERVER:
    NUM_WORKERS_CONTEXT = 128
    NUM_WORKERS_DECISION = 64
else:
    NUM_WORKERS_CONTEXT = 40
    NUM_WORKERS_DECISION = 40

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "out"
SLANTS_TRAIN = DATA_DIR / "slants_train.csv"
MODEL_DIR = ROOT / "models"