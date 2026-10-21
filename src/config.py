from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "out"
SLANTS_TRAIN = DATA_DIR / "slants_train.csv"
MODEL_DIR = ROOT / "models"