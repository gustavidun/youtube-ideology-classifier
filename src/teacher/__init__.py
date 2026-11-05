from typing import Literal
from dotenv import load_dotenv

from src.config import ROOT

load_dotenv(ROOT / ".env", override=True)

LOCAL_SERVER = True

D_MODELS = Literal["typesafe/jev-1.14", "autotrust/JEV-27B"]
C_MODELS = Literal["gpt-4o-mini", "Qwen/Qwen3-8B", "Qwen/Qwen3.8-Flash-Next"]

DECISION_MODEL: D_MODELS = "typesafe/jev-1.14"
CONTEXT_MODEL: C_MODELS = "Qwen/Qwen3.8-Flash-Next"

MODEL_EXTRA_ARGS: dict[str, dict] = {
    "Qwen/Qwen3.8-Flash-Next": {
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        }
    },
    "Qwen/Qwen3-8B": {},
    "gpt-4o-mini": {},
}

def get_model_name(name: str) -> str:
    return name.split("/")[-1].lower()

