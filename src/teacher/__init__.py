from typing import Literal
from dotenv import load_dotenv

from src.config import ROOT

load_dotenv(ROOT / ".env", override=True)

LOCAL_SERVER = True

D_MODELS = Literal["typesafe/jev-1.14", "autotrust/JEV-27B", "gpt-4o-mini", "Qwen/Qwen3-8B", "Qwen/Qwen3.8-Flash-Next"]
C_MODELS = Literal["gpt-4o-mini", "Qwen/Qwen3-8B", "Qwen/Qwen3.8-Flash-Next"]

DECISION_MODEL: D_MODELS = "Qwen/Qwen3.8-Flash-Next"
CONTEXT_MODEL: C_MODELS = "Qwen/Qwen3.8-Flash-Next"

MODEL_EXTRA_ARGS: dict[str, dict] = {
    "Qwen/Qwen3.8-Flash-Next": {
        "temperature": 0.7,
        "top_p": 0.80,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    },
    "Qwen/Qwen3-8B": {},
    "gpt-4o-mini": {},
}

def get_model_name(name: str) -> str:
    return name.split("/")[-1].lower()

