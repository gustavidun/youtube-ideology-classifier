from openai import OpenAI
import os
from dotenv import load_dotenv
import httpx

from src.config import LOCAL_SERVER, ROOT

load_dotenv(ROOT / ".env", override=True)

if LOCAL_SERVER == False:
    context_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    decision_client = httpx.Client(
        base_url="https://openrouter.ai/api/alpha",
        headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
        timeout=30.0,
    )
    decision_endpoint = "/decisions"
else:
    context_client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    limits = httpx.Limits(
        max_connections=256,
        max_keepalive_connections=128 
    )

    decision_client = httpx.Client(
        base_url="http://localhost:8000/v1",
        headers={"Authorization": "Bearer EMPTY"},
        timeout=300.0,
        limits=limits
    )
    decision_endpoint = "/decide"