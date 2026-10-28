import itertools
import os
import random
from concurrent.futures import ThreadPoolExecutor
import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
from pathlib import Path
from functools import partial

from typing import Literal

from src.config import OUT_DIR, ROOT
from src.teacher.context import VideoAnalysis
from src.teacher import DECISION_MODEL

LOCAL_SERVER = True
NUM_WORKERS_DECISION = 64 if LOCAL_SERVER else 40

CHUNK_SIZE = 1000
PAIRS_PER_VIDEO = 5

Q_ALIGNMENT = {
    "economic_alignment": {
        "type": "score",
        "instructions": "Evaluate ideological alignment of the video-pair on economic/fiscal policy. Focus on underlying worldview alignmentn even if the specific topics differ.",
        "criteria": [
            "Level 0: Direct Opposition. Diametrically opposed political objectives on fiscal, tax, and labor policy.",
            "Level 1: Moderate Opposition. Substantial conflict on economic policy priorities.",
            "Level 2: Neutral / Orthogonal. Economic policy is not relevant, or positions do not conflict.",
            "Level 3: Moderate Alignment. Substantial shared economic assumptions.",
            "Level 4: Identical Stance. Shared economic assumptions and identical desired outcomes.",
        ],
    },
    "galtan_alignment": {
        "type": "score",
        "instructions": "Evaluate ideological alignment of the video-pair on GAL-TAN (Green/Alternative/Libertarian vs Traditional/Authoritarian/Nationalist) cultural axes. Focus on underlying worldview alignment even if the specific topics differ.",
        "criteria": [
            "Level 0: Direct Opposition. Diametrically opposed cultural, social, and institutional values.",
            "Level 1: Moderate Opposition. Substantial social or cultural disagreement.",
            "Level 2: Neutral / Orthogonal. Cultural issues are not relevant, or positions do not conflict.",
            "Level 3: Moderate Alignment. Substantial shared cultural or social positioning.",
            "Level 4: Identical Stance. Identical social, cultural, and institutional values.",
        ],
    },
    "overall_ideological_alignment": {
        "type": "score",
        "instructions": "Evaluate the overall directional ideological alignment of the video-pair.",
        "criteria": [
            "Level 0: Direct Opposition. Diametrically opposed ideological viewpoints.",
            "Level 1: Moderate Opposition. Substantial conflict on ideological viewpoints.",
            "Level 2: Neutral / Orthogonal. No discernible ideological conflict or alignment.",
            "Level 3: Moderate Alignment. Substantial shared ideological viewpoints.",
            "Level 4: Identical Stance. Identical ideological viewpoints.",
        ],
    },
}

Q_RANKING = {
    "is_more_right_leaning_economic": {
        "type": "noul",
        "instructions": "Is video A more right-leaning than video B on an economic ideology scale?",
    },
    "is_more_right_leaning_cultural": {
        "type": "noul",
        "instructions": "Is video A more right-leaning than video B on a GAL-TAN (Green/Alternative/Libertarian vs Traditional/Authoritarian/Nationalist) ideology scale?",
    },
    "is_more_right_leaning_overall": {
        "type": "noul",
        "instructions": "Is video A more right-leaning than video B on an overall ideological scale?",
    },
}

Q_ABSOLUTE = {
    "galtan_score": {
        "type": "score",
        "instructions": "Evaluate the GAL-TAN (Green/Alternative/Libertarian vs Traditional/Authoritarian/Nationalist) ideology score of the video." ,
        "criteria": [
            "Level 0 Far-left.",
            "Level 1 Left.",
            "Level 2 Center.",
            "Level 3 Right.",
            "Level 4 Far-right.",
        ]
    },
    "economic_score": {
        "type": "score",
        "instructions": "Evaluate the economic ideology score of the video." ,
        "criteria": [
            "Level 0 Far-left.",
            "Level 1 Left.",
            "Level 2 Center.",
            "Level 3 Right.",
            "Level 4 Far-right.",
        ]
    },
    "overall_score": {
        "type": "score",
        "instructions": "Evaluate the overall ideological score of the video.",
        "criteria": [
            "Level 0 Far-left.",
            "Level 1 Left.",
            "Level 2 Center.",
            "Level 3 Right.",
            "Level 4 Far-right.",
        ]
    }
}

def _extract_probs(question_data: dict) -> dict[str, float]:
    probs = question_data.get("probabilities", {})
    return {str(k): round(float(probs.get(str(k), probs.get(k, 0.0))), 4) for k in range(5)}

def _local_api_inference(state : str, questions : dict, client) -> dict:
    ans = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for q_key, q_data in questions.items():
        # local Jev needs the scoring criteria baked into the question text if we aren't using "choice"
        criteria_text = "\n".join(q_data.get("criteria", []))
        full_question = f"{q_data['instructions']}\n{criteria_text}".strip()

        payload = {
            "kind": q_data["type"],
            "state": state,
            "question": full_question
        }

        resp = client.post("/decide", json=payload)
        resp.raise_for_status()
        res_data = resp.json()

        if "usage" in res_data:
            usage["prompt_tokens"] += res_data["usage"].get("prompt_tokens", 0)
            usage["completion_tokens"] += res_data["usage"].get("completion_tokens", 0)

        probs_dict = dict(zip(res_data["options"], res_data["probabilities"]))
        confidence = max(res_data["probabilities"])

        if q_data["type"] == "score":
            ans[q_key] = {
                "score": res_data.get("score", sum(i * p for i, p in enumerate(res_data["probabilities"]))),
                "confidence": confidence,
                "probabilities": probs_dict
            }
        elif q_data["type"] == "noul":
            ans[q_key] = {
                "noul": res_data.get("choice") == "true",
                "confidence": confidence,
                "probabilities": probs_dict  
            }

    return ans, usage 

def _remote_api_inference(state : str, questions : dict, client) -> dict:
    payload = {
        "model": DECISION_MODEL,
        "state": state,
        "questions": questions,
    }
    resp = client.post("/decisions", json=payload)
    resp.raise_for_status()
    res_data = resp.json()
    return res_data.get("answers"), res_data.get("usage")

def create_decision_client():
    if LOCAL_SERVER:
        limits = httpx.Limits(
            max_connections=256,
            max_keepalive_connections=128,
        )
        decision_client = httpx.Client(
            base_url="http://localhost:8000/v1",
            headers={"Authorization": "Bearer EMPTY"},
            timeout=300.0,
            limits=limits,
        )
    else:
        decision_client = httpx.Client(
            base_url="https://openrouter.ai/api/alpha",
            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
            timeout=30.0,
        )
    return decision_client

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
def score_pair(pair: tuple, client, questions = Q_ALIGNMENT):
    (video_a, analysis_a), (video_b, analysis_b) = pair

    state = f"""
    [VIDEO A]
    Channel: {video_a['channel']}
    Title: {video_a['title']}
    {f'Analysis: {analysis_a.model_dump_json()}' if analysis_a else ''}

    [VIDEO B]
    Channel: {video_b['channel']}
    Title: {video_b['title']}
    {f'Analysis: {analysis_b.model_dump_json()}' if analysis_b else ''}
    """

    if LOCAL_SERVER:
        ans, usage = _local_api_inference(state, questions, client)
    else:
        ans, usage = _remote_api_inference(state, questions, client)

    output = {
        "id_a": video_a.get("id"),
        "id_b": video_b.get("id"),
        "channel_a": video_a.get("channel"),
        "channel_b": video_b.get("channel"),
        "title_a": video_a["title"],
        "title_b": video_b["title"],
        "slant_a": video_a.get("slant"),
        "slant_b": video_b.get("slant"),
    }

    if questions == Q_ALIGNMENT or questions == Q_ABSOLUTE:
        for q in ans.keys():
            output[q] = round((ans[q]["score"] - 2.0) / 2.0, 4)
            output[f"{q}_confidence"] = round(ans[q].get("confidence", 0.0), 4)
            output[f"{q}_probabilities"] = _extract_probs(ans[q])      

    if questions == Q_RANKING:
        for q in ans.keys():
            output[q] = ans[q]["noul"]
            output[f"{q}_confidence"] = round(ans[q].get("confidence", 0.0), 4)
            output[f"{q}_probabilities"] = _extract_probs(ans[q])

    if usage:
        output.update({
            "cost": round(float(usage.get("cost", 0.0)), 6),
        })

    return output

def generate_pairs(videos : list[dict], pairs_per_video = PAIRS_PER_VIDEO) -> list[tuple[dict, dict]]:
    random.seed(42)
    return [
        pair
        for _ in range(pairs_per_video)
        for pair in itertools.pairwise(random.sample(videos, len(videos)))
    ]    

def score_pairs(pairs : list[tuple[dict, dict|None]], dest: Path, client, questions=Q_ALIGNMENT):
    df_pairs = pl.read_parquet(dest) if dest.exists() else pl.DataFrame()
    start_idx = len(df_pairs)
    remaining_pairs = pairs[start_idx:]

    with tqdm(total=len(pairs), initial=start_idx, desc=f"Scoring pairs ({DECISION_MODEL})", unit="pair") as pbar:
        for i in range(0, len(remaining_pairs), CHUNK_SIZE):
            chunk = remaining_pairs[i : i + CHUNK_SIZE]
            with ThreadPoolExecutor(max_workers=NUM_WORKERS_DECISION) as executor:
                scored = []
                score_pair_args = partial(score_pair, questions=questions, client=client)
                for res in executor.map(score_pair_args, chunk):
                    scored.append(res)
                    pbar.update(1)
            chunk_df = pl.DataFrame(scored)
            df_pairs = pl.concat([df_pairs, chunk_df]) if len(df_pairs) > 0 else chunk_df
            df_pairs.write_parquet(dest)

    print(f"\nDone! Saved {len(df_pairs)} pairs to {dest.name}")


