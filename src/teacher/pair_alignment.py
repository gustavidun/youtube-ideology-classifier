import itertools
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

from config import OUT_DIR, ROOT
from teacher.context import VideoAnalysis

load_dotenv(ROOT / ".env", override=True)

MODEL_NAME = "typesafe/jev-1.14"
NUM_WORKERS = 40
CHUNK_SIZE = 1000
PAIRS_PER_VIDEO = 5

http_client = httpx.Client(
    base_url="https://openrouter.ai/api/alpha",
    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
    timeout=30.0,
)

QUESTIONS = {
    "economic_alignment": {
        "type": "score",
        "instructions": "Evaluate ideological alignment of the video-pair on economic/fiscal policy. Focus on underlying worldview alignment versus opposition, even if the specific topics differ.",
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
        "instructions": "Evaluate ideological alignment of the video-pair on GAL-TAN (Green/Alternative/Libertarian vs Traditional/Authoritarian/Nationalist) cultural axes. Focus on underlying worldview alignment versus opposition, even if the specific topics differ.",
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
        "instructions": "Evaluate the overall directional ideological alignment of the video-pair. Heavily penalize pairs that discuss the same topic from opposing stances.",
        "criteria": [
            "Level 0: Direct Opposition. Diametrically opposed ideological viewpoints.",
            "Level 1: Moderate Opposition. Substantial conflict on ideological viewpoints.",
            "Level 2: Neutral / Orthogonal. No discernible ideological conflict or alignment.",
            "Level 3: Moderate Alignment. Substantial shared ideological viewpoints.",
            "Level 4: Identical Stance. Identical ideological viewpoints.",
        ],
    },
}


def _extract_probs(question_data: dict) -> dict[str, float]:
    probs = question_data.get("probabilities", {})
    return {str(k): round(float(probs.get(str(k), probs.get(k, 0.0))), 4) for k in range(5)}


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
def score_pair(pair: tuple):
    (video_a, analysis_a), (video_b, analysis_b) = pair
    state = f"""
    [VIDEO A]
    Channel: {video_a['channel']}
    Title: {video_a['title']}
    Analysis: {analysis_a.model_dump_json()}

    [VIDEO B]
    Channel: {video_b['channel']}
    Title: {video_b['title']}
    Analysis: {analysis_b.model_dump_json()}
    """
    payload = {
        "model": MODEL_NAME,
        "state": state,
        "questions": QUESTIONS,
    }
    resp = http_client.post("/decisions", json=payload)
    resp.raise_for_status()
    res_data = resp.json()
    ans = res_data["answers"]
    usage = res_data.get("usage", {})

    return {
        "id_a": video_a.get("id"),
        "id_b": video_b.get("id"),
        "channel_a": video_a.get("channel"),
        "channel_b": video_b.get("channel"),
        "title_a": video_a["title"],
        "title_b": video_b["title"],
        "slant_a": video_a.get("slant"),
        "slant_b": video_b.get("slant"),
        "economic_alignment": round((ans["economic_alignment"]["score"] - 2.0) / 2.0, 4),
        "economic_confidence": round(ans["economic_alignment"].get("confidence", 0.0), 4),
        "economic_probabilities": _extract_probs(ans["economic_alignment"]),
        "galtan_alignment": round((ans["galtan_alignment"]["score"] - 2.0) / 2.0, 4),
        "galtan_confidence": round(ans["galtan_alignment"].get("confidence", 0.0), 4),
        "galtan_probabilities": _extract_probs(ans["galtan_alignment"]),
        "overall_ideological_alignment": round((ans["overall_ideological_alignment"]["score"] - 2.0) / 2.0, 4),
        "overall_ideological_confidence": round(ans["overall_ideological_alignment"].get("confidence", 0.0), 4),
        "overall_ideological_probabilities": _extract_probs(ans["overall_ideological_alignment"]),
        "cost": round(float(usage.get("cost", 0.0)), 6),
    }


if __name__ == "__main__":
    context = OUT_DIR / "context.parquet"
    pairwise_parquet = OUT_DIR / "pair_similarity.parquet"

    df_videos = pl.read_parquet(context)
    political_records = df_videos.filter(pl.col("political_saliency") > 0).to_dicts() # filter out completely non-political videos
    print(f"Loaded {len(political_records)} political videos from {context.name}.")

    political = [(v, VideoAnalysis.model_validate_json(v["analysis_json"])) for v in political_records]

    random.seed(42)
    pairs = [
        pair
        for _ in range(PAIRS_PER_VIDEO)
        for pair in itertools.pairwise(random.sample(political, len(political)))
    ]
    print(f"Total pairs to score: {len(pairs)} ({PAIRS_PER_VIDEO} pairs/video)")

    df_pairs = pl.read_parquet(pairwise_parquet) if pairwise_parquet.exists() else pl.DataFrame()
    start_idx = len(df_pairs)
    remaining_pairs = pairs[start_idx:]

    with tqdm(total=len(pairs), initial=start_idx, desc=f"Scoring pairs ({MODEL_NAME})", unit="pair") as pbar:
        for i in range(0, len(remaining_pairs), CHUNK_SIZE):
            chunk = remaining_pairs[i : i + CHUNK_SIZE]
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                scored = []
                for res in executor.map(score_pair, chunk):
                    scored.append(res)
                    pbar.update(1)
            chunk_df = pl.DataFrame(scored)
            df_pairs = pl.concat([df_pairs, chunk_df]) if len(df_pairs) > 0 else chunk_df
            df_pairs.write_parquet(pairwise_parquet)

    print(f"\nDone! Saved {len(df_pairs)} pairs to {pairwise_parquet.name}")
