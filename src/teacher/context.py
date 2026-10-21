import itertools
import random
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import polars as pl
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from config import OUT_DIR, SLANTS_TRAIN, ROOT

load_dotenv(ROOT / ".env", override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

NUM_WORKERS = 60

VIDEO_ANALYSIS_PROMPT = """Analyze a single YouTube video to assess its political character.

- Use the channel context to establish the creator's ideological baseline.
- Decode irony, satire, and dog-whistles through that lens.
- Identify Actors: figures, guests, or authorities featured or deferred to.
- Deconstruct Targets & Stances:
  * Extracted Targets: institutions, policies, groups, or ideas discussed (separate from the speakers).
  * Detected Stances: posture (supportive, critical, satirical, oppositional) toward those targets.
- Assign political_saliency [0.0 to 1.0]:
  * 1.0: Highly explicit political commentary.
  * 0.5: Moderate or latent political commentary (retail PR, partisan guests, cultural grievances).
  * 0.0: Completely devoid of political commentary or ideological signaling.
"""

class VideoAnalysis(BaseModel):
    channel_context: str = Field(description="Creator's niche, genre, and known ideological leanings.")
    tone: str = Field(description="Irony, satire, literal framing, clickbait etc.")
    actors: list[str] = Field(description="Actors that appear to feature in the video (e.g. politicians, activists, journalists, corporations etc.)")
    extracted_targets: list[str] = Field(description="Policies, institutions, or groups referenced")
    detected_stances: str = Field(description="Stance toward identified targets")
    functional_intent: str = Field(description="The deeper purpose of the video (e.g. to inform, persuade, entertain, satirize)")
    political_saliency_reasoning: str = Field(description="Brief reasoning on video's political salience.")
    political_saliency: float = Field(description="Political salience of this video (0.0 = apolitical, 1.0 = explicit political commentary)", ge=0.0, le=1.0)

class PairwiseEvaluation(BaseModel):
    economic_overlap_reasoning: str = Field(
        description="Reasoning on economic policy alignment, conflict, or irrelevance."
    )
    economic_similarity_score: float | None = Field(
        default=None,
        ge=-1.0, le=1.0,
        description="Score from -1.0 to +1.0."
    )
    gal_tan_overlap_reasoning: str = Field(
        description="Reasoning on GAL-TAN alignment, conflict, or irrelevance."
    )
    galtan_similarity_score: float | None = Field(
        default=None,
        ge=-1.0, le=1.0,
        description="Score from -1.0 to +1.0."
    )

MODEL_NAME = "gpt-4o-mini"

def query_model(system_prompt: str, user_prompt: str, response_format, client: OpenAI = client):
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_format,
    )
    return response.choices[0].message.parsed, response.usage

def analyze_video(video_data: dict, client: OpenAI = client):
    user_prompt = f"""
    Channel: {video_data['channel']}
    Title: {video_data['title']}
    Description: {video_data['description']}
    Tags: {video_data['tags']}
    """
    analysis, usage = query_model(
        system_prompt=VIDEO_ANALYSIS_PROMPT,
        user_prompt=user_prompt,
        response_format=VideoAnalysis,
        client=client,
    )
    return {
        "id": video_data.get("id"),
        "title": video_data["title"],
        "channel": video_data["channel"],
        "slant": video_data.get("slant"),
        "political_saliency": analysis.political_saliency,
        "analysis_json": analysis.model_dump_json(),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


CHUNK_SIZE = 5000

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filter_parquet = OUT_DIR / "context.parquet"

    # Stage 1: Analyze videos (with chunked disk writes and auto-resume)
    videos = (
        pl.read_csv(SLANTS_TRAIN, ignore_errors=True)
        .select(["id", "title", "channel", "description", "tags", "slant"])
        .filter(pl.col("id").str.contains(r"^[A-Za-z0-9_-]{11}$"))
        .unique(subset=["id"], keep="first")
        .to_dicts()
    )

    df_videos = pl.read_parquet(filter_parquet) if filter_parquet.exists() else pl.DataFrame()
    if len(df_videos) < len(videos):
        remaining = videos[len(df_videos):]
        for i in range(0, len(remaining), CHUNK_SIZE):
            chunk = remaining[i : i + CHUNK_SIZE]
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                analyses = list(tqdm(executor.map(analyze_video, chunk), total=len(chunk), desc=f"Stage 1 chunk {i // CHUNK_SIZE + 1}"))
            df_videos = pl.concat([df_videos, pl.DataFrame(analyses)])
            df_videos.write_parquet(filter_parquet)
    print(f"{len(df_videos)} analyzed videos ready in {filter_parquet.name}")
