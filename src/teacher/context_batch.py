import json
from pathlib import Path
import polars as pl
from openai.lib._parsing._completions import type_to_response_format_param

from config import OUT_DIR, SLANTS_TRAIN
from teacher.context import (
    client, MODEL_NAME, PAIRS_PER_VIDEO,
    VIDEO_ANALYSIS_PROMPT, PAIRWISE_PROMPT,
    VideoAnalysis, PairwiseEvaluation,
)

BATCH_DIR = OUT_DIR / "batch" / "prompts"
MAX_BATCH = 50000

def export_batch(items, prompt_fn, model_cls, prefix: str):
    """Shared generator: chunks any item list into 50k JSONL requests using model_cls schema."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    rf = type_to_response_format_param(model_cls)
    for part, start in enumerate(range(0, len(items), MAX_BATCH), 1):
        chunk = items[start : start + MAX_BATCH]
        path = BATCH_DIR / f"{prefix}_part_{part}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in chunk:
                cid, sys_p, usr_p = prompt_fn(item)
                f.write(json.dumps({
                    "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME, "temperature": 0.1,
                        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                        "response_format": rf,
                    }
                }, ensure_ascii=False) + "\n")
        print(f"Created {path.name} ({len(chunk)} items, {round(path.stat().st_size / (1024*1024), 1)} MB)")


if __name__ == "__main__":
    filter_p = OUT_DIR / "filter_dataset.parquet"
    videos_df = (
        pl.read_csv(SLANTS_TRAIN, ignore_errors=True)
        .select(["id", "title", "channel", "description", "tags", "slant"])
        .filter(pl.col("id").str.contains(r"^[A-Za-z0-9_-]{11}$"))
        .unique(subset=["id"], keep="first")
    )
    existing_ids = set(pl.read_parquet(filter_p)["id"]) if filter_p.exists() else set()
    remaining = [v for v in videos_df.to_dicts() if v["id"] not in existing_ids]
    print(f"Exporting Stage 1 batch ({len(remaining)} unique videos, skipping {len(existing_ids)} already analyzed)...")
    export_batch(
        remaining,
        lambda v: (f"video_{v['id']}", VIDEO_ANALYSIS_PROMPT, f"Channel: {v['channel']}\nTitle: {v['title']}\nDescription: {v['description']}\nTags: {v['tags']}"),
        VideoAnalysis,
        "stage1"
    )
