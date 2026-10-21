import polars as pl
from config import OUT_DIR, SLANTS_TRAIN


def unpack_batch_results():
    """Unpacks all completed batch JSONL files from batch/processed into filter_dataset.parquet."""
    processed_dir = OUT_DIR / "context" / "batch" / "processed"
    context = OUT_DIR / "context.parquet"
    files = sorted(processed_dir.glob("*.jsonl"))
    if not files:
        print(f"No processed batch files found in {processed_dir}")
        return

    print(f"Unpacking {len(files)} batch files from {processed_dir}...")
    dfs = []
    for p in files:
        df = pl.read_ndjson(p)
        body = df["response"].struct.field("body")
        content = body.struct.field("choices").list.get(0).struct.field("message").struct.field("content")
        usage = body.struct.field("usage")
        extracted = pl.DataFrame({
            "id": df["custom_id"].str.replace("video_", ""),
            "analysis_json": content,
            "prompt_tokens": usage.struct.field("prompt_tokens"),
            "completion_tokens": usage.struct.field("completion_tokens"),
            "political_saliency": content.str.json_path_match("$.political_saliency").cast(pl.Float64),
        })
        dfs.append(extracted)

    new_data = pl.concat(dfs)
    meta = (
        pl.read_csv(SLANTS_TRAIN, ignore_errors=True)
        .select(["id", "title", "channel", "description", "tags", "slant"])
        .filter(pl.col("id").str.contains(r"^[A-Za-z0-9_-]{11}$"))
        .unique(subset=["id"], keep="first")
    )
    cols = ["id", "title", "channel", "description", "tags", "slant", "political_saliency", "analysis_json", "prompt_tokens", "completion_tokens"]
    joined = new_data.join(meta, on="id", how="left").select(cols)
    joined.write_parquet(context)

    political_n = len(joined.filter(pl.col("political_saliency") > 0))
    print(f"Saved {len(joined)} total videos to {context.name} ({political_n} political).")


if __name__ == "__main__":
    unpack_batch_results()
