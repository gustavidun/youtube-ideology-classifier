import polars as pl

from src.inference import load_model, encode_batch
from src.config import OUT_DIR

CONTEXT_PATH = OUT_DIR / "context.parquet"
MODEL_NAMES = [
    "bge-m3"
]

if __name__ == "__main__":
    df_context = pl.read_parquet(CONTEXT_PATH)

    # Filter for political videos with valid titles
    df_political = (
        df_context
        .filter((pl.col("political_saliency") > 0) & pl.col("title").is_not_null())
        .with_columns(
            assembled_text=pl.concat_str(
                [
                    pl.lit("Channel: "), pl.col("channel").fill_null(""), pl.lit("\n"),
                    pl.lit("Title: "), pl.col("title").fill_null(""), pl.lit("\n"),
                    pl.lit("Description: "), pl.col("description").fill_null(""), pl.lit("\n"),
                    pl.lit("Tags: "), pl.col("tags").fill_null("")
                ]
            )
        )
    )

    print(f"Total videos in context: {len(df_context):,}")
    print(f"Political videos to embed: {len(df_political):,} ({len(df_political)/len(df_context):.1%})")


    for name in MODEL_NAMES:
        model = load_model(name)
        embeddings = encode_batch(model, df_political["assembled_text"].to_list())

        embeddings_path = OUT_DIR / f"political_embeddings_{name}.parquet"
        print(f"Writing embeddings to: {embeddings_path}")

        df_embeddings = df_political.select([
            "id", "channel", "title", "slant", "political_saliency"
        ]).with_columns(
            embedding=pl.Series("embedding", embeddings.tolist())
        )

        df_embeddings.write_parquet(embeddings_path)
        print(f"Done! Successfully saved {len(df_embeddings):,} rows to {embeddings_path.name}.")