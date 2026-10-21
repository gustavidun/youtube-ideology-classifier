import polars as pl
from config import OUT_DIR

CONFIDENCE_THRESHOLD = 0.3

if __name__ == "__main__":
    context = pl.read_parquet(OUT_DIR / "context.parquet")
    pairs = pl.read_parquet(OUT_DIR / "pair_similarity.parquet")

    #concatenate metadata
    context = context.with_columns(
        assembled_text=pl.concat_str(
            [
                pl.lit("Channel: "), pl.col("channel"), pl.lit("\n"),
                pl.lit("Title: "), pl.col("title"), pl.lit("\n"),
                pl.lit("Description: "), pl.col("description"), pl.lit("\n"),
                pl.lit("Tags: "), pl.col("tags")
            ]
        )
    )

    #join
    pairs = pairs.join(
        context.select(["id", "assembled_text"]), 
        left_on="id_a", 
        right_on="id", 
        how="left"
    ).rename({"assembled_text": "sentence1"})

    pairs = pairs.join(
        context.select(["id", "assembled_text"]), 
        left_on="id_b", 
        right_on="id", 
        how="left"
    ).rename({"assembled_text": "sentence2"})

    # filter nulls and scores with low confidence
    economic = (
        pairs.filter(pl.col("economic_confidence") >= CONFIDENCE_THRESHOLD)
        .select(["sentence1", "sentence2", "economic_alignment"])
        .drop_nulls()
        .rename({"economic_alignment": "score"})
    )
    galtan = (
        pairs.filter(pl.col("galtan_confidence") >= CONFIDENCE_THRESHOLD)
        .select(["sentence1", "sentence2", "galtan_alignment"])
        .drop_nulls()
        .rename({"galtan_alignment": "score"})
    )
    overall = (
        pairs.filter(pl.col("overall_ideological_confidence") >= CONFIDENCE_THRESHOLD)
        .select(["sentence1", "sentence2", "overall_ideological_alignment"])
        .drop_nulls()
        .rename({"overall_ideological_alignment": "score"})
    )

    (OUT_DIR / "train").mkdir(parents=True, exist_ok=True)

    economic.write_parquet(OUT_DIR / "train" / "economic.parquet")
    galtan.write_parquet(OUT_DIR / "train" / "galtan.parquet")
    overall.write_parquet(OUT_DIR / "train" / "overall.parquet")

    print(len(economic), len(galtan), len(overall))