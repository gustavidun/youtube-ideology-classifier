import polars as pl
from datasets import Dataset
from src.config import OUT_DIR

CONFIDENCE_THRESHOLD = 0.3
TEST_SIZE = 0.1
SEED = 42

def process_and_split(df: pl.DataFrame, name: str):
    """Converts Polars to HF Dataset, splits deterministically, and saves."""
    hf_dataset = Dataset(df.to_arrow())
    
    split_dataset = hf_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    
    (OUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "test").mkdir(parents=True, exist_ok=True)
    
    split_dataset["train"].to_parquet(OUT_DIR / "train" / f"{name}.parquet")
    split_dataset["test"].to_parquet(OUT_DIR / "test" / f"{name}.parquet")
    
    print(f"{name.capitalize()}: {len(split_dataset['train'])} train, {len(split_dataset['test'])} test")

if __name__ == "__main__":
    context = pl.read_parquet(OUT_DIR / "context.parquet")
    pairs = pl.read_parquet(OUT_DIR / "pair_similarity.parquet")

    # concatenate metadata
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

    # join
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

    process_and_split(economic, "economic")
    process_and_split(galtan, "galtan")
    process_and_split(overall, "overall")