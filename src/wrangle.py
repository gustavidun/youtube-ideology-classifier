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

def wrangle_pairs(pairs: pl.DataFrame, context: pl.DataFrame):
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
    return pairs

def filter_pairs(
    pairs: pl.DataFrame,
    score_cols: list = ["economic_alignment", "galtan_alignment", "overall_ideological_alignment"],
    confidence_cols: list = ["economic_confidence", "galtan_confidence", "overall_ideological_confidence"]
):
    if confidence_cols:
        for col in confidence_cols:
            pairs = pairs.filter(pl.col(col) >= CONFIDENCE_THRESHOLD)

    economic = (
        pairs
        .select(["sentence1", "sentence2", score_cols[0]])
        .drop_nulls()
        .rename({score_cols[0]: "score"})
    )
    galtan = (
        pairs
        .select(["sentence1", "sentence2", score_cols[1]])
        .drop_nulls()
        .rename({score_cols[1]: "score"})
    )
    overall = (
        pairs
        .select(["sentence1", "sentence2", score_cols[2]])
        .drop_nulls()
        .rename({score_cols[2]: "score"})
    )
    return economic, galtan, overall

if __name__ == "__main__":
    context = pl.read_parquet(OUT_DIR / "context.parquet")
    pairs = pl.read_parquet(OUT_DIR / "pair_similarity.parquet")
    pairs_ranked = pl.read_parquet(OUT_DIR / "pair_ranked.parquet")

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

    pairs = wrangle_pairs(pairs, context)
    pairs_ranked = wrangle_pairs(pairs_ranked, context)

    # filter nulls and scores with low confidence
    economic, galtan, overall = filter_pairs(pairs)
    economic_ranked, galtan_ranked, overall_ranked = filter_pairs(
        pairs_ranked,
        score_cols=["is_more_right_leaning_economic", "is_more_right_leaning_cultural", "is_more_right_leaning_overall"],
        confidence_cols=[] #there isnt a separate confidence score for nouls
    )

    #process_and_split(economic, "economic")
    #process_and_split(galtan, "galtan")
    #process_and_split(overall, "overall")
    process_and_split(economic_ranked, "economic_ranked")
    process_and_split(galtan_ranked, "galtan_ranked")
    process_and_split(overall_ranked, "overall_ranked")