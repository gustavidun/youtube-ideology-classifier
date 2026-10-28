""" Generate and evaluate teacher scores on context, alignment, ranking, and absolute estimation methods. """

import polars as pl
from pathlib import Path

from src.teacher.pair_alignment import generate_pairs, score_pairs, Q_ALIGNMENT, Q_RANKING, Q_ABSOLUTE
from src.teacher.context import analyze_videos, VideoAnalysis
from src.config import DATA_DIR, get_model_name, CONTEXT_MODEL, DECISION_MODEL

PAIRS_PER_VIDEO = 5
EVAL_PATH = DATA_DIR / "out" / "eval"
model_names = get_model_name(CONTEXT_MODEL) + "-" + get_model_name(DECISION_MODEL)

CONTEXT_DEST = EVAL_PATH / f"{model_names}" / f"human_metadata_with_context.parquet"
print("Metadata dest: ", CONTEXT_DEST)

CONDITIONS = {
    "context": {
        "ranking": Q_RANKING,
        "absolute": Q_ABSOLUTE,
        "alignment": Q_ALIGNMENT,
    },
    "no_context": {
        "ranking": Q_RANKING,
        "absolute": Q_ABSOLUTE,
        "alignment": Q_ALIGNMENT,
    }
}

CONTEXT_DEST.parent.mkdir(exist_ok=True)
human_metadata = pl.read_parquet(DATA_DIR / "human_labels" / "vid_metadata.parquet")

def generate_context():
    if CONTEXT_DEST.exists():
        return pl.read_parquet(CONTEXT_DEST)
    analyze_videos(human_metadata.to_dicts(), CONTEXT_DEST)
    return pl.read_parquet(CONTEXT_DEST)

if __name__ == "__main__":
    for cond in CONDITIONS["context"].keys():
        context = generate_context()
        context = [(v, VideoAnalysis.model_validate_json(v["analysis_json"])) for v in context.to_dicts()]
        pairs = generate_pairs(context, PAIRS_PER_VIDEO)
        score_pairs(pairs, Path(CONTEXT_DEST.parent / f"context_{cond}.parquet"), CONDITIONS["context"][cond])
    
    for cond in CONDITIONS["no_context"].keys():
        metadata = [(v, None) for v in human_metadata.to_dicts()]
        pairs = generate_pairs(metadata, PAIRS_PER_VIDEO)
        score_pairs(pairs, Path(CONTEXT_DEST.parent / f"no_context_{cond}.parquet"), CONDITIONS["no_context"][cond])