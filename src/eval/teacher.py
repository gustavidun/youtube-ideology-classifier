""" Generate teacher scores on human-labeled test set across conditions (with/without context, ranking, alignment, pointwise estimation etc.) """

import argparse
import polars as pl
from pathlib import Path

from src.teacher.pair_alignment import generate_pairs, score_pairs, Q_ALIGNMENT, Q_RANKING, Q_ABSOLUTE, create_decision_client
from src.teacher.context import analyze_videos, VideoAnalysis, create_context_client
from src.teacher import get_model_name, CONTEXT_MODEL, DECISION_MODEL, LOCAL_SERVER
from src.config import DATA_DIR
import src.teacher.pair_alignment as pa
import src.teacher.context as pc

parser = argparse.ArgumentParser()
parser.add_argument("--context_model", default=CONTEXT_MODEL)
parser.add_argument("--decision_model", default=DECISION_MODEL)
parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=LOCAL_SERVER)
args, _ = parser.parse_known_args()

pa.LOCAL_SERVER = args.local
pc.LOCAL_SERVER = args.local

PAIRS_PER_VIDEO = 5
EVAL_PATH = DATA_DIR / "out" / "eval"
model_names = get_model_name(args.context_model) + "-" + get_model_name(args.decision_model)

CONTEXT_DEST = EVAL_PATH / "teacher" / f"{model_names}" / "human_metadata_with_context.parquet"
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

CONTEXT_DEST.parent.mkdir(parents=True, exist_ok=True)
human_metadata = pl.read_parquet(DATA_DIR / "human_labels" / "vid_metadata.parquet")

def generate_context():
    if CONTEXT_DEST.exists():
        return pl.read_parquet(CONTEXT_DEST)
    context_client = create_context_client()
    analyze_videos(human_metadata.to_dicts(), CONTEXT_DEST, context_client, model=args.context_model)
    return pl.read_parquet(CONTEXT_DEST)

if __name__ == "__main__":
    decision_client = create_decision_client(model=args.decision_model)
    for cond in CONDITIONS["context"].keys():
        context = generate_context()
        context = [(v, VideoAnalysis.model_validate_json(v["analysis_json"])) for v in context.to_dicts()]
        pairs = generate_pairs(context, PAIRS_PER_VIDEO)
        score_pairs(pairs, Path(CONTEXT_DEST.parent / f"context_{cond}.parquet"), decision_client, CONDITIONS["context"][cond], model=args.decision_model)
    
    for cond in CONDITIONS["no_context"].keys():
        metadata = [(v, None) for v in human_metadata.to_dicts()]
        pairs = generate_pairs(metadata, PAIRS_PER_VIDEO)
        score_pairs(pairs, Path(CONTEXT_DEST.parent / f"no_context_{cond}.parquet"), decision_client, CONDITIONS["no_context"][cond], model=args.decision_model)
