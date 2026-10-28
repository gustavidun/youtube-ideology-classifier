import argparse
import polars as pl
from typing import get_args

from src.config import OUT_DIR, SLANTS_TRAIN
from src.teacher import DECISION_MODEL, CONTEXT_MODEL, C_MODELS, D_MODELS, LOCAL_SERVER
import src.teacher.pair_alignment as pa
import src.teacher.context as pc

def main():
    parser = argparse.ArgumentParser(description="Generate context and pairwise predictions.")
    parser.add_argument("--mode", choices=["context", "pairs"], required=True, help="Mode to run")
    parser.add_argument("--videos_path", default=str(SLANTS_TRAIN), required=False)
    parser.add_argument("--context_path", default="context.parquet", required=False)
    parser.add_argument("--out_path", default="pair_similarity.parquet", required=False)
    parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--context_model", default="Qwen/Qwen3-8B", required=False)
    parser.add_argument("--decision_model", default="autotrust/JEV-27B", required=False)
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=False)
    
    args_cli = parser.parse_args()

    if args_cli.decision_model not in get_args(D_MODELS):
        raise Exception(f"Decision model '{args_cli.decision_model}' not in supported models {get_args(D_MODELS)}.")

    if args_cli.context_model not in get_args(C_MODELS):
        raise Exception(f"Context model '{args_cli.context_model}' not in supported models {get_args(C_MODELS)}.")

    pa.LOCAL_SERVER = args_cli.local
    pc.LOCAL_SERVER = args_cli.local

    decision_client = pa.create_decision_client()
    context_client = pc.create_context_client()

    if args_cli.mode == "context":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        filter_parquet = OUT_DIR / args_cli.context_path

        if filter_parquet.exists() and not args_cli.overwrite:
            raise Exception(f"Out file {filter_parquet} exists and overwrite is set to False.")

        videos = (
            pl.read_csv(args_cli.videos_path, ignore_errors=True)
            .select(["id", "title", "channel", "description", "tags", "slant"])
            .filter(pl.col("id").str.contains(r"^[A-Za-z0-9_-]{11}$"))
            .unique(subset=["id"], keep="first")
            .to_dicts()
        )
        pc.analyze_videos(videos, filter_parquet, context_client)

    elif args_cli.mode == "pairs":
        context = OUT_DIR / args_cli.context_path
        pairwise_parquet = OUT_DIR / args_cli.out_path

        if pairwise_parquet.exists() and not args_cli.overwrite:
            raise Exception(f"Out file {pairwise_parquet} exists and overwrite is set to False.")

        df_videos = pl.read_parquet(context)
        political_records = df_videos.filter(pl.col("political_saliency") > 0).to_dicts()  # filter out completely non-political videos
        print(f"Loaded {len(political_records)} political videos from {context.name}.")

        political = [(v, pc.VideoAnalysis.model_validate_json(v["analysis_json"])) for v in political_records]

        pairs = pa.generate_pairs(political)
        pa.score_pairs(pairs, pairwise_parquet, decision_client, questions=pa.Q_RANKING)


if __name__ == "__main__":
    main()
