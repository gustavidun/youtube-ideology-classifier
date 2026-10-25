import argparse
import os
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator

from src.config import OUT_DIR, ROOT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train():
    parser = argparse.ArgumentParser(description="Train a political sentence embedding model from a config file.")
    parser.add_argument("--path", required=True, help="Path to YAML or JSON config file")
    args_cli = parser.parse_args()

    with open(args_cli.path) as f:
        cfg = yaml.safe_load(f)

    model = SentenceTransformer(cfg["model_name"])
    if "max_seq_length" in cfg:
        model.max_seq_length = cfg["max_seq_length"]

    train_dict, eval_dict, loss_functions = {}, {}, {}
    for task in cfg.get("tasks", ["galtan"]):
        ds = load_dataset(
            "parquet",
            data_files={
                "train": str(OUT_DIR / "train" / f"{task}.parquet"),
                "test": str(OUT_DIR / "test" / f"{task}.parquet"),
            },
            keep_in_memory=True,
        )
        task_key = f"{task}_task"
        train_dict[task_key] = ds["train"]
        eval_dict[task_key] = ds["test"]
        loss_functions[task_key] = losses.CoSENTLoss(model)

    training_args = cfg.get("training_args", {})
    out_dir = training_args.get("output_dir", "models/political-embedder")
    training_args["output_dir"] = str(ROOT / out_dir) if not os.path.isabs(out_dir) else out_dir
    args = SentenceTransformerTrainingArguments(**training_args)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dict,
        eval_dataset=eval_dict,
        loss=loss_functions
    )
    trainer.train()


if __name__ == "__main__":
    train()