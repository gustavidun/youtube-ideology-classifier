from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

from config import MODEL_DIR, OUT_DIR

economic_dataset = load_dataset("parquet", data_files=str(OUT_DIR / "train" / "economic.parquet"), split="train")
galtan_dataset = load_dataset("parquet", data_files=str(OUT_DIR / "train" / "galtan.parquet"), split="train")

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

loss_functions = {
    "economic_task": losses.CoSENTLoss(model),
    "gal_tan_task": losses.CoSENTLoss(model)
}

args = SentenceTransformerTrainingArguments(
    output_dir=str(MODEL_DIR / "qwen-political-embedder-final"),
    multi_dataset_batch_sampler="round_robin",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,        
    gradient_accumulation_steps=1,        
    bf16=True,                            
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset={
        "economic_task": economic_dataset,
        "gal_tan_task": galtan_dataset
    },
    loss=loss_functions
)

trainer.train()