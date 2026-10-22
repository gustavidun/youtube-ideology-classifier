from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator

from config import MODEL_DIR, OUT_DIR

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRUNCATE = 1024

economic_dataset = load_dataset(
    "parquet", 
    data_files=str(OUT_DIR / "train" / "economic.parquet"), 
    split="train", 
    keep_in_memory=True  
)
galtan_dataset = load_dataset(
    "parquet", 
    data_files=str(OUT_DIR / "train" / "galtan.parquet"), 
    split="train", 
    keep_in_memory=True
)

economic_split = economic_dataset.train_test_split(test_size=0.1)
galtan_split = galtan_dataset.train_test_split(test_size=0.1)

train_dict = {
    "economic_task": economic_split["train"],
    "gal_tan_task": galtan_split["train"]
}

eval_dict = {
    "economic_task": economic_split["test"],
    "gal_tan_task": galtan_split["test"]
}

model = SentenceTransformer(
    "Alibaba-NLP/gte-large-en-v1.5",
    #model_kwargs={"attn_implementation": "sdpa", "device_map": "auto"},
    #tokenizer_kwargs={"padding_side": "left"}
    trust_remote_code=True
)

model.max_seq_length = TRUNCATE


loss_functions = {
    "economic_task": losses.CoSENTLoss(model),
    "gal_tan_task": losses.CoSENTLoss(model)
}

args = SentenceTransformerTrainingArguments(
    output_dir=str(MODEL_DIR / "qwen-political-embedder-final"),
    multi_dataset_batch_sampler="round_robin",
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=64,                       
    bf16=True,
    dataloader_num_workers=8,             
    dataloader_prefetch_factor=4,
    dataloader_pin_memory=True,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    metric_for_best_model="eval_economic_spearman_cosine", # Optional: track best model
    load_best_model_at_end=True
)

def create_similarity_evaluator(dataset, name):
    return EmbeddingSimilarityEvaluator(
        sentences1=dataset["sentence1"],
        sentences2=dataset["sentence2"],
        scores=dataset["score"], 
        name=name,
        show_progress_bar=False
    )

economic_evaluator = create_similarity_evaluator(eval_dict["economic_task"], "economic")
galtan_evaluator = create_similarity_evaluator(eval_dict["gal_tan_task"], "galtan")
combined_evaluator = SequentialEvaluator([economic_evaluator, galtan_evaluator])

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dict,
    eval_dataset=eval_dict,
    loss=loss_functions,
    evaluator=combined_evaluator
)

if __name__ == "__main__":
    trainer.train() 