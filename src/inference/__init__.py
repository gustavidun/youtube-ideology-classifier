from src.config import MODEL_DIR

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def load_model(name:str):
    model_dir = MODEL_DIR / name

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(str(model_dir), device=device)
    print(f"Model {name} successfully loaded on {device}.")
    return model

def encode_batch(model : SentenceTransformer, texts:list[str]):
    print(f"Generating embeddings for {len(texts):,} videos...")

    embeddings = model.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embeddings = embeddings.astype(np.float32)
    print(f"Embeddings complete.")
    return embeddings