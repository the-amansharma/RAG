from sentence_transformers import SentenceTransformer
import numpy as np

# --------------------------------------------------
# MODEL (LOADED ONCE)
# --------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


_model = SentenceTransformer(MODEL_NAME)

def embed_text(text: str) -> list[float]:
    """
    Deterministic local embedding.
    Supports long text safely.
    """
    vector = _model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    if vector is None or len(vector) != EMBEDDING_DIM:
        raise RuntimeError("Embedding generation failed")

    return vector.tolist()
