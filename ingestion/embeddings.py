from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(">>> Loading embedding model <<<")
        _model = SentenceTransformer(MODEL_NAME)
        print(">>> Embedding model loaded <<<")
    return _model

def embed_text(text: str) -> list[float]:
    """
    Deterministic local embedding.
    Model is loaded lazily on first request.
    """
    model = _get_model()

    vector = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    if vector is None or len(vector) != EMBEDDING_DIM:
        raise RuntimeError("Embedding generation failed")

    return vector.tolist()
