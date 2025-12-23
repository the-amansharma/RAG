import os
from huggingface_hub import InferenceClient

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_ID = "https://router.huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

HF_TOKEN = os.getenv("HF_TOKEN")
HF_ENDPOINT = os.getenv("HF_ENDPOINT")  # 👈 take from env

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing from environment variables.")

if not HF_ENDPOINT:
    raise RuntimeError("HF_ENDPOINT is missing from environment variables.")

# NOTE:
# InferenceClient treats `base_url` as the actual endpoint
# so here we pass HF_ENDPOINT as base_url and DO NOT pass model there.
client = InferenceClient(
    model=MODEL_ID,
    token=HF_TOKEN
    
)

def embed_text(text: str) -> list[float]:
    """
    Generate embedding via Hugging Face Inference API.
    Returns a flat list[float] of length EMBEDDING_DIM.
    """
    try:
        response = client.feature_extraction(text)

        if hasattr(response, "tolist"):
            vector = response.tolist()
        else:
            vector = response

        if isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]

        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(vector)}"
            )

        return vector

    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}") from e
