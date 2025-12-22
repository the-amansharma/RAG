import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

HF_TOKEN = os.getenv("HF_TOKEN")  # ✅ standard name

if not HF_TOKEN:
    raise RuntimeError("❌ HF_TOKEN is missing from environment variables.")

# ✅ Force Hugging Face Router
client = InferenceClient(
    model=MODEL_ID,
    token=HF_TOKEN,
    base_url="https://router.huggingface.co"
)

def embed_text(text: str) -> list[float]:
    """
    Generate embeddings using Hugging Face Router (feature-extraction).
    Returns a flat list[float] of length EMBEDDING_DIM.
    """
    try:
        response = client.feature_extraction(
            text,
            model=MODEL_ID
        )

        # Normalize output shape
        if hasattr(response, "tolist"):
            vector = response.tolist()
        else:
            vector = response

        # Handle [[...]] vs [...]
        if isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]

        # Hard validation (prevents Qdrant corruption)
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(vector)}"
            )

        return vector

    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}") from e
