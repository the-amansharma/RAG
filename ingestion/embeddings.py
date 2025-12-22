import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("❌ HF_TOKEN is missing from environment variables.")

# ✅ DO NOT pass base_url
client = InferenceClient(
    model=MODEL_ID,
    token=HF_TOKEN
)

def embed_text(text: str) -> list[float]:
    try:
        response = client.feature_extraction(text)

        # Normalize output
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
        raise RuntimeError(f"Embedding failed: {str(e)}") from e
