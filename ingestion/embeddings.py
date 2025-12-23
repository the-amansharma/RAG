import os
import requests

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing from environment variables.")

API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

def embed_text(text: str) -> list[float]:
    """
    Generate embedding using Hugging Face Router (raw HTTP).
    Returns a flat list[float] of length 384.
    """
    payload = {"inputs": text}

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()

    # HF may return [[...]] or [...]
    if isinstance(data, list) and data and isinstance(data[0], list):
        vector = data[0]
    else:
        vector = data

    if not isinstance(vector, list):
        raise RuntimeError("Invalid embedding response format")

    if len(vector) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(vector)}"
        )

    return vector
