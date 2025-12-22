import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# You must set HF_TOKEN in your .env or Vercel env vars
# Get one here: https://huggingface.co/settings/tokens
client = InferenceClient(token=os.environ.get("HF_API_KEY"))

def embed_text(text: str) -> list[float]:
    """
    Generate embedding via HuggingFace Hub Client.
    Uses 'feature_extraction' to get the raw vector.
    """
    if not client.token:
        # Graceful error to help debug deployment
        raise ValueError("❌ HF_API_KEY is missing from environment variables.")

    try:
        # feature_extraction returns the raw vector (list of floats)
        # We assume the input is a single string.
        response = client.feature_extraction(
            text,
            model=MODEL_ID
        )
        
        # The client might return a numpy array or list depending on version/response.
        # We ensure it's a flat list.
        # Response format for single string often: [0.1, 0.2, ...] or [[0.1, ...]]
        
        # Safe cast to list
        if hasattr(response, "tolist"):
             vector = response.tolist()
        else:
             vector = response

        # Handle nesting if the API returns [[...]] for a single input
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            return vector[0]
            
        return vector

    except Exception as e:
        print(f"⚠️ HuggingFace Client Error: {e}")
        raise e
