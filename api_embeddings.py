"""
Embeddings module for API - Uses only HuggingFace Hub API (no local models).
This ensures consistent behavior across different deployment environments.
"""
import os
import logging
from typing import List
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_ID = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# Initialize HuggingFace client
HF_TOKEN = os.getenv("HF_TOKEN")
_hf_client = None

try:
    if HF_TOKEN:
        _hf_client = InferenceClient(token=HF_TOKEN)
        logger.info(f"✅ Using HuggingFace Hub API with token: {MODEL_ID}")
    else:
        _hf_client = InferenceClient()
        logger.warning("⚠️ Using HuggingFace Hub API without token (rate limits may apply)")
        logger.info(f"✅ Using HuggingFace Hub API: {MODEL_ID}")
except Exception as e:
    logger.error(f"❌ Failed to initialize HuggingFace client: {e}")
    raise

# --------------------------------------------------
# EMBEDDING FUNCTIONS
# --------------------------------------------------

def embed_text(text: str) -> List[float]:
    """
    Embed a single text using HuggingFace Hub API.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return [0.0] * EMBEDDING_DIM
    
    try:
        # Use HuggingFace Inference API
        response = _hf_client.feature_extraction(text=text, model=MODEL_ID)
        
        # Response format: [[float, float, ...]] - list of lists
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], list):
                return response[0]  # Return first (and usually only) embedding
            else:
                return response  # Already a flat list
        else:
            logger.error(f"Unexpected response format from HuggingFace API: {type(response)}")
            return [0.0] * EMBEDDING_DIM
            
    except Exception as e:
        logger.error(f"Failed to embed text via HuggingFace API: {e}")
        # Return zero vector on error
        return [0.0] * EMBEDDING_DIM

def embed_batch(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    Embed multiple texts using HuggingFace Hub API.
    Processes texts individually (API doesn't support true batch processing).
    Uses smaller batch_size to avoid rate limits.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process before a small delay (for rate limiting)
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    # Filter out empty texts
    texts = [t.strip() if t else "" for t in texts]
    texts = [t for t in texts if t]
    
    if not texts:
        return []
    
    all_embeddings = []
    
    # Process texts individually (HuggingFace API processes one at a time)
    for i, text in enumerate(texts):
        try:
            embedding = embed_text(text)
            all_embeddings.append(embedding)
            
            # Small delay every batch_size items to avoid rate limits
            if (i + 1) % batch_size == 0:
                import time
                time.sleep(0.1)  # 100ms delay
                
        except Exception as e:
            logger.error(f"Failed to embed text {i+1}/{len(texts)}: {e}")
            # Return zero vector for failed text
            all_embeddings.append([0.0] * EMBEDDING_DIM)
    
    return all_embeddings

# Export constants
__all__ = ["embed_text", "embed_batch", "MODEL_ID", "EMBEDDING_DIM"]

