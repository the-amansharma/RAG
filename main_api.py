import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from ingestion.embeddings import embed_text
from legal_core import load_composite_text, is_ambiguous, MIN_SCORE

load_dotenv()

app = FastAPI(title="GST Legal Assistant")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "notification_instruments_cloud"
TOP_K = 3

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    success: bool
    message: str | None = None
    data: dict | None = None

# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/search", response_model=SearchResponse)
def search_notifications(payload: SearchRequest):
    query = payload.query.strip()
    if not query:
        return SearchResponse(success=False, message="Query is empty.")

    try:
        vector = embed_text(query)
    except Exception as e:
        return SearchResponse(success=False, message=f"Embedding failed: {str(e)}")

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=TOP_K,
            with_payload=True
        ).points
    except Exception as e:
        return SearchResponse(success=False, message=f"Database search failed: {str(e)}")

    if not results or results[0].score < MIN_SCORE:
        return SearchResponse(success=False, message="No relevant notification found.")

    top_result = results[0]
    instrument = top_result.payload

    try:
        full_text = load_composite_text(instrument["group_id"])
    except Exception:
        full_text = "Error loading text."

    return SearchResponse(
        success=True,
        data={
            "notification_no": instrument.get("notification_no"),
            "tax_type": instrument.get("tax_type"),
            "score": top_result.score,
            "is_ambiguous": is_ambiguous(results),
            "legal_text": full_text
        }
    )
