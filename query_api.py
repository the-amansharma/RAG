"""
JSON API for Query CLI Functionality
Exposes query CLI features as REST API endpoints.
"""
import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import logging

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("query-api")

# Import embeddings from API-specific module FIRST (uses only HuggingFace Hub API)
# This ensures all embedding calls use the HuggingFace API (no local models)
try:
    from api_embeddings import embed_text, embed_batch
    logger.info("✅ Loaded api_embeddings module")
except Exception as e:
    logger.error(f"❌ Failed to import api_embeddings: {e}", exc_info=True)
    # Create fallback functions to prevent app from failing
    def embed_text(text: str):
        raise RuntimeError("api_embeddings module failed to load. Check logs for details.")
    def embed_batch(texts: List[str]):
        raise RuntimeError("api_embeddings module failed to load. Check logs for details.")
    logger.warning("⚠️ Using fallback embedding functions - API will not work properly")

# Patch ingestion.embeddings to use our API embeddings BEFORE importing enhanced_search
# This ensures enhanced_search uses HuggingFace API instead of local models
try:
    import ingestion.embeddings as ingestion_embeddings_module
    ingestion_embeddings_module.embed_text = embed_text
    ingestion_embeddings_module.embed_batch = embed_batch
    logger.info("✅ Patched ingestion.embeddings to use HuggingFace Hub API (no local models)")
except Exception as e:
    logger.warning(f"⚠️ Failed to patch ingestion.embeddings: {e}")

# Import functions from query_cli (enhanced_search will now use our patched embeddings)
try:
    from query_cli import (
        COLLECTION_NAME,
        MIN_SCORE_THRESHOLD,
        CONTEXT_CHUNKS_BEFORE,
        CONTEXT_CHUNKS_AFTER,
        SHOW_TOP_N_RESULTS,
        evaluate_answer_quality,
        get_surrounding_chunks,
        build_filter,
        format_score,
        enhanced_search
    )
    logger.info("✅ Loaded query_cli module")
except Exception as e:
    logger.error(f"❌ Failed to import from query_cli: {e}", exc_info=True)
    # Set defaults if import fails
    COLLECTION_NAME = "notification_chunks"
    MIN_SCORE_THRESHOLD = 0.6
    CONTEXT_CHUNKS_BEFORE = 2
    CONTEXT_CHUNKS_AFTER = 2
    SHOW_TOP_N_RESULTS = 3
    enhanced_search = None
    evaluate_answer_quality = lambda q, r, b: {}
    get_surrounding_chunks = lambda *args, **kwargs: []
    build_filter = lambda *args, **kwargs: None
    format_score = lambda s: f"{s:.2%}"
    logger.warning("⚠️ Using fallback defaults - some features may not work")

# Initialize Qdrant client (defer to avoid blocking app creation)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = None
if QDRANT_URL:
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("✅ Qdrant client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Qdrant client: {e}")
        client = None
else:
    logger.warning("⚠️ QDRANT_URL not set - Qdrant client not initialized")

# Server configuration (for deployment compatibility)
# Render uses PORT environment variable, local can use QUERY_API_PORT
# Default to 8001 for local development
SERVER_PORT = int(os.getenv("PORT", os.getenv("QUERY_API_PORT", "8001")))
SERVER_HOST = os.getenv("HOST", "0.0.0.0")

# FastAPI app - MUST be created even if imports fail
app = FastAPI(
    title="Query CLI JSON API",
    description="REST API for querying GST notifications with enhanced search",
    version="1.0.0"
)

# --------------------------------------------------
# REQUEST/RESPONSE MODELS
# --------------------------------------------------

class QueryRequest(BaseModel):
    """Search query request model - simplified to only query and top_k."""
    query: str = Field(..., description="Natural language query")
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")

class ChunkResult(BaseModel):
    """Individual chunk result model."""
    chunk_index: int
    chunk_text: str
    page_no: Optional[int] = None
    score: Optional[float] = None

class ResultWithContext(BaseModel):
    """Result with surrounding context chunks."""
    score: float
    confidence: str  # HIGH, MEDIUM, LOW
    notification_no: Optional[str] = None
    tax_type: Optional[str] = None
    issued_on: Optional[str] = None
    latest_effective_date: Optional[str] = None
    document_nature: Optional[str] = None
    group_id: Optional[str] = None
    chunk_index: Optional[int] = None
    page_no: Optional[int] = None
    chunk_text: str
    context_chunks: List[ChunkResult] = []

class SearchResponse(BaseModel):
    """Search response model."""
    success: bool
    query: str
    total_results: int
    results_returned: int
    query_time_ms: float
    is_ambiguous: bool
    evaluation: Dict[str, Any]
    statistics: Dict[str, Any]
    results: List[ResultWithContext]
    message: Optional[str] = None

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def is_ambiguous(filtered_results: List) -> bool:
    """Check if results are ambiguous (similar scores)."""
    if len(filtered_results) < 2:
        return False
    score_diff = filtered_results[0].score - filtered_results[1].score
    return score_diff < 0.05  # AMBIGUITY_GAP

def calculate_statistics(results: List, query_time: float) -> Dict[str, Any]:
    """Calculate search statistics."""
    if not results:
        return {
            "total_results": 0,
            "average_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "query_time_ms": query_time * 1000
        }
    
    scores = [r.score for r in results]
    return {
        "total_results": len(results),
        "average_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "query_time_ms": query_time * 1000
    }

def get_confidence_level(score: float) -> str:
    """Get confidence level from score."""
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint - redirects to search info."""
    return {
        "name": "GST Notification Search API",
        "version": "1.0.0",
        "endpoint": "/search",
        "method": "POST",
        "payload": {
            "query": "string (required) - Natural language query",
            "top_k": "integer (optional, default: 5) - Number of results to return"
        }
    }

@app.post("/search", response_model=SearchResponse)
async def search_notifications(request: QueryRequest):
    """
    Search notifications using natural language query.
    
    Request payload:
    - query (required): Natural language query string
    - top_k (optional, default: 5): Number of results to return (1-50)
    
    Features (automatically applied):
    - Enhanced search with tax type isolation
    - Multi-query expansion
    - Hybrid search (semantic + keyword)
    - Intelligent reranking
    - Contextual chunk retrieval
    - Answer quality evaluation
    """
    start_time = time.time()
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Use default values for all other parameters
    min_score = MIN_SCORE_THRESHOLD  # Use default from config
    include_context = True
    context_before = CONTEXT_CHUNKS_BEFORE
    context_after = CONTEXT_CHUNKS_AFTER
    show_top_n = min(request.top_k, SHOW_TOP_N_RESULTS)  # Don't exceed top_k
    
    logger.info(f"Search request: query='{query[:100]}...', top_k={request.top_k}, min_score={min_score}")
    
    # Check if client is initialized
    if client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized. Check QDRANT_URL environment variable.")
    
    try:
        # No filters - enhanced search will extract tax_type and notification_no from query automatically
        qdrant_filter = None
        
        # Use enhanced search if available
        if enhanced_search:
            try:
                results = enhanced_search(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    query=query,
                    top_k=request.top_k * 3,  # Get more for filtering and reranking
                    use_hybrid=True,
                    use_multi_query=True,
                    use_reranking=True,
                    filters=qdrant_filter,
                    min_score=0.0  # We'll filter by min_score later
                )
                logger.info(f"Enhanced search returned {len(results)} results")
            except Exception as e:
                logger.warning(f"Enhanced search failed: {e}, falling back to basic search")
                # Fallback to basic search
                query_vector = embed_text(query)
                search_result = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=request.top_k,
                    query_filter=qdrant_filter,
                    with_payload=True
                )
                results = search_result.points
        else:
            # Basic search fallback
            query_vector = embed_text(query)
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=request.top_k,
                query_filter=qdrant_filter,
                with_payload=True
            )
            results = search_result.points
        
        query_time = time.time() - start_time
        
        # Filter by minimum score and sort by score (high to low)
        filtered_results = [r for r in results if r.score >= min_score]
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        if not filtered_results:
            return SearchResponse(
                success=False,
                query=query,
                total_results=0,
                results_returned=0,
                query_time_ms=query_time * 1000,
                is_ambiguous=False,
                evaluation={},
                statistics=calculate_statistics([], query_time),
                results=[],
                message=f"No results found (minimum score: {min_score:.2f})"
            )
        
        # Get top N results
        top_results = filtered_results[:show_top_n]
        
        # Evaluate answer quality
        evaluation = evaluate_answer_quality(query, filtered_results, filtered_results[0] if filtered_results else None)
        
        # Build response results with context
        response_results = []
        for result in top_results:
            payload = result.payload
            group_id = payload.get("group_id")
            chunk_index = payload.get("chunk_index")
            score = result.score
            confidence = get_confidence_level(score)
            
            # Get context chunks if requested
            context_chunks = []
            if include_context and group_id and chunk_index is not None:
                try:
                    context_chunks_data = get_surrounding_chunks(
                        client,
                        group_id,
                        chunk_index,
                        before=context_before,
                        after=context_after
                    )
                    context_chunks = [
                        ChunkResult(
                            chunk_index=chunk.payload.get("chunk_index", 0),
                            chunk_text=chunk.payload.get("chunk_text", ""),
                            page_no=chunk.payload.get("page_no"),
                            score=None  # Context chunks don't have individual scores
                        )
                        for chunk in context_chunks_data
                    ]
                except Exception as e:
                    logger.warning(f"Failed to get context chunks: {e}")
                    context_chunks = []
            
            result_data = ResultWithContext(
                score=score,
                confidence=confidence,
                notification_no=payload.get("notification_no"),
                tax_type=payload.get("tax_type"),
                issued_on=payload.get("issued_on"),
                latest_effective_date=payload.get("latest_effective_date"),
                document_nature=payload.get("document_nature"),
                group_id=group_id,
                chunk_index=chunk_index,
                page_no=payload.get("page_no"),
                chunk_text=payload.get("chunk_text", ""),
                context_chunks=context_chunks
            )
            response_results.append(result_data)
        
        # Check for ambiguity
        is_ambiguous_result = is_ambiguous(filtered_results)
        
        # Calculate statistics
        stats = calculate_statistics(filtered_results, query_time)
        
        logger.info(f"Search completed: {len(response_results)} results returned in {query_time*1000:.2f}ms")
        
        return SearchResponse(
            success=True,
            query=query,
            total_results=len(filtered_results),
            results_returned=len(response_results),
            query_time_ms=query_time * 1000,
            is_ambiguous=is_ambiguous_result,
            evaluation=evaluation,
            statistics=stats,
            results=response_results
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Query API server on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Environment: PORT={os.getenv('PORT', 'not set')}, QUERY_API_PORT={os.getenv('QUERY_API_PORT', 'not set')}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)

