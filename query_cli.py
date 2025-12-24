"""
Interactive Query CLI for RAG System
Search notifications using natural language queries with rich results display.
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ingestion.embeddings import embed_text
from dotenv import load_dotenv
import logging

# Try to import enhanced_search_v2, fallback to enhanced_search
try:
    from enhanced_search_v2 import enhanced_search
    logger = logging.getLogger(__name__)
    logger.info("Using improved enhanced_search_v2")
except ImportError:
    try:
        from enhanced_search import enhanced_search
        logger = logging.getLogger(__name__)
        logger.warning("Using original enhanced_search (enhanced_search_v2 not found)")
    except ImportError:
        enhanced_search = None
        logger = logging.getLogger(__name__)
        logger.error("No enhanced_search module found, will use basic search")

load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "notification_chunks"  # Using new chunked collection
DEFAULT_TOP_K = 5
MIN_SCORE_THRESHOLD = 0.6  # Increased threshold for better quality
AMBIGUITY_GAP = 0.05
MIN_ANSWER_CONFIDENCE = 0.5  # Minimum score for "answered"
PARTIAL_TRUTH_THRESHOLD = 0.4  # Score below which partial truth warning
CONTEXT_CHUNKS_BEFORE = 2  # Number of chunks before target
CONTEXT_CHUNKS_AFTER = 2   # Number of chunks after target
TOP_RESULTS_WITH_CONTEXT = 3  # Number of top results to show with context
SHOW_TOP_N_RESULTS = 3  # Show top N highest scoring results

# --------------------------------------------------
# ANSWER QUALITY EVALUATION
# --------------------------------------------------
def evaluate_answer_quality(query: str, results: List, top_result) -> Dict[str, Any]:
    """
    Evaluate if the question was actually answered by the retrieved documents.
    Returns evaluation with status flags.
    """
    evaluation = {
        "document_retrieved": len(results) > 0,
        "question_answered": False,
        "partial_truth": False,
        "confidence": 0.0,
        "warnings": [],
        "recommendations": []
    }
    
    if not results:
        evaluation["warnings"].append("No documents retrieved")
        return evaluation
    
    top_score = top_result.score
    evaluation["confidence"] = top_score
    
    # Check if document was retrieved
    if evaluation["document_retrieved"]:
        evaluation["warnings"].append("✅ Relevant document retrieved")
    
    # Check if question was answered (heuristic-based)
    question_answered = check_if_question_answered(query, top_result, results)
    evaluation["question_answered"] = question_answered
    
    if not question_answered:
        evaluation["warnings"].append("❌ Question not answered")
        evaluation["recommendations"].append("Try refining your query or checking more results")
        evaluation["recommendations"].append("The retrieved document may not contain the answer")
    
    # Check for partial truth (low score or incomplete information)
    if top_score < PARTIAL_TRUTH_THRESHOLD or is_partial_truth(query, top_result, results):
        evaluation["partial_truth"] = True
        evaluation["warnings"].append("⚠️ Dangerous partial truth")
        evaluation["recommendations"].append("Verify information from multiple sources")
        evaluation["recommendations"].append("Consider checking related notifications")
    
    # Additional checks
    if len(results) == 1:
        evaluation["recommendations"].append("Only one result found - consider expanding search")
    
    if top_score < MIN_ANSWER_CONFIDENCE:
        evaluation["recommendations"].append(f"Low confidence (score: {format_score(top_score)}) - verify answer")
    
    return evaluation

def check_if_question_answered(query: str, top_result, all_results: List) -> bool:
    """
    Heuristic check if the question was actually answered.
    Uses keyword matching and context analysis.
    """
    query_lower = query.lower()
    chunk_text = top_result.payload.get("chunk_text", "").lower()
    
    # Extract question words (remove common stop words)
    question_words = extract_key_terms(query_lower)
    
    # Check if chunk contains relevant information
    relevant_terms_found = sum(1 for term in question_words if term in chunk_text)
    coverage = relevant_terms_found / len(question_words) if question_words else 0
    
    # Check for answer indicators
    answer_indicators = [
        "rate", "percentage", "exempt", "exemption", "condition", "eligible",
        "required", "must", "shall", "applicable", "rate of", "gst rate"
    ]
    
    has_answer_indicators = any(indicator in chunk_text for indicator in answer_indicators)
    
    # Check score threshold
    score_ok = top_result.score >= MIN_ANSWER_CONFIDENCE
    
    # Question is answered if:
    # 1. Good coverage of question terms AND
    # 2. Contains answer indicators AND
    # 3. Good similarity score
    answered = (coverage >= 0.5) and (has_answer_indicators or score_ok) and score_ok
    
    return answered

def is_partial_truth(query: str, top_result, all_results: List) -> bool:
    """
    Detect if the result might be a partial or misleading truth.
    """
    # Low score suggests partial relevance
    if top_result.score < PARTIAL_TRUTH_THRESHOLD:
        return True
    
    # Check if chunk is very short (might be incomplete)
    chunk_text = top_result.payload.get("chunk_text", "")
    if len(chunk_text) < 200:
        return True
    
    # Check for incomplete statements
    incomplete_indicators = [
        "...", "etc.", "and so on", "refer to", "see above", "see below",
        "as mentioned", "as stated", "refer", "see notification"
    ]
    
    has_incomplete = any(indicator in chunk_text.lower() for indicator in incomplete_indicators)
    
    # Check if multiple results have very different scores (suggests uncertainty)
    if len(all_results) >= 2:
        score_diff = all_results[0].score - all_results[1].score
        if score_diff < AMBIGUITY_GAP:
            return True  # Ambiguous results suggest partial truth
    
    return has_incomplete

def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from query, removing common stop words."""
    stop_words = {
        "what", "is", "the", "for", "of", "and", "or", "but", "in", "on", "at",
        "to", "a", "an", "as", "are", "was", "were", "been", "be", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "this", "that", "these", "those", "how", "when", "where",
        "why", "which", "who", "whom", "whose", "gst", "tax"
    }
    
    # Simple extraction - split and filter
    words = text.lower().split()
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]
    
    return key_terms

def print_evaluation(evaluation: Dict[str, Any]):
    """Print answer quality evaluation."""
    print_separator("=")
    print("📊 Answer Quality Evaluation")
    print_separator("=")
    
    # Status indicators
    for warning in evaluation["warnings"]:
        print(f"  {warning}")
    
    print()
    
    # Confidence
    confidence_level = "High" if evaluation["confidence"] >= 0.7 else "Medium" if evaluation["confidence"] >= 0.5 else "Low"
    print(f"🎯 Confidence: {confidence_level} ({format_score(evaluation['confidence'])})")
    
    # Recommendations
    if evaluation["recommendations"]:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(evaluation["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    print_separator("=")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def format_score(score: float) -> str:
    """Format similarity score as percentage."""
    return f"{score * 100:.1f}%"

def format_text(text: str, max_length: int = 500) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)

def print_header(text: str):
    """Print a formatted header."""
    print_separator()
    print(f"  {text}")
    print_separator()

def print_result(result, index: int, show_full_text: bool = False):
    """Print a single search result."""
    payload = result.payload
    score = result.score
    
    print(f"\n{'─' * 80}")
    print(f"📄 Result #{index + 1} (Score: {format_score(score)})")
    print(f"{'─' * 80}")
    
    # Metadata
    if payload.get("tax_type"):
        print(f"📋 Tax Type: {payload['tax_type']}")
    if payload.get("notification_no"):
        print(f"🔢 Notification: {payload['notification_no']}")
    if payload.get("issued_on"):
        print(f"📅 Issued On: {payload['issued_on']}")
    if payload.get("latest_effective_date"):
        print(f"✅ Effective Date: {payload['latest_effective_date']}")
    if payload.get("document_nature"):
        print(f"📝 Nature: {payload['document_nature'].upper()}")
    if payload.get("page_no"):
        print(f"📄 Page: {payload['page_no']}")
    if payload.get("chunk_index") is not None:
        print(f"🔖 Chunk: {payload['chunk_index'] + 1}/{payload.get('total_chunks', '?')}")
    if payload.get("file_path"):
        print(f"📁 File: {Path(payload['file_path']).name}")
    
    # Chunk text
    chunk_text = payload.get("chunk_text", "")
    if chunk_text:
        print(f"\n📝 Content:")
        if show_full_text:
            print(f"   {chunk_text}")
        else:
            print(f"   {format_text(chunk_text, 400)}")
            if len(chunk_text) > 400:
                print(f"   ... (truncated, full length: {len(chunk_text)} chars)")

def print_result_with_context(
    result,
    context_chunks: List,
    index: int,
    show_full_text: bool = False
):
    """Print a result with surrounding context chunks."""
    payload = result.payload
    score = result.score
    target_chunk_index = payload.get("chunk_index", 0)
    
    # Determine confidence level
    if score >= 0.7:
        confidence = "HIGH"
        conf_emoji = "🟢"
    elif score >= 0.5:
        confidence = "MEDIUM"
        conf_emoji = "🟡"
    else:
        confidence = "LOW"
        conf_emoji = "🔴"
    
    print(f"\n{'═' * 80}")
    print(f"{conf_emoji} Result #{index + 1} - Confidence: {confidence} (Score: {format_score(score)})")
    print(f"{'═' * 80}")
    
    # Metadata
    if payload.get("tax_type"):
        print(f"📋 Tax Type: {payload['tax_type']}")
    if payload.get("notification_no"):
        print(f"🔢 Notification: {payload['notification_no']}")
    if payload.get("issued_on"):
        print(f"📅 Issued On: {payload['issued_on']}")
    if payload.get("latest_effective_date"):
        print(f"✅ Effective Date: {payload['latest_effective_date']}")
    if payload.get("document_nature"):
        print(f"📝 Nature: {payload['document_nature'].upper()}")
    if payload.get("file_path"):
        print(f"📁 File: {Path(payload['file_path']).name}")
    
    # Combined context (2 before + target + 2 after)
    print(f"\n📚 Contextual Content ({len(context_chunks)} chunks):")
    print(f"{'─' * 80}")
    
    for i, chunk_point in enumerate(context_chunks):
        chunk_payload = chunk_point.payload
        chunk_idx = chunk_payload.get("chunk_index", 0)
        chunk_text = chunk_payload.get("chunk_text", "")
        
        # Mark the target chunk
        if chunk_idx == target_chunk_index:
            marker = ">>> TARGET CHUNK <<<"
            prefix = "🎯"
        else:
            marker = f"Context chunk {i + 1}"
            if chunk_idx < target_chunk_index:
                prefix = "⬆️"
            else:
                prefix = "⬇️"
        
        print(f"\n{prefix} {marker} (Chunk #{chunk_idx + 1})")
        print(f"{'·' * 80}")
        
        if chunk_text:
            if show_full_text:
                print(f"{chunk_text}")
            else:
                # Show more text for context chunks
                print(f"{format_text(chunk_text, 600)}")
                if len(chunk_text) > 600:
                    print(f"... (truncated, full length: {len(chunk_text)} chars)")
        
        # Show page info if available
        if chunk_payload.get("page_no"):
            print(f"   [Page {chunk_payload['page_no']}]")
    
    print(f"{'─' * 80}")

# Cache to track if indexes are available (to avoid repeated 400 errors)
_index_availability_cache = {"group_id": None, "chunk_index": None}

def get_surrounding_chunks(
    client: QdrantClient,
    group_id: str,
    chunk_index: int,
    before: int = 2,
    after: int = 2
) -> List[Any]:
    """
    Get surrounding chunks from the same document.
    Returns chunks with indices from (chunk_index - before) to (chunk_index + after).
    Uses Python-side filtering if Qdrant indexes are not available.
    """
    # Build filter for same group_id and chunk_index range
    min_index = max(0, chunk_index - before)
    max_index = chunk_index + after
    
    # Check cache - if we know indexes aren't available, skip filter attempt
    if _index_availability_cache.get("group_id") is False:
        # Indexes not available, go straight to Python-side filtering
        return _get_surrounding_chunks_python_filter(
            client, group_id, min_index, max_index, before, after
        )
    
    try:
        # Try to use Range filter with group_id filter (requires indexes)
        from qdrant_client.models import Range
        
        conditions = [
            FieldCondition(key="group_id", match=MatchValue(value=group_id)),
            FieldCondition(
                key="chunk_index",
                range=Range(gte=min_index, lte=max_index)
            )
        ]
        
        filter_query = Filter(must=conditions)
        
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_query,
            limit=100,
            with_payload=True
        )
        
        chunks = scroll_result[0]
        chunks.sort(key=lambda x: x.payload.get("chunk_index", 0))
        
        # Cache that indexes are available
        _index_availability_cache["group_id"] = True
        _index_availability_cache["chunk_index"] = True
        
        return chunks
        
    except Exception as filter_error:
        # Check if this is an index-related error (400 Bad Request)
        error_str = str(filter_error).lower()
        is_index_error = (
            "index required" in error_str or 
            "400" in error_str or 
            "bad request" in error_str or
            "index" in error_str
        )
        
        if is_index_error:
            # Cache that indexes are not available to avoid future attempts
            _index_availability_cache["group_id"] = False
            _index_availability_cache["chunk_index"] = False
            
            # Fallback to Python-side filtering (silently, no error logging)
            return _get_surrounding_chunks_python_filter(
                client, group_id, min_index, max_index, before, after
            )
        else:
            # Some other error - log and return empty
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error fetching surrounding chunks: {filter_error}")
            return []

def _get_surrounding_chunks_python_filter(
    client: QdrantClient,
    group_id: str,
    min_index: int,
    max_index: int,
    before: int,
    after: int
) -> List[Any]:
    """
    Fallback method: Scroll through collection and filter in Python.
    This is less efficient but works without Qdrant indexes.
    """
    try:
        all_matching_chunks = []
        offset = None
        
        # Scroll in batches to find matching chunks
        for batch_num in range(20):  # Limit to 20 batches (20k chunks max)
            scroll_result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=True
            )
            
            # Qdrant scroll returns (points, next_offset)
            points, next_offset = scroll_result
            if not points:
                break
            
            # Filter by group_id and chunk_index range in Python
            for chunk in points:
                payload = chunk.payload
                chunk_idx = payload.get("chunk_index", -1)
                if (payload.get("group_id") == group_id and
                    min_index <= chunk_idx <= max_index):
                    all_matching_chunks.append(chunk)
            
            # Early exit if we found enough chunks
            if len(all_matching_chunks) >= (before + after + 1):
                break
            
            if next_offset is None:
                break
            offset = next_offset
        
        # Sort by chunk_index
        all_matching_chunks.sort(key=lambda x: x.payload.get("chunk_index", 0))
        return all_matching_chunks
        
    except Exception as e:
        # Log warning but don't crash
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Python-side filtering failed: {e}")
        return []

def build_filter(
    tax_type: Optional[str] = None,
    notification_no: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    document_nature: Optional[str] = None
) -> Optional[Filter]:
    """
    Build Qdrant filter from parameters.
    Note: Enhanced search will automatically extract tax_type and notification_no
    from the query and apply filters, so manual filters are mainly for additional constraints.
    """
    conditions = []
    
    if tax_type:
        # Support tax type variants (with/without "Rate" suffix)
        from qdrant_client.models import Should
        tax_variants = [
            FieldCondition(key="tax_type", match=MatchValue(value=tax_type)),
            FieldCondition(key="tax_type", match=MatchValue(value=f"{tax_type} (Rate)"))
        ]
        conditions.append(Should(should=tax_variants))
    
    if notification_no:
        # Support notification number variants (with/without leading zeros)
        # Normalize notification number
        import re
        notif_match = re.search(r'(\d+)[/\-](\d{4})', notification_no)
        if notif_match:
            num = notif_match.group(1)
            year = notif_match.group(2)
            num_int = int(num)
            # Create variants
            variants = [
                f"{num}/{year}",  # Original
                f"{num_int}/{year}",  # Without leading zeros
                f"{str(num_int).zfill(2)}/{year}" if num_int < 10 else f"{num_int}/{year}"  # With leading zero
            ]
            from qdrant_client.models import Should
            notif_conditions = [
                FieldCondition(key="notification_no", match=MatchValue(value=variant))
                for variant in variants
            ]
            conditions.append(Should(should=notif_conditions))
        else:
            # Fallback to exact match
            conditions.append(
                FieldCondition(key="notification_no", match=MatchValue(value=notification_no))
            )
    
    if document_nature:
        conditions.append(
            FieldCondition(key="document_nature", match=MatchValue(value=document_nature))
        )
    
    if date_from:
        # Note: Date filtering would need proper date comparison
        # This is a simplified version
        pass
    
    if date_to:
        pass
    
    if not conditions:
        return None
    
    return Filter(must=conditions)

def show_statistics(results: List, query_time: float):
    """Display search statistics."""
    if not results:
        return
    
    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    
    print_separator("-")
    print("📊 Search Statistics:")
    print(f"   • Total results: {len(results)}")
    print(f"   • Average score: {format_score(avg_score)}")
    print(f"   • Highest score: {format_score(max_score)}")
    print(f"   • Lowest score: {format_score(min_score)}")
    print(f"   • Query time: {query_time:.3f} seconds")
    print_separator("-")

def export_results(results: List, query: str, output_file: str = None):
    """Export search results to JSON file."""
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"query_results_{timestamp}.json"
    
    export_data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "total_results": len(results),
        "results": [
            {
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]
    }
    
    output_path = Path(output_file)
    output_path.write_text(
        json.dumps(export_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"✅ Results exported to: {output_path.resolve()}")

def show_help():
    """Display help information."""
    print_header("📖 Query CLI Help")
    print("""
Commands:
  /help, /h          - Show this help message
  /exit, /quit, /q   - Exit the CLI
  /stats             - Show collection statistics
  /filter            - Set search filters (tax_type, date, etc.)
  /clear             - Clear filters
  /top <N>           - Set number of results (default: 5)
  /full              - Toggle full text display
  /export [filename] - Export results to JSON file
  /score <threshold> - Set minimum score threshold (0.0-1.0)
  /eval              - Show answer quality evaluation (auto-shown)

Filter Options:
  /filter tax_type=<value>
  /filter notification_no=<value>
  /filter nature=<value> (original/corrigendum/amendment)

Enhanced Search Features:
  ✅ Tax type isolation - searches within correct tax type only
  ✅ Exact notification matching - prevents cross-year matches
  ✅ Multi-query expansion - generates query variations
  ✅ Hybrid search - combines semantic + keyword matching
  ✅ Intelligent reranking - boosts relevant results
  ✅ Intent validation - filters irrelevant results

Examples:
  /top 10            - Show top 10 results
  /filter tax_type=Central Tax
  /score 0.5         - Only show results with score >= 0.5
  /export my_results.json
  
Query Examples:
  "Central Tax notification 01/2017" - Only Central Tax, exact notification
  "GSTR-9 Table 5D" - Finds exact form and table references
  "SAC code for services" - Keyword + semantic search
    """)

def show_collection_stats(client: QdrantClient):
    """Show collection statistics."""
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print_header("📊 Collection Statistics")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total points: {collection_info.points_count:,}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
    except Exception as e:
        print(f"❌ Error getting collection stats: {e}")

# --------------------------------------------------
# MAIN CLI
# --------------------------------------------------
def run_query_cli():
    """Run the interactive query CLI."""
    if not QDRANT_URL:
        print("❌ Error: QDRANT_URL not set in environment variables")
        return
    
    # Initialize client
    print("🔗 Connecting to Qdrant...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Test connection
        client.get_collections()
        print("✅ Connected successfully\n")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return
    
    # CLI state
    top_k = DEFAULT_TOP_K
    min_score = MIN_SCORE_THRESHOLD
    show_full = False
    current_filter = None
    last_results = []
    last_query = ""
    
    # Welcome message
    print_header("🔍 Notification Query CLI")
    print("Enter your query or type /help for commands")
    print("Type /exit to quit\n")
    
    while True:
        try:
            # Get user input
            user_input = input("❓ Query: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split()
                cmd = cmd_parts[0].lower() if cmd_parts else ""
                args = cmd_parts[1:] if len(cmd_parts) > 1 else []
                
                if cmd in ["exit", "quit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                elif cmd in ["help", "h"]:
                    show_help()
                
                elif cmd == "stats":
                    show_collection_stats(client)
                
                elif cmd == "top":
                    if args and args[0].isdigit():
                        top_k = int(args[0])
                        print(f"✅ Top K set to: {top_k}")
                    else:
                        print(f"Current Top K: {top_k}")
                
                elif cmd == "score":
                    if args:
                        try:
                            min_score = float(args[0])
                            if 0 <= min_score <= 1:
                                print(f"✅ Minimum score threshold: {min_score}")
                            else:
                                print("❌ Score must be between 0.0 and 1.0")
                        except ValueError:
                            print("❌ Invalid score value")
                    else:
                        print(f"Current minimum score: {min_score}")
                
                elif cmd == "full":
                    show_full = not show_full
                    print(f"✅ Full text display: {'ON' if show_full else 'OFF'}")
                
                elif cmd == "filter":
                    if args:
                        # Simple filter parsing
                        filter_str = " ".join(args)
                        if "=" in filter_str:
                            key, value = filter_str.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if key == "tax_type":
                                current_filter = build_filter(tax_type=value)
                                print(f"✅ Filter: tax_type = {value}")
                            elif key == "notification_no":
                                current_filter = build_filter(notification_no=value)
                                print(f"✅ Filter: notification_no = {value}")
                            elif key == "nature":
                                current_filter = build_filter(document_nature=value)
                                print(f"✅ Filter: document_nature = {value}")
                            else:
                                print(f"❌ Unknown filter key: {key}")
                        else:
                            print("❌ Filter format: /filter key=value")
                    else:
                        print("Current filter: " + (str(current_filter) if current_filter else "None"))
                
                elif cmd == "clear":
                    current_filter = None
                    print("✅ Filters cleared")
                
                elif cmd == "export":
                    if not last_results:
                        print("❌ No results to export. Run a query first.")
                    else:
                        filename = args[0] if args else None
                        export_results(last_results, last_query, filename)
                
                else:
                    print(f"❌ Unknown command: {cmd}. Type /help for help.")
                
                print()  # Empty line after command
                continue
            
            # Process query
            print(f"\n🔍 Searching for: '{user_input}'...")
            start_time = time.time()
            
            # Use enhanced search if available, otherwise fallback to basic search
            if enhanced_search:
                try:
                    # Use enhanced search with all features enabled
                    results = enhanced_search(
                        client=client,
                        collection_name=COLLECTION_NAME,
                        query=user_input,
                        top_k=top_k * 3,  # Get more for filtering and reranking
                        use_hybrid=True,
                        use_multi_query=True,
                        use_reranking=True,
                        filters=current_filter,
                        min_score=0.0  # We'll filter by min_score later
                    )
                    print(f"✅ Enhanced search completed (found {len(results)} results)")
                except Exception as e:
                    print(f"⚠️  Enhanced search failed: {e}, falling back to basic search")
                    # Fallback to basic search
                    try:
                        query_vector = embed_text(user_input)
                        search_result = client.query_points(
                            collection_name=COLLECTION_NAME,
                            query=query_vector,
                            limit=top_k,
                            query_filter=current_filter,
                            with_payload=True
                        )
                        results = search_result.points
                    except Exception as e2:
                        print(f"❌ Basic search also failed: {e2}")
                        continue
            else:
                # Basic search fallback
                try:
                    query_vector = embed_text(user_input)
                    search_result = client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector,
                        limit=top_k,
                        query_filter=current_filter,
                        with_payload=True
                    )
                    results = search_result.points
                except Exception as e:
                    print(f"❌ Search failed: {e}")
                    continue
            
            query_time = time.time() - start_time
            
            # Filter by minimum score and sort by score (high to low)
            filtered_results = [r for r in results if r.score >= min_score]
            filtered_results.sort(key=lambda x: x.score, reverse=True)  # Sort by confidence (high to low)
            
            if not filtered_results:
                print(f"\n❌ No results found (minimum score: {min_score:.2f})")
                if results:
                    print(f"   Found {len(results)} results below threshold")
                    print(f"   Highest score: {format_score(results[0].score)}")
                print()
                continue
            
            # Get top N highest scoring results (default: 3)
            top_results = filtered_results[:SHOW_TOP_N_RESULTS]
            
            # Store for export
            last_results = filtered_results
            last_query = user_input
            
            # Evaluate answer quality
            evaluation = evaluate_answer_quality(user_input, filtered_results, filtered_results[0] if filtered_results else None)
            
            # Display results
            print_header(f"📋 Top {len(top_results)} Highest Scoring Results (Sorted by Confidence: High to Low)")
            
            # Show answer quality evaluation
            print_evaluation(evaluation)
            print()
            
            # Check for ambiguity
            if len(filtered_results) >= 2:
                score_diff = filtered_results[0].score - filtered_results[1].score
                if score_diff < AMBIGUITY_GAP:
                    print("⚠️  Warning: Multiple similar results found (ambiguous query)")
                    print()
            
            # Show top results with context
            for idx, result in enumerate(top_results, 1):
                payload = result.payload
                group_id = payload.get("group_id")
                chunk_index = payload.get("chunk_index")
                
                if group_id and chunk_index is not None:
                    # Get surrounding chunks
                    print(f"\n🔍 Fetching context for result #{idx}...")
                    context_chunks = get_surrounding_chunks(
                        client,
                        group_id,
                        chunk_index,
                        before=CONTEXT_CHUNKS_BEFORE,
                        after=CONTEXT_CHUNKS_AFTER
                    )
                    
                    # If we got context chunks, show with context
                    if context_chunks:
                        print_result_with_context(result, context_chunks, idx, show_full)
                    else:
                        # Fallback to regular display if context unavailable
                        print(f"⚠️  Context unavailable, showing single chunk:")
                        print_result(result, idx, show_full)
                else:
                    # Fallback if metadata missing
                    print(f"⚠️  Missing metadata for context, showing single chunk:")
                    print_result(result, idx, show_full)
            
            # Show statistics
            show_statistics(filtered_results, query_time)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    run_query_cli()
