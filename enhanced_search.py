"""
Enhanced Search Methods for RAG System
Implements multiple retrieval strategies to improve search accuracy.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ingestion.embeddings import embed_text, embed_batch

logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MAX_QUERY_EXPANSIONS = 3
KEYWORD_BOOST_FACTOR = 1.2
RECENCY_BOOST_FACTOR = 1.1
EXACT_MATCH_BOOST = 1.5

# --------------------------------------------------
# QUERY EXPANSION & REWRITING
# --------------------------------------------------
def expand_query(query: str) -> List[str]:
    """
    Generate query variations for better recall, with GST-specific patterns.
    """
    expansions = [query]  # Original query
    
    query_lower = query.lower()
    
    # GST-specific expansions first
    # Expand Schedule references (I, II, III, IV, V, VI)
    schedule_map = {
        'i': '1', '1': 'i', 'ii': '2', '2': 'ii',
        'iii': '3', '3': 'iii', 'iv': '4', '4': 'iv',
        'v': '5', '5': 'v', 'vi': '6', '6': 'vi'
    }
    for roman, num in schedule_map.items():
        if f'schedule {roman}' in query_lower:
            expanded = query.replace(f'Schedule {roman.upper()}', f'Schedule {num}').replace(f'schedule {roman}', f'schedule {num}')
            if expanded not in expansions:
                expansions.append(expanded)
        elif f'schedule {num}' in query_lower:
            expanded = query.replace(f'Schedule {num}', f'Schedule {roman.upper()}').replace(f'schedule {num}', f'schedule {roman}')
            if expanded not in expansions:
                expansions.append(expanded)
    
    # 1. Abbreviation expansion
    abbreviations = {
        "gst": "goods and services tax",
        "cgst": "central goods and services tax",
        "igst": "integrated goods and services tax",
        "sgst": "state goods and services tax",
        "utgst": "union territory goods and services tax",
        "notif": "notification",
        "notfn": "notification"
    }
    
    expanded = query
    for abbr, full in abbreviations.items():
        if abbr in query_lower:
            expanded = expanded.replace(abbr, full)
            expanded = expanded.replace(abbr.upper(), full)
            expanded = expanded.replace(abbr.capitalize(), full)
    
    if expanded != query:
        expansions.append(expanded)
    
    # 2. Remove date qualifiers for broader search
    date_pattern = r'\s+as\s+(on|of)\s+\w+\s+\d{4}'
    without_date = re.sub(date_pattern, '', query, flags=re.IGNORECASE)
    if without_date != query:
        expansions.append(without_date.strip())
    
    # 3. Extract key terms (remove stop words)
    stop_words = {
        "what", "is", "the", "for", "of", "and", "or", "but", "in", "on", "at",
        "to", "a", "an", "as", "are", "was", "were", "been", "be", "have", "has",
        "which", "who", "when", "where", "why", "how", "please", "tell", "me"
    }
    
    words = query_lower.split()
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]
    if len(key_terms) >= 3 and len(key_terms) < len(words):
        expansions.append(" ".join(key_terms))
    
    # 4. Add question format if missing
    if not any(query_lower.startswith(q) for q in ["what", "which", "who", "when", "where", "how"]):
        expansions.append(f"What is {query}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expansions = []
    for exp in expansions:
        exp_lower = exp.lower().strip()
        if exp_lower and exp_lower not in seen:
            seen.add(exp_lower)
            unique_expansions.append(exp)
    
    return unique_expansions[:MAX_QUERY_EXPANSIONS]

def extract_keywords(query: str) -> List[str]:
    """
    Extract important keywords from query, preserving exact phrases and identifiers.
    Examples:
    - "Table 5D of GSTR-9 for FY 2024-25, and what SAC" 
      -> ["table 5d", "gstr-9", "fy 2024-25", "sac"]
    """
    keywords = []
    query_lower = query.lower()
    
    # Stop words to filter out
    stop_words = {
        "what", "is", "the", "for", "of", "and", "or", "but", "in", "on", "at",
        "to", "a", "an", "as", "are", "was", "were", "been", "be", "have", "has",
        "which", "who", "when", "where", "why", "how", "please", "tell", "me",
        "whether", "that", "this", "these", "those"
    }
    
    # 1. Extract Table references (Table 5D, Table 5, table 5d, etc.)
    table_pattern = r'\btable\s+(\d+[a-z]?)'
    table_matches = re.findall(table_pattern, query_lower)
    for match in table_matches:
        keywords.append(f"table {match}")
        # Also add without "table" prefix
        keywords.append(match)
    
    # 2. Extract GSTR form references (GSTR-9, GSTR-3B, gstr-9, etc.)
    gstr_pattern = r'\bgstr[-\s]?(\d+[a-z]?)\b'
    gstr_matches = re.findall(gstr_pattern, query_lower)
    for match in gstr_matches:
        keywords.append(f"gstr-{match}")
        keywords.append(f"gstr {match}")
    
    # 3. Extract code references (SAC, HSN, SAC code, HSN code, etc.)
    code_pattern = r'\b(sac|hsn)(?:\s+code)?\b'
    code_matches = re.findall(code_pattern, query_lower)
    keywords.extend(code_matches)
    
    # 4. Extract Financial Year (FY 2024-25, FY2024-25, 2024-25, etc.)
    fy_pattern = r'\bfy\s*(\d{4}[-/]\d{2,4})'
    fy_matches = re.findall(fy_pattern, query_lower)
    for match in fy_matches:
        keywords.append(f"fy {match}")
        keywords.append(match)
    
    # Also catch standalone date ranges (2024-25)
    date_range_pattern = r'\b(\d{4}[-/]\d{2,4})\b'
    date_ranges = re.findall(date_range_pattern, query_lower)
    for dr in date_ranges:
        if dr not in [f for f in fy_matches]:  # Avoid duplicates
            keywords.append(dr)
    
    # 5. Extract Notification numbers (12/2024, 12-2024, etc.)
    notif_pattern = r'\b(\d+)[/\-](\d{4})\b'
    notif_matches = re.findall(notif_pattern, query_lower)
    for match in notif_matches:
        keywords.append(f"{match[0]}/{match[1]}")
        keywords.append(f"{match[0]}-{match[1]}")
    
    # 6. Extract Rule/Section references (Rule 5, Section 9, etc.)
    rule_pattern = r'\b(rule|section|sec)\s+(\d+[a-z]?)\b'
    rule_matches = re.findall(rule_pattern, query_lower)
    for match in rule_matches:
        keywords.append(f"{match[0]} {match[1]}")
        keywords.append(match[1])
    
    # 7. Extract Schedule references (Schedule I, Schedule II, etc.)
    schedule_pattern = r'\bschedule\s+([ivx]+|[1-6])\b'
    schedule_matches = re.findall(schedule_pattern, query_lower)
    for match in schedule_matches:
        keywords.append(f"schedule {match}")
        keywords.append(f"schedule {match.upper()}")
    
    # 8. Extract HSN/Tariff codes (4-8 digit codes)
    hsn_pattern = r'\b\d{4,8}\s*\b'
    hsn_matches = re.findall(hsn_pattern, query)
    for match in hsn_matches:
        code = match.strip()
        if len(code) >= 4 and len(code) <= 8:
            keywords.append(code)
    
    # 9. Extract serial numbers (S. No., S.No., Serial No, etc.)
    serial_pattern = r'\b(?:s\.?\s*no\.?|serial\s+no\.?)\s*:?\s*(\d+[a-z]?)\b'
    serial_matches = re.findall(serial_pattern, query_lower)
    for match in serial_matches:
        keywords.append(f"s. no. {match}")
        keywords.append(f"serial no {match}")
        keywords.append(match)
    
    # 10. Extract remaining important words (excluding stop words and already extracted phrases)
    # Create a copy of query to remove already matched phrases
    remaining_query = query_lower
    # Remove already matched patterns
    remaining_query = re.sub(table_pattern, '', remaining_query)
    remaining_query = re.sub(gstr_pattern, '', remaining_query)
    remaining_query = re.sub(code_pattern, '', remaining_query)
    remaining_query = re.sub(fy_pattern, '', remaining_query)
    remaining_query = re.sub(date_range_pattern, '', remaining_query)
    remaining_query = re.sub(notif_pattern, '', remaining_query)
    remaining_query = re.sub(rule_pattern, '', remaining_query)
    remaining_query = re.sub(schedule_pattern, '', remaining_query)
    remaining_query = re.sub(serial_pattern, '', remaining_query)
    
    # Extract words from remaining query
    words = re.findall(r'\b\w+\b', remaining_query)
    for word in words:
        if word not in stop_words and len(word) > 2:
            # Skip if it's part of an already extracted phrase
            skip = False
            for keyword in keywords:
                if word in keyword:
                    skip = True
                    break
            if not skip:
                keywords.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower().strip()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw_lower)
    
    return unique_keywords

# --------------------------------------------------
# HYBRID SEARCH (SEMANTIC + KEYWORD)
# --------------------------------------------------
def calculate_keyword_score(chunk_text: str, keywords: List[str]) -> float:
    """
    Calculate keyword matching score.
    Uses exact phrase matching for multi-word keywords (e.g., "table 5d", "gstr-9").
    This ensures phrases like "table 5d" match exactly, not just individual words.
    """
    if not keywords:
        return 0.0
    
    chunk_lower = chunk_text.lower()
    matches = sum(1 for keyword in keywords if keyword in chunk_lower)
    
    # Normalize score (0-1)
    return matches / len(keywords) if keywords else 0.0

def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    query_vector: List[float],
    top_k: int = 10,
    filters: Optional[Filter] = None
) -> List[Any]:
    """
    Perform hybrid search combining semantic similarity and keyword matching.
    """
    # 1. Semantic search
    try:
        semantic_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k * 2,  # Get more for reranking
            query_filter=filters,
            with_payload=True
        ).points
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []
    
    if not semantic_results:
        return []
    
    # 2. Extract keywords
    keywords = extract_keywords(query)
    
    # 3. Calculate hybrid scores
    hybrid_results = []
    for result in semantic_results:
        chunk_text = result.payload.get("chunk_text", "").lower()
        
        # Semantic score (already normalized)
        semantic_score = result.score
        
        # Keyword score
        keyword_score = calculate_keyword_score(chunk_text, keywords)
        
        # Combine scores (weighted average)
        # Semantic: 70%, Keyword: 30%
        hybrid_score = (0.7 * semantic_score) + (0.3 * keyword_score)
        
        # Boost for exact keyword matches
        query_terms = set(query.lower().split())
        chunk_terms = set(re.findall(r'\b\w+\b', chunk_text))
        exact_matches = len(query_terms.intersection(chunk_terms))
        if exact_matches > 0:
            hybrid_score *= (1 + 0.1 * exact_matches)
        
        # Create result with hybrid score
        hybrid_results.append({
            "point": result,
            "hybrid_score": hybrid_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score
        })
    
    # 4. Sort by hybrid score
    hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # Return top K with hybrid scores
    return hybrid_results[:top_k]

# --------------------------------------------------
# MULTI-QUERY SEARCH WITH FUSION
# --------------------------------------------------
def multi_query_search(
    client: QdrantClient,
    collection_name: str,
    original_query: str,
    top_k: int = 10,
    filters: Optional[Filter] = None
) -> List[Any]:
    """
    Search using multiple query variations and fuse results (Reciprocal Rank Fusion).
    """
    # Generate query expansions
    query_variations = expand_query(original_query)
    
    logger.info(f"Searching with {len(query_variations)} query variations")
    
    # Search with each variation
    all_results = {}  # point_id -> {point, scores: [], rank: float}
    
    for variation in query_variations:
        try:
            vector = embed_text(variation)
            results = client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=top_k * 2,
                query_filter=filters,
                with_payload=True
            ).points
            
            # Reciprocal Rank Fusion (RRF)
            for rank, result in enumerate(results, 1):
                # Create unique ID from payload (group_id + chunk_index)
                payload = result.payload
                group_id = payload.get("group_id", "")
                chunk_idx = payload.get("chunk_index", 0)
                point_id = f"{group_id}__{chunk_idx}"
                
                rrf_score = 1.0 / (60 + rank)  # RRF formula
                
                if point_id not in all_results:
                    all_results[point_id] = {
                        "point": result,
                        "rrf_score": 0.0,
                        "scores": [],
                        "original_rank": rank
                    }
                
                all_results[point_id]["rrf_score"] += rrf_score
                all_results[point_id]["scores"].append(result.score)
        
        except Exception as e:
            logger.warning(f"Query variation failed: {variation[:50]}... - {e}")
            continue
    
    # Convert to list and sort by RRF score
    fused_results = list(all_results.values())
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    # Update original result scores to average of all scores
    final_results = []
    for item in fused_results[:top_k]:
        result = item["point"]
        # Use average of all semantic scores
        avg_score = sum(item["scores"]) / len(item["scores"]) if item["scores"] else result.score
        # Combine with RRF score (normalize RRF to 0-1 range)
        # RRF score is already accumulated, so normalize by number of variations
        max_possible_rrf = sum(1.0 / (60 + i) for i in range(1, len(query_variations) + 1))
        normalized_rrf = min(item["rrf_score"] / max_possible_rrf if max_possible_rrf > 0 else 0, 1.0)
        final_score = (0.7 * avg_score) + (0.3 * normalized_rrf)
        
        # Update the original result's score
        result.score = final_score
        final_results.append(result)
    
    return final_results

# --------------------------------------------------
# RERANKING
# --------------------------------------------------
def calculate_rerank_score(
    result: Any,
    query: str,
    query_keywords: List[str],
    base_score: float
) -> float:
    """
    Calculate reranking score based on multiple factors.
    """
    payload = result.payload
    chunk_text = payload.get("chunk_text", "").lower()
    query_lower = query.lower()
    
    rerank_score = base_score
    
    # 1. Exact phrase matching boost
    if query_lower in chunk_text:
        rerank_score *= EXACT_MATCH_BOOST
    
    # 2. Keyword density boost
    keyword_matches = sum(1 for kw in query_keywords if kw in chunk_text)
    if keyword_matches > 0:
        density = keyword_matches / len(query_keywords) if query_keywords else 0
        rerank_score *= (1 + 0.2 * density)
    
    # 3. Recency boost (if date available)
    issued_on = payload.get("issued_on") or payload.get("latest_effective_date")
    if issued_on:
        try:
            from datetime import datetime
            date_obj = datetime.fromisoformat(issued_on)
            # Boost recent documents (within last 3 years gets small boost)
            years_ago = (datetime.now() - date_obj).days / 365
            if years_ago < 3:
                rerank_score *= RECENCY_BOOST_FACTOR
        except:
            pass
    
    # 4. Notification number exact match boost
    notification_no = payload.get("notification_no", "")
    # Check if query mentions notification number
    notif_pattern = r'(\d+)[/\-](\d{4})'
    query_notif = re.search(notif_pattern, query, re.IGNORECASE)
    if query_notif and notification_no:
        query_notif_str = f"{query_notif.group(1)}/{query_notif.group(2)}"
        if query_notif_str.lower() in notification_no.lower():
            rerank_score *= EXACT_MATCH_BOOST
    
    # 5. Tax type match boost
    tax_type = payload.get("tax_type", "").lower()
    if "central tax" in query_lower or "cgst" in query_lower:
        if "central" in tax_type:
            rerank_score *= 1.1
    if "integrated tax" in query_lower or "igst" in query_lower:
        if "integrated" in tax_type:
            rerank_score *= 1.1
    
    # 6. Schedule match boost (Schedule I, II, etc.)
    schedule_pattern = r'\bschedule\s+([ivx]+|[1-6])\b'
    query_schedule_match = re.search(schedule_pattern, query_lower)
    if query_schedule_match:
        query_schedule = query_schedule_match.group(1).lower()
        chunk_lower = chunk_text.lower()
        # Check for schedule in chunk
        if f'schedule {query_schedule}' in chunk_lower or f'schedule {query_schedule.upper()}' in chunk_lower:
            rerank_score *= 1.15
    
    # 7. Form/Table match boost (GSTR-9, Table 5D, etc.)
    form_pattern = r'\bgstr[-\s]?(\d+[a-z]?)\b'
    form_match = re.search(form_pattern, query_lower)
    if form_match:
        form_num = form_match.group(1)
        if f'gstr-{form_num}' in chunk_text.lower() or f'gstr {form_num}' in chunk_text.lower() or f'form gstr-{form_num}' in chunk_text.lower():
            rerank_score *= 1.2
    
    table_pattern = r'\btable\s+(\d+[a-z]?)'
    table_match = re.search(table_pattern, query_lower)
    if table_match:
        table_num = table_match.group(1)
        if f'table {table_num}' in chunk_text.lower():
            rerank_score *= 1.2
    
    # 8. HSN/Tariff code match boost (exact code match)
    hsn_pattern = r'\b(\d{4,8})\b'
    query_hsn_codes = set(re.findall(hsn_pattern, query))
    if query_hsn_codes:
        chunk_codes = set(re.findall(hsn_pattern, chunk_text))
        if query_hsn_codes.intersection(chunk_codes):
            rerank_score *= 1.25  # Strong boost for exact HSN matches
    
    return rerank_score

def rerank_results(
    results: List[Any],
    query: str,
    top_k: int = 10
) -> List[Any]:
    """
    Rerank results using multiple signals.
    """
    if not results:
        return []
    
    query_keywords = extract_keywords(query)
    
    # Calculate rerank scores
    reranked = []
    for result in results:
        # Handle both regular results and wrapped results
        if hasattr(result, 'point'):
            # Wrapped result from hybrid search
            actual_result = result.point
            base_score = result.hybrid_score if hasattr(result, 'hybrid_score') else actual_result.score
        else:
            # Regular Qdrant result
            actual_result = result
            base_score = result.score if hasattr(result, 'score') else 0.0
        
        rerank_score = calculate_rerank_score(actual_result, query, query_keywords, base_score)
        
        reranked.append({
            "result": actual_result,
            "rerank_score": rerank_score,
            "original_score": base_score
        })
    
    # Sort by rerank score
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Return top K with updated scores
    final_results = []
    for item in reranked[:top_k]:
        result = item["result"]
        # Update score to rerank score
        result.score = item["rerank_score"]
        final_results.append(result)
    
    return final_results

# --------------------------------------------------
# MAIN ENHANCED SEARCH FUNCTION
# --------------------------------------------------
def enhanced_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 10,
    use_hybrid: bool = True,
    use_multi_query: bool = True,
    use_reranking: bool = True,
    filters: Optional[Filter] = None,
    min_score: float = 0.0
) -> List[Any]:
    """
    Main enhanced search function combining multiple strategies.
    """
    logger.info(f"Enhanced search for query: {query[:100]}...")
    
    # Step 1: Choose search strategy
    if use_multi_query:
        # Multi-query search with fusion
        results = multi_query_search(client, collection_name, query, top_k * 2, filters)
    else:
        # Single query semantic search
        try:
            vector = embed_text(query)
            search_results = client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=top_k * 2,
                query_filter=filters,
                with_payload=True
            ).points
            results = search_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    if not results:
        return []
    
    # Step 2: Apply hybrid scoring if enabled
    if use_hybrid and not use_multi_query:  # Multi-query already does some fusion
        # Convert results to format expected by hybrid_search
        keywords = extract_keywords(query)
        
        hybrid_scored = []
        for result in results:
            chunk_text = result.payload.get("chunk_text", "").lower()
            semantic_score = result.score
            keyword_score = calculate_keyword_score(chunk_text, keywords)
            hybrid_score = (0.7 * semantic_score) + (0.3 * keyword_score)
            result.score = hybrid_score
            hybrid_scored.append(result)
        
        results = hybrid_scored
        results.sort(key=lambda x: x.score, reverse=True)
    
    # Step 3: Rerank results
    if use_reranking:
        results = rerank_results(results, query, top_k * 2)
    
    # Step 4: Filter by minimum score
    if min_score > 0:
        results = [r for r in results if r.score >= min_score]
    
    # Step 5: Return top K
    return results[:top_k]

