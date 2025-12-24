"""
Enhanced Search Methods for RAG System - Improved Version
Fixes accuracy issues with better validation and score normalization.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ingestion.embeddings import embed_text, embed_batch

logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MAX_QUERY_EXPANSIONS = 3
RECENCY_BOOST_FACTOR = 1.05  # Reduced from 1.1 to prevent over-boosting
EXACT_MATCH_BOOST = 1.3  # Reduced from 1.5 to be more conservative
MIN_KEYWORD_MATCH_RATIO = 0.3  # At least 30% of important keywords must match

# --------------------------------------------------
# QUERY ANALYSIS & VALIDATION
# --------------------------------------------------
def analyze_query_intent(query: str) -> Dict[str, Any]:
    """
    Analyze query to determine intent and extract mandatory requirements.
    Returns dict with intent type and mandatory fields that MUST be present in results.
    CRITICAL: Tax types have independent notification numbering streams.
    """
    query_lower = query.lower()
    intent = {
        "type": "general",
        "mandatory_notification": None,
        "mandatory_tax_type": None,  # CRITICAL: Tax type must match exactly
        "mandatory_form": None,
        "mandatory_table": None,
        "mandatory_schedule": None,
        "mandatory_keywords": [],
        "has_specific_lookup": False
    }
    
    # Extract tax type (MANDATORY if present) - MUST match exactly
    # CRITICAL: Each tax type has independent notification numbering streams
    tax_type_patterns = {
        "Central Tax": ["central tax", "cgst", "central goods"],
        "Integrated Tax": ["integrated tax", "igst", "integrated goods"],
        "Union Territory Tax": ["union territory tax", "utgst", "ut tax", "union territory"],
        "Compensation Cess": ["compensation cess", "cess", "compensation tax"]
    }
    
    for tax_type, patterns in tax_type_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            intent["mandatory_tax_type"] = tax_type
            intent["has_specific_lookup"] = True
            logger.info(f"Extracted mandatory tax type: {tax_type}")
            break
    
    # Extract notification number (MANDATORY if present)
    # Pattern: "notification 01/2017" or "01/2017" or "notif 01/2017"
    notif_pattern = r'(?:notification|notif|notfn|notfctn)?\s*(\d+)[/\-](\d{4})'
    notif_match = re.search(notif_pattern, query_lower)
    if notif_match:
        # Normalize: "01/2017" format - preserve exact format for matching
        num = notif_match.group(1)
        year = notif_match.group(2)
        # Store exact format and variants for matching
        intent["mandatory_notification"] = f"{num}/{year}"
        # Create variants: with/without leading zeros
        num_int = int(num)  # Remove leading zeros: "01" -> 1
        intent["mandatory_notification_variants"] = [
            f"{num}/{year}",  # Original: "01/2017"
            f"{num_int}/{year}",  # Without leading zeros: "1/2017"
            f"{str(num_int).zfill(2)}/{year}" if num_int < 10 else f"{num_int}/{year}"  # With leading zero: "01/2017"
        ]
        intent["has_specific_lookup"] = True
        intent["type"] = "notification_lookup"
        logger.info(f"Extracted mandatory notification: {intent['mandatory_notification']}")
    
    # Extract form references (MANDATORY if present)
    form_pattern = r'\bgstr[-\s]?(\d+[a-z]?)\b'
    form_match = re.search(form_pattern, query_lower)
    if form_match:
        intent["mandatory_form"] = f"gstr-{form_match.group(1)}"
        intent["mandatory_keywords"].append(intent["mandatory_form"])
        intent["has_specific_lookup"] = True
    
    # Extract table references (MANDATORY if present)
    table_pattern = r'\btable\s+(\d+[a-z]?)'
    table_match = re.search(table_pattern, query_lower)
    if table_match:
        intent["mandatory_table"] = f"table {table_match.group(1)}"
        intent["mandatory_keywords"].append(intent["mandatory_table"])
        intent["has_specific_lookup"] = True
    
    # Extract schedule (MANDATORY if present)
    schedule_pattern = r'\bschedule\s+([ivx]+|[1-6])\b'
    schedule_match = re.search(schedule_pattern, query_lower)
    if schedule_match:
        intent["mandatory_schedule"] = schedule_match.group(1).lower()
        intent["mandatory_keywords"].append(f"schedule {intent['mandatory_schedule']}")
        intent["has_specific_lookup"] = True
    
    # For rate queries
    if any(word in query_lower for word in ["rate", "tax rate", "gst rate", "percentage"]):
        intent["type"] = "rate_query"
    
    return intent

def validate_result_against_intent(result: Any, intent: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Validate if result matches mandatory requirements from query intent.
    Returns (is_valid, penalty_score) where penalty_score reduces score if validation fails.
    CRITICAL: Tax types have independent notification numbering streams - must match exactly.
    """
    payload = result.payload
    chunk_text = payload.get("chunk_text", "").lower()
    penalty = 0.0
    
    # CRITICAL: If tax type is mandatory, it MUST match exactly FIRST
    # Different tax types have completely independent notification numbering streams
    # Example: "01/2017 - Central Tax" is DIFFERENT from "01/2017 - Integrated Tax"
    if intent.get("mandatory_tax_type"):
        tax_type = payload.get("tax_type", "")
        # Normalize tax type for comparison (remove "(Rate)" suffix if present)
        tax_type_normalized = tax_type.replace(" (Rate)", "").replace(" (Rate", "").strip()
        mandatory_tax = intent["mandatory_tax_type"]
        
        # Exact matching - tax types must match
        # Map variations to canonical names
        tax_type_map = {
            "Central Tax": ["Central Tax", "Central"],
            "Integrated Tax": ["Integrated Tax", "Integrated"],
            "Union Territory Tax": ["Union Territory Tax", "Union Territory", "UT Tax"],
            "Compensation Cess": ["Compensation Cess", "Compensation Tax", "Cess"]
        }
        
        mandatory_variants = tax_type_map.get(mandatory_tax, [mandatory_tax])
        
        # Check if result's tax type matches any variant of mandatory tax type
        tax_matches = any(
            variant.lower() in tax_type_normalized.lower() or 
            tax_type_normalized.lower() in variant.lower()
            for variant in mandatory_variants
        )
        
        if not tax_matches:
            logger.debug(f"Tax type mismatch: required '{mandatory_tax}', got '{tax_type}'")
            return False, 1.0  # Completely reject if tax type doesn't match
    
    # CRITICAL: If notification number is mandatory, it MUST match exactly
    # AND must be from the same tax type (already validated above)
    if intent.get("mandatory_notification"):
        notification_no = payload.get("notification_no", "")
        notification_no_lower = notification_no.lower()
        
        # Check all variants of the notification number (with/without leading zeros)
        variants = intent.get("mandatory_notification_variants", [intent["mandatory_notification"]])
        
        # Exact match required - notification number must be exactly as specified
        notification_matches = any(
            variant.lower() in notification_no_lower or 
            notification_no_lower in variant.lower()
            for variant in variants
        )
        
        if not notification_matches:
            logger.debug(f"Notification mismatch: required '{intent['mandatory_notification']}', got '{notification_no}'")
            return False, 1.0  # Completely reject if notification doesn't match
    
    # CRITICAL: If form is mandatory, it MUST be in chunk text
    if intent["mandatory_form"]:
        form_variants = [
            intent["mandatory_form"],
            intent["mandatory_form"].replace("-", " "),
            f"form {intent['mandatory_form']}"
        ]
        if not any(variant in chunk_text for variant in form_variants):
            return False, 1.0  # Completely reject if form not found
    
    # CRITICAL: If table is mandatory, it MUST be in chunk text
    if intent["mandatory_table"]:
        if intent["mandatory_table"] not in chunk_text:
            return False, 1.0  # Completely reject if table not found
    
    # CRITICAL: If schedule is mandatory, it MUST be in chunk text
    if intent["mandatory_schedule"]:
        schedule_variants = [
            f"schedule {intent['mandatory_schedule']}",
            f"schedule {intent['mandatory_schedule'].upper()}",
        ]
        # Also check numeric/roman equivalents
        schedule_map = {'i': '1', '1': 'i', 'ii': '2', '2': 'ii', 'iii': '3', '3': 'iii',
                       'iv': '4', '4': 'iv', 'v': '5', '5': 'v', 'vi': '6', '6': 'vi'}
        alt_schedule = schedule_map.get(intent["mandatory_schedule"])
        if alt_schedule:
            schedule_variants.extend([f"schedule {alt_schedule}", f"schedule {alt_schedule.upper()}"])
        
        if not any(variant in chunk_text for variant in schedule_variants):
            return False, 1.0  # Completely reject if schedule not found
    
    # Validate mandatory keywords (at least 50% must be present)
    if intent["mandatory_keywords"]:
        matches = sum(1 for kw in intent["mandatory_keywords"] if kw in chunk_text)
        match_ratio = matches / len(intent["mandatory_keywords"]) if intent["mandatory_keywords"] else 0
        if match_ratio < 0.5:
            penalty = 0.5  # Heavy penalty if less than 50% match
    
    return True, penalty

# --------------------------------------------------
# IMPROVED QUERY EXPANSION
# --------------------------------------------------
def expand_query(query: str, conservative: bool = True) -> List[str]:
    """
    Generate query variations more conservatively to avoid bad expansions.
    """
    expansions = [query]  # Original query is always first
    
    query_lower = query.lower()
    
    # Only expand if query is not too specific (has mandatory requirements)
    intent = analyze_query_intent(query)
    if intent["has_specific_lookup"] and conservative:
        # For specific lookups, don't expand too much - just add original
        logger.info(f"Query has specific lookup requirements, using minimal expansion")
        return expansions[:1]  # Only return original
    
    # 1. Abbreviation expansion (conservative)
    abbreviations = {
        "gst": "goods and services tax",
        "cgst": "central goods and services tax",
        "igst": "integrated goods and services tax",
    }
    
    expanded = query
    for abbr, full in abbreviations.items():
        if f" {abbr} " in query_lower or query_lower.startswith(abbr + " "):
            expanded = expanded.replace(f" {abbr} ", f" {full} ").replace(f" {abbr.upper()} ", f" {full} ")
            if expanded != query and expanded not in expansions:
                expansions.append(expanded)
                break  # Only do one expansion for conservative mode
    
    # 2. Remove date qualifiers for broader search (only if not a date-specific query)
    if "rate" not in query_lower or "as on" not in query_lower:
        date_pattern = r'\s+as\s+(on|of)\s+\w+\s+\d{4}'
        without_date = re.sub(date_pattern, '', query, flags=re.IGNORECASE)
        if without_date != query and without_date.strip() not in expansions:
            expansions.append(without_date.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expansions = []
    for exp in expansions:
        exp_lower = exp.lower().strip()
        if exp_lower and exp_lower not in seen:
            seen.add(exp_lower)
            unique_expansions.append(exp)
    
    return unique_expansions[:MAX_QUERY_EXPANSIONS]

# --------------------------------------------------
# IMPROVED KEYWORD EXTRACTION
# --------------------------------------------------
def extract_keywords(query: str) -> List[str]:
    """
    Extract important keywords, prioritizing exact phrases and identifiers.
    More conservative - only extracts truly important terms.
    """
    keywords = []
    query_lower = query.lower()
    
    # Stop words
    stop_words = {
        "what", "is", "the", "for", "of", "and", "or", "but", "in", "on", "at",
        "to", "a", "an", "as", "are", "was", "were", "been", "be", "have", "has",
        "which", "who", "when", "where", "why", "how", "please", "tell", "me",
        "whether", "that", "this", "these", "those"
    }
    
    # 1. Extract Table references
    table_pattern = r'\btable\s+(\d+[a-z]?)'
    table_matches = re.findall(table_pattern, query_lower)
    for match in table_matches:
        keywords.append(f"table {match}")
    
    # 2. Extract GSTR form references
    gstr_pattern = r'\bgstr[-\s]?(\d+[a-z]?)\b'
    gstr_matches = re.findall(gstr_pattern, query_lower)
    for match in gstr_matches:
        keywords.append(f"gstr-{match}")
        keywords.append(f"gstr {match}")
    
    # 3. Extract code references (SAC, HSN)
    code_pattern = r'\b(sac|hsn)(?:\s+code)?\b'
    code_matches = re.findall(code_pattern, query_lower)
    keywords.extend(code_matches)
    
    # 4. Extract Financial Year
    fy_pattern = r'\bfy\s*(\d{4}[-/]\d{2,4})'
    fy_matches = re.findall(fy_pattern, query_lower)
    for match in fy_matches:
        keywords.append(f"fy {match}")
        keywords.append(match)
    
    # 5. Extract Notification numbers
    notif_pattern = r'\b(\d+)[/\-](\d{4})\b'
    notif_matches = re.findall(notif_pattern, query_lower)
    for match in notif_matches:
        keywords.append(f"{match[0]}/{match[1]}")
    
    # 6. Extract Rule/Section references
    rule_pattern = r'\b(rule|section|sec)\s+(\d+[a-z]?)\b'
    rule_matches = re.findall(rule_pattern, query_lower)
    for match in rule_matches:
        keywords.append(f"{match[0]} {match[1]}")
    
    # 7. Extract Schedule references
    schedule_pattern = r'\bschedule\s+([ivx]+|[1-6])\b'
    schedule_matches = re.findall(schedule_pattern, query_lower)
    for match in schedule_matches:
        keywords.append(f"schedule {match}")
    
    # 8. Extract important nouns/adjectives (skip common words)
    remaining_query = query_lower
    for pattern in [table_pattern, gstr_pattern, code_pattern, fy_pattern, notif_pattern, rule_pattern, schedule_pattern]:
        remaining_query = re.sub(pattern, '', remaining_query)
    
    # Extract meaningful words (length > 3, not stop words, not numbers only)
    words = re.findall(r'\b\w+\b', remaining_query)
    important_words = [
        w for w in words 
        if w not in stop_words and len(w) > 3 and not w.isdigit()
        and w not in ['rate', 'gst', 'tax']  # Too generic
    ]
    
    # Only add top 5 most important words to avoid dilution
    keywords.extend(important_words[:5])
    
    # Remove duplicates
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower().strip()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw_lower)
    
    return unique_keywords

def calculate_keyword_score(chunk_text: str, keywords: List[str]) -> float:
    """
    Calculate keyword matching score with emphasis on exact phrase matches.
    """
    if not keywords:
        return 0.0
    
    chunk_lower = chunk_text.lower()
    matches = 0
    
    for keyword in keywords:
        # Exact phrase matching
        if keyword in chunk_lower:
            matches += 1
    
    # Normalize score (0-1)
    return matches / len(keywords) if keywords else 0.0

# --------------------------------------------------
# IMPROVED MULTI-QUERY SEARCH
# --------------------------------------------------
def build_qdrant_filters(intent: Dict[str, Any]) -> Optional[Filter]:
    """
    Build Qdrant filters based on query intent.
    CRITICAL: Use filters to restrict search to correct tax type and notification.
    """
    conditions = []
    
    # Filter by tax type if specified
    if intent.get("mandatory_tax_type"):
        tax_type = intent["mandatory_tax_type"]
        # Map to possible tax_type values in database
        tax_type_variants = {
            "Central Tax": ["Central Tax", "Central Tax (Rate)"],
            "Integrated Tax": ["Integrated Tax", "Integrated Tax (Rate)"],
            "Union Territory Tax": ["Union Territory Tax", "Union Territory Tax (Rate)"],
            "Compensation Cess": ["Compensation Cess", "Compensation Cess (Rate)"]
        }
        
        variants = tax_type_variants.get(tax_type, [tax_type])
        # Use "should" to match any variant
        from qdrant_client.models import Should
        tax_conditions = [
            FieldCondition(key="tax_type", match=MatchValue(value=variant))
            for variant in variants
        ]
        conditions.append(Should(should=tax_conditions))
    
    # Filter by notification number if specified
    if intent.get("mandatory_notification"):
        notification = intent["mandatory_notification"]
        variants = intent.get("mandatory_notification_variants", [notification])
        # Use "should" to match any variant
        from qdrant_client.models import Should
        notif_conditions = [
            FieldCondition(key="notification_no", match=MatchValue(value=variant))
            for variant in variants
        ]
        conditions.append(Should(should=notif_conditions))
    
    if conditions:
        return Filter(must=conditions)
    return None

def multi_query_search(
    client: QdrantClient,
    collection_name: str,
    original_query: str,
    top_k: int = 10,
    filters: Optional[Filter] = None
) -> List[Any]:
    """
    Improved multi-query search with better fusion and validation.
    CRITICAL: Applies tax type and notification filters to prevent cross-tax-type matches.
    """
    # Generate conservative query expansions
    query_variations = expand_query(original_query, conservative=True)
    
    logger.info(f"Searching with {len(query_variations)} query variations")
    
    # Analyze query intent
    intent = analyze_query_intent(original_query)
    
    # Build Qdrant filters from intent (CRITICAL for tax type isolation)
    intent_filters = build_qdrant_filters(intent)
    
    # Combine with provided filters
    if intent_filters and filters:
        # Merge filters (both must be satisfied)
        from qdrant_client.models import Must
        combined_conditions = []
        if hasattr(filters, 'must'):
            combined_conditions.extend(filters.must)
        if hasattr(intent_filters, 'must'):
            combined_conditions.extend(intent_filters.must)
        final_filter = Filter(must=combined_conditions) if combined_conditions else filters
    elif intent_filters:
        final_filter = intent_filters
    else:
        final_filter = filters
    
    if intent_filters:
        logger.info(f"Applying Qdrant filters: tax_type={intent.get('mandatory_tax_type')}, notification={intent.get('mandatory_notification')}")
    
    all_results = {}
    
    for variation in query_variations:
        try:
            vector = embed_text(variation)
            results = client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=top_k * 3,  # Get more candidates
                query_filter=final_filter,  # Use combined filters
                with_payload=True
            ).points
            
            # Reciprocal Rank Fusion (RRF) with validation
            for rank, result in enumerate(results, 1):
                # Validate against intent first
                is_valid, penalty = validate_result_against_intent(result, intent)
                if not is_valid:
                    continue  # Skip invalid results
                
                # Create unique ID
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
                        "penalties": [],
                        "count": 0
                    }
                
                all_results[point_id]["rrf_score"] += rrf_score
                all_results[point_id]["scores"].append(result.score)
                all_results[point_id]["penalties"].append(penalty)
                all_results[point_id]["count"] += 1
        
        except Exception as e:
            logger.warning(f"Query variation failed: {variation[:50]}... - {e}")
            continue
    
    # Convert to list and sort by RRF score
    fused_results = list(all_results.values())
    
    # Apply penalties
    for item in fused_results:
        avg_penalty = sum(item["penalties"]) / len(item["penalties"]) if item["penalties"] else 0
        item["rrf_score"] *= (1 - avg_penalty)
    
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    # Update scores
    final_results = []
    for item in fused_results[:top_k * 2]:
        result = item["point"]
        # Use average semantic score
        avg_score = sum(item["scores"]) / len(item["scores"]) if item["scores"] else result.score
        # Normalize RRF score
        max_possible_rrf = sum(1.0 / (60 + i) for i in range(1, len(query_variations) + 1)) * len(query_variations)
        normalized_rrf = min(item["rrf_score"] / max_possible_rrf if max_possible_rrf > 0 else 0, 1.0)
        # Combine: 80% semantic, 20% RRF (reduced RRF weight)
        final_score = (0.8 * avg_score) + (0.2 * normalized_rrf)
        
        result.score = final_score
        final_results.append(result)
    
    return final_results

# --------------------------------------------------
# IMPROVED HYBRID SEARCH
# --------------------------------------------------
def apply_hybrid_scoring(
    results: List[Any],
    query: str,
    query_keywords: List[str]
) -> List[Any]:
    """
    Apply hybrid scoring (semantic + keyword) to results.
    """
    if not results:
        return results
    
    hybrid_scored = []
    for result in results:
        chunk_text = result.payload.get("chunk_text", "").lower()
        semantic_score = result.score
        
        # Keyword score
        keyword_score = calculate_keyword_score(chunk_text, query_keywords)
        
        # Combine: 75% semantic, 25% keyword (slightly more weight to semantic)
        hybrid_score = (0.75 * semantic_score) + (0.25 * keyword_score)
        
        # Only boost if keyword match is strong (at least 50% keywords match)
        if keyword_score >= 0.5:
            hybrid_score *= 1.1
        
        result.score = hybrid_score
        hybrid_scored.append(result)
    
    hybrid_scored.sort(key=lambda x: x.score, reverse=True)
    return hybrid_scored

# --------------------------------------------------
# IMPROVED RERANKING
# --------------------------------------------------
def calculate_rerank_score(
    result: Any,
    query: str,
    query_keywords: List[str],
    base_score: float
) -> float:
    """
    Calculate reranking score with conservative boosts.
    """
    payload = result.payload
    chunk_text = payload.get("chunk_text", "").lower()
    query_lower = query.lower()
    
    rerank_score = base_score
    
    # 1. Exact phrase matching (conservative boost)
    if len(query) > 10 and query_lower in chunk_text:
        rerank_score *= 1.2  # Reduced from 1.5
    
    # 2. Notification number exact match (strong boost)
    notification_no = payload.get("notification_no", "")
    notif_pattern = r'(\d+)[/\-](\d{4})'
    query_notif = re.search(notif_pattern, query, re.IGNORECASE)
    if query_notif and notification_no:
        query_notif_str = f"{query_notif.group(1)}/{query_notif.group(2)}"
        if query_notif_str.lower() in notification_no.lower():
            rerank_score *= 1.3
    
    # 3. Form/Table match (strong boost)
    form_pattern = r'\bgstr[-\s]?(\d+[a-z]?)\b'
    form_match = re.search(form_pattern, query_lower)
    if form_match:
        form_num = form_match.group(1)
        if f'gstr-{form_num}' in chunk_text or f'gstr {form_num}' in chunk_text:
            rerank_score *= 1.2
    
    table_pattern = r'\btable\s+(\d+[a-z]?)'
    table_match = re.search(table_pattern, query_lower)
    if table_match:
        table_num = table_match.group(1)
        if f'table {table_num}' in chunk_text:
            rerank_score *= 1.2
    
    # 4. Schedule match
    schedule_pattern = r'\bschedule\s+([ivx]+|[1-6])\b'
    query_schedule_match = re.search(schedule_pattern, query_lower)
    if query_schedule_match:
        query_schedule = query_schedule_match.group(1).lower()
        if f'schedule {query_schedule}' in chunk_text or f'schedule {query_schedule.upper()}' in chunk_text:
            rerank_score *= 1.15
    
    # 5. Tax type match (small boost)
    tax_type = payload.get("tax_type", "").lower()
    if "central tax" in query_lower or "cgst" in query_lower:
        if "central" in tax_type:
            rerank_score *= 1.05
    if "integrated tax" in query_lower or "igst" in query_lower:
        if "integrated" in tax_type:
            rerank_score *= 1.05
    
    # Cap the rerank score to prevent excessive inflation
    rerank_score = min(rerank_score, base_score * 1.5)  # Max 50% boost
    
    return rerank_score

def rerank_results(
    results: List[Any],
    query: str,
    top_k: int = 10
) -> List[Any]:
    """
    Rerank results with improved validation.
    """
    if not results:
        return []
    
    query_keywords = extract_keywords(query)
    intent = analyze_query_intent(query)
    
    reranked = []
    for result in results:
        # Validate against intent
        is_valid, penalty = validate_result_against_intent(result, intent)
        if not is_valid:
            continue  # Skip invalid results
        
        base_score = result.score if hasattr(result, 'score') else 0.0
        rerank_score = calculate_rerank_score(result, query, query_keywords, base_score)
        
        # Apply penalty
        rerank_score *= (1 - penalty)
        
        reranked.append({
            "result": result,
            "rerank_score": rerank_score,
            "original_score": base_score
        })
    
    # Sort by rerank score
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Update scores
    final_results = []
    for item in reranked[:top_k]:
        result = item["result"]
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
    Main enhanced search function with improved accuracy.
    """
    logger.info(f"Enhanced search for query: {query[:100]}...")
    
    # Analyze query intent
    intent = analyze_query_intent(query)
    logger.info(f"Query intent: {intent['type']}, mandatory requirements: {len(intent['mandatory_keywords'])} keywords")
    
    # Step 1: Multi-query search if enabled
    if use_multi_query:
        results = multi_query_search(client, collection_name, query, top_k * 2, filters)
    else:
        # Single query search
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
    
    # Step 2: Validate and filter by intent
    validated_results = []
    for result in results:
        is_valid, penalty = validate_result_against_intent(result, intent)
        if is_valid:
            validated_results.append(result)
        else:
            logger.debug(f"Result filtered out due to intent validation failure")
    
    if not validated_results:
        logger.warning("No results passed intent validation")
        return []
    
    results = validated_results
    
    # Step 3: Apply hybrid scoring (NOW WORKS WITH MULTI-QUERY)
    if use_hybrid:
        query_keywords = extract_keywords(query)
        results = apply_hybrid_scoring(results, query, query_keywords)
    
    # Step 4: Rerank results
    if use_reranking:
        results = rerank_results(results, query, top_k * 2)
    
    # Step 5: Filter by minimum score
    if min_score > 0:
        results = [r for r in results if r.score >= min_score]
    
    # Step 6: Return top K
    return results[:top_k]

