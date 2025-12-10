import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger
from dataclasses import dataclass
from collections import defaultdict

# Configure logger
logger.remove()
logger.add(lambda msg: print(msg), format="{message}", level="INFO")

# ============================================================================
# Test Queries with Expected Results
# ============================================================================

# These are test queries with expected relevant sections based on the document
# "Lesson15-Preclass-Reading-Pandas_SQL.md"
TEST_CASES = [
    {
        "query": "Cách chọn cột trong Pandas giống SELECT trong SQL",
        "expected_keywords": ["SELECT", "df['column']", "chọn cột", "column_name"],
        "expected_section": "II.2",  # Thao tác 1: Chọn cột dữ liệu
        "description": "Test SELECT column operation"
    },
    {
        "query": "Lọc dữ liệu theo điều kiện WHERE trong Pandas",
        "expected_keywords": ["WHERE", ".loc", "điều kiện", "lọc"],
        "expected_section": "II.3",  # Thao tác 2: Lọc theo điều kiện đơn
        "description": "Test WHERE filtering"
    },
    {
        "query": "Sử dụng toán tử AND OR trong Pandas để kết hợp nhiều điều kiện",
        "expected_keywords": ["AND", "OR", "&", "|", "đa điều kiện", "bitwise"],
        "expected_section": "II.4",  # Thao tác 3: Lọc đa điều kiện
        "description": "Test AND/OR operators"
    },
    {
        "query": "DataFrame trong Pandas tương đương với gì trong SQL",
        "expected_keywords": ["DataFrame", "Table", "cơ sở dữ liệu", "SQL"],
        "expected_section": "I",  # Tư duy SQL trong Pandas
        "description": "Test DataFrame concept"
    },
    {
        "query": "Bộ dữ liệu Advertising Simple có những cột nào",
        "expected_keywords": ["TV", "Radio", "Newspaper", "Sales", "Advertising"],
        "expected_section": "II",  # Ví dụ minh họa
        "description": "Test dataset columns"
    },
    {
        "query": "Cách đọc file CSV trong Pandas",
        "expected_keywords": ["read_csv", "CSV", "đọc dữ liệu"],
        "expected_section": "II.1",  # Thực hành thao tác dữ liệu
        "description": "Test CSV reading"
    },
    {
        "query": "Boolean Series trong Pandas hoạt động như thế nào",
        "expected_keywords": ["Boolean", "True", "False", "Series", "lọc"],
        "expected_section": "II.3",  # Lưu ý về cú pháp
        "description": "Test Boolean indexing concept"
    }
]


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    text: str
    section_path: List[str]
    is_relevant: bool = False
    keyword_matches: int = 0


@dataclass
class SearchEvaluation:
    method_name: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    mrr: float  # Mean Reciprocal Rank
    avg_keyword_match: float
    avg_latency_ms: float


# ============================================================================
# Current Search Method
# ============================================================================

def search_current_method(
    prompt: str,
    faiss_index,
    embedding_model,
    top_k: int = 5,
    document_id: str = None
) -> Tuple[List[SearchResult], float]:
    start_time = time.time()
    
    # Generate embedding for prompt
    prompt_embedding = embedding_model.encode([prompt], convert_to_numpy=True)[0]
    
    # Search in FAISS
    search_k = top_k * 3 if document_id else top_k
    results = faiss_index.search(prompt_embedding, k=search_k)
    
    # Filter by document_id and format
    search_results = []
    for chunk_id, score, metadata in results:
        if document_id and metadata.get("document_id") != document_id:
            continue
        
        search_results.append(SearchResult(
            chunk_id=chunk_id,
            score=score,
            text=metadata.get("text", ""),
            section_path=metadata.get("section_path", [])
        ))
        
        if len(search_results) >= top_k:
            break
    
    latency_ms = (time.time() - start_time) * 1000
    return search_results, latency_ms


# ============================================================================
# Improved Search Methods
# ============================================================================

def search_with_query_expansion(
    prompt: str,
    faiss_index,
    embedding_model,
    top_k: int = 5,
    document_id: str = None
) -> Tuple[List[SearchResult], float]:
    start_time = time.time()
    
    # Simple query expansion mappings
    expansions = {
        "chọn": ["select", "lấy", "pick"],
        "lọc": ["filter", "where", "điều kiện"],
        "cột": ["column", "trường", "field"],
        "hàng": ["row", "bản ghi", "record"],
        "bảng": ["table", "dataframe"],
        "pandas": ["python", "dataframe"],
        "sql": ["query", "truy vấn", "database"],
    }
    
    # Expand query
    expanded_terms = [prompt]
    prompt_lower = prompt.lower()
    for key, values in expansions.items():
        if key in prompt_lower:
            expanded_terms.extend(values)
    
    expanded_query = prompt + " " + " ".join(expanded_terms[:5])
    
    # Generate embedding for expanded query
    prompt_embedding = embedding_model.encode([expanded_query], convert_to_numpy=True)[0]
    
    # Search in FAISS
    search_k = top_k * 3 if document_id else top_k
    results = faiss_index.search(prompt_embedding, k=search_k)
    
    # Filter and format
    search_results = []
    for chunk_id, score, metadata in results:
        if document_id and metadata.get("document_id") != document_id:
            continue
        
        search_results.append(SearchResult(
            chunk_id=chunk_id,
            score=score,
            text=metadata.get("text", ""),
            section_path=metadata.get("section_path", [])
        ))
        
        if len(search_results) >= top_k:
            break
    
    latency_ms = (time.time() - start_time) * 1000
    return search_results, latency_ms


def search_with_mmr(
    prompt: str,
    faiss_index,
    embedding_model,
    top_k: int = 5,
    document_id: str = None,
    lambda_param: float = 0.7  # Balance between relevance and diversity
) -> Tuple[List[SearchResult], float]:
    start_time = time.time()
    
    # Generate embedding for prompt
    prompt_embedding = embedding_model.encode([prompt], convert_to_numpy=True)[0]
    
    # Get more candidates for MMR selection
    search_k = top_k * 5 if document_id else top_k * 3
    candidates = faiss_index.search(prompt_embedding, k=search_k)
    
    # Filter by document_id first
    filtered_candidates = []
    for chunk_id, score, metadata in candidates:
        if document_id and metadata.get("document_id") != document_id:
            continue
        filtered_candidates.append((chunk_id, score, metadata))
    
    if not filtered_candidates:
        return [], (time.time() - start_time) * 1000
    
    # Get embeddings for all candidates (from metadata text)
    candidate_texts = [m.get("text", "") for _, _, m in filtered_candidates]
    candidate_embeddings = embedding_model.encode(candidate_texts, convert_to_numpy=True)
    
    # Normalize embeddings
    candidate_embeddings = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9)
    prompt_norm = prompt_embedding / (np.linalg.norm(prompt_embedding) + 1e-9)
    
    # MMR selection
    selected_indices = []
    selected_results = []
    
    while len(selected_results) < top_k and len(selected_indices) < len(filtered_candidates):
        best_score = -float('inf')
        best_idx = -1
        
        for i, (chunk_id, relevance_score, metadata) in enumerate(filtered_candidates):
            if i in selected_indices:
                continue
            
            # Relevance score (query-document similarity)
            relevance = relevance_score
            
            # Diversity penalty (max similarity to selected documents)
            diversity_penalty = 0
            if selected_indices:
                selected_embeddings = candidate_embeddings[selected_indices]
                similarities = np.dot(selected_embeddings, candidate_embeddings[i])
                diversity_penalty = np.max(similarities)
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx >= 0:
            selected_indices.append(best_idx)
            chunk_id, score, metadata = filtered_candidates[best_idx]
            selected_results.append(SearchResult(
                chunk_id=chunk_id,
                score=score,
                text=metadata.get("text", ""),
                section_path=metadata.get("section_path", [])
            ))
    
    latency_ms = (time.time() - start_time) * 1000
    return selected_results, latency_ms


def search_with_bm25_hybrid(
    prompt: str,
    faiss_index,
    embedding_model,
    chunks: List[Dict],  # Raw chunks with full text
    top_k: int = 5,
    document_id: str = None,
    semantic_weight: float = 0.7
) -> Tuple[List[SearchResult], float]:
    from rank_bm25 import BM25Okapi
    import re
    
    start_time = time.time()
    
    # Filter chunks by document_id
    if document_id:
        doc_chunks = [c for c in chunks if c.get("document_id") == document_id or 
                     c.get("chunk_id", "").startswith(document_id)]
    else:
        doc_chunks = chunks
    
    if not doc_chunks:
        return [], (time.time() - start_time) * 1000
    
    # Tokenize for BM25
    def tokenize(text: str) -> List[str]:
        # Simple Vietnamese-aware tokenization
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    corpus = [tokenize(c.get("text", "")) for c in doc_chunks]
    bm25 = BM25Okapi(corpus)
    
    # BM25 scores
    query_tokens = tokenize(prompt)
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores
    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_normalized = bm25_scores / bm25_max
    
    # Semantic search
    prompt_embedding = embedding_model.encode([prompt], convert_to_numpy=True)[0]
    search_k = len(doc_chunks)
    results = faiss_index.search(prompt_embedding, k=search_k, deduplicate=True)
    
    # Create score mapping from FAISS results
    semantic_scores = {}
    for chunk_id, score, metadata in results:
        if document_id and metadata.get("document_id") != document_id:
            continue
        semantic_scores[chunk_id] = score
    
    # Combine scores
    combined_results = []
    for i, chunk in enumerate(doc_chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        semantic_score = semantic_scores.get(chunk_id, 0)
        bm25_score = bm25_normalized[i]
        
        # Hybrid score
        hybrid_score = semantic_weight * semantic_score + (1 - semantic_weight) * bm25_score
        
        combined_results.append({
            "chunk_id": chunk_id,
            "score": hybrid_score,
            "text": chunk.get("text", "")[:200],
            "section_path": chunk.get("section_path", []),
            "semantic_score": semantic_score,
            "bm25_score": bm25_score
        })
    
    # Sort by hybrid score
    combined_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Format results
    search_results = []
    for r in combined_results[:top_k]:
        search_results.append(SearchResult(
            chunk_id=r["chunk_id"],
            score=r["score"],
            text=r["text"],
            section_path=r["section_path"]
        ))
    
    latency_ms = (time.time() - start_time) * 1000
    return search_results, latency_ms


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_relevance(
    result: SearchResult,
    expected_keywords: List[str],
    expected_section: str
) -> Tuple[bool, int]:
    text_lower = result.text.lower()
    section_str = " > ".join(result.section_path).lower() if result.section_path else ""
    
    # Count keyword matches
    keyword_matches = 0
    for keyword in expected_keywords:
        if keyword.lower() in text_lower or keyword.lower() in section_str:
            keyword_matches += 1
    
    # Check section match
    section_match = expected_section.lower() in section_str
    
    # Consider relevant if section matches OR has >= 2 keyword matches
    is_relevant = section_match or keyword_matches >= 2
    
    return is_relevant, keyword_matches


def calculate_mrr(results: List[SearchResult], expected_keywords: List[str], expected_section: str) -> float:
    for i, result in enumerate(results):
        is_relevant, _ = evaluate_relevance(result, expected_keywords, expected_section)
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_search_method(
    method_name: str,
    search_func,
    test_cases: List[Dict],
    faiss_index,
    embedding_model,
    chunks: List[Dict] = None,
    document_id: str = None
) -> SearchEvaluation:
    precision_at_1_scores = []
    precision_at_3_scores = []
    precision_at_5_scores = []
    mrr_scores = []
    keyword_match_scores = []
    latencies = []
    
    for test_case in test_cases:
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]
        expected_section = test_case["expected_section"]
        
        # Run search
        if "bm25" in method_name.lower() and chunks:
            results, latency = search_func(
                query, faiss_index, embedding_model, chunks, 
                top_k=5, document_id=document_id
            )
        else:
            results, latency = search_func(
                query, faiss_index, embedding_model, 
                top_k=5, document_id=document_id
            )
        
        latencies.append(latency)
        
        # Evaluate results
        relevant_count_at_1 = 0
        relevant_count_at_3 = 0
        relevant_count_at_5 = 0
        total_keyword_matches = 0
        
        for i, result in enumerate(results):
            is_relevant, keyword_matches = evaluate_relevance(
                result, expected_keywords, expected_section
            )
            result.is_relevant = is_relevant
            result.keyword_matches = keyword_matches
            total_keyword_matches += keyword_matches
            
            if is_relevant:
                if i < 1:
                    relevant_count_at_1 += 1
                if i < 3:
                    relevant_count_at_3 += 1
                if i < 5:
                    relevant_count_at_5 += 1
        
        precision_at_1_scores.append(relevant_count_at_1 / 1)
        precision_at_3_scores.append(relevant_count_at_3 / 3)
        precision_at_5_scores.append(relevant_count_at_5 / 5)
        mrr_scores.append(calculate_mrr(results, expected_keywords, expected_section))
        keyword_match_scores.append(total_keyword_matches / len(expected_keywords) if expected_keywords else 0)
    
    return SearchEvaluation(
        method_name=method_name,
        precision_at_1=np.mean(precision_at_1_scores),
        precision_at_3=np.mean(precision_at_3_scores),
        precision_at_5=np.mean(precision_at_5_scores),
        mrr=np.mean(mrr_scores),
        avg_keyword_match=np.mean(keyword_match_scores),
        avg_latency_ms=np.mean(latencies)
    )


def print_evaluation_results(evaluations: List[SearchEvaluation]):
    
    print("\n" + "=" * 100)
    print("SEARCH METHOD EVALUATION RESULTS")
    print("=" * 100)
    print(f"\n{'Method':<30} {'P@1':>8} {'P@3':>8} {'P@5':>8} {'MRR':>8} {'KeyMatch':>10} {'Latency':>10}")
    print("-" * 100)
    
    for eval in evaluations:
        print(f"{eval.method_name:<30} {eval.precision_at_1:>8.2%} {eval.precision_at_3:>8.2%} "
              f"{eval.precision_at_5:>8.2%} {eval.mrr:>8.3f} {eval.avg_keyword_match:>10.2f} "
              f"{eval.avg_latency_ms:>8.1f}ms")
    
    print("=" * 100)
    
    # Find best method
    best_by_mrr = max(evaluations, key=lambda x: x.mrr)
    best_by_precision = max(evaluations, key=lambda x: x.precision_at_1)
    
    print(f"\n✓ Best by MRR: {best_by_mrr.method_name} (MRR: {best_by_mrr.mrr:.3f})")
    print(f"✓ Best by P@1: {best_by_precision.method_name} (P@1: {best_by_precision.precision_at_1:.2%})")


def print_detailed_results(method_name: str, results: List[SearchResult], test_case: Dict):
    
    print(f"\n{'='*80}")
    print(f"Query: {test_case['query']}")
    print(f"Expected Section: {test_case['expected_section']}")
    print(f"Expected Keywords: {', '.join(test_case['expected_keywords'][:5])}")
    print(f"Method: {method_name}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        is_relevant, keyword_matches = evaluate_relevance(
            result, test_case["expected_keywords"], test_case["expected_section"]
        )
        relevance_indicator = "✓ RELEVANT" if is_relevant else "✗ not relevant"
        section_display = " > ".join(result.section_path[-2:]) if result.section_path else "N/A"
        
        print(f"\n[{i+1}] Score: {result.score:.4f} | Section: {section_display}")
        print(f"    Keywords matched: {keyword_matches} | {relevance_indicator}")
        print(f"    Text: {result.text[:150]}...")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    
    from sentence_transformers import SentenceTransformer
    from document_parse import create_faiss_index_for_chunks, chunk_document, search_chunks_by_prompt
    from app.database.faiss_index import FAISSIndex
    from dataclasses import asdict
    
    # Setup
    file_path = "sample_documents/Lesson15-Preclass-Reading-Pandas_SQL.md"
    document_id = Path(file_path).stem
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    print("\n" + "=" * 80)
    print("SEARCH ACCURACY EVALUATION")
    print("=" * 80)
    print(f"Document: {file_path}")
    print(f"Embedding Model: {embedding_model_name}")
    print(f"Test Cases: {len(TEST_CASES)}")
    
    # Load embedding model
    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Create fresh FAISS index
    print("\nCreating FAISS index...")
    faiss_index, chunk_ids = create_faiss_index_for_chunks(file_path, embedding_model_name)
    
    # Get raw chunks for BM25
    print("\nLoading chunks for BM25...")
    chunks = chunk_document(file_path)
    chunks_dict = []
    for idx, chunk in enumerate(chunks):
        chunk_dict = asdict(chunk)
        chunk_dict["chunk_id"] = f"{document_id}_chunk_{idx}"
        chunk_dict["document_id"] = document_id
        chunks_dict.append(chunk_dict)
    
    print(f"Loaded {len(chunks_dict)} chunks")
    
    # Run evaluations
    print("\n" + "-" * 80)
    print("Running search method evaluations...")
    print("-" * 80)
    
    evaluations = []
    
    # Method 1: Current (FAISS IndexFlatIP)
    print("\n[1/4] Evaluating: Current Method (FAISS Cosine)")
    eval_current = evaluate_search_method(
        "Current (FAISS Cosine)",
        search_current_method,
        TEST_CASES,
        faiss_index,
        embedding_model,
        document_id=document_id
    )
    evaluations.append(eval_current)
    
    # Method 2: Query Expansion
    print("[2/4] Evaluating: Query Expansion")
    eval_expansion = evaluate_search_method(
        "Query Expansion",
        search_with_query_expansion,
        TEST_CASES,
        faiss_index,
        embedding_model,
        document_id=document_id
    )
    evaluations.append(eval_expansion)
    
    # Method 3: MMR (Diversity)
    print("[3/4] Evaluating: MMR (Diversity)")
    eval_mmr = evaluate_search_method(
        "MMR (λ=0.7)",
        search_with_mmr,
        TEST_CASES,
        faiss_index,
        embedding_model,
        document_id=document_id
    )
    evaluations.append(eval_mmr)
    
    # Method 4: BM25 Hybrid
    try:
        print("[4/4] Evaluating: BM25 + Semantic Hybrid")
        eval_hybrid = evaluate_search_method(
            "BM25 Hybrid (0.7 semantic)",
            search_with_bm25_hybrid,
            TEST_CASES,
            faiss_index,
            embedding_model,
            chunks=chunks_dict,
            document_id=document_id
        )
        evaluations.append(eval_hybrid)
    except ImportError:
        print("  ⚠ rank_bm25 not installed, skipping BM25 hybrid evaluation")
        print("  Install with: pip install rank-bm25")
    
    # Print results
    print_evaluation_results(evaluations)
    
    # Show detailed results for one query
    print("\n" + "=" * 80)
    print("DETAILED RESULTS FOR SAMPLE QUERY")
    print("=" * 80)
    
    sample_test = TEST_CASES[0]  # First test case
    
    results_current, _ = search_current_method(
        sample_test["query"], faiss_index, embedding_model, 
        top_k=5, document_id=document_id
    )
    print_detailed_results("Current (FAISS Cosine)", results_current, sample_test)
    
    # Test the integrated search_chunks_by_prompt function
    print("\n" + "=" * 80)
    print("TESTING INTEGRATED HYBRID SEARCH (search_chunks_by_prompt)")
    print("=" * 80)
    
    print(f"\nQuery: {sample_test['query']}")
    print("-" * 80)
    
    # Test with hybrid search enabled
    hybrid_results = search_chunks_by_prompt(
        prompt=sample_test["query"],
        file_path=file_path,
        top_k=5,
        use_hybrid=True,
        semantic_weight=0.7
    )
    
    print("\nHybrid Search Results:")
    for i, r in enumerate(hybrid_results):
        section = " > ".join(r["section_path"][-2:]) if r["section_path"] else "N/A"
        print(f"[{i+1}] Hybrid: {r['similarity_score']:.4f} | Sem: {r['semantic_score']:.4f} | BM25: {r['bm25_score']:.4f}")
        print(f"    Section: {section}")
        print(f"    Text: {r['text_preview'][:100]}...")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    best_method = max(evaluations, key=lambda x: (x.mrr + x.precision_at_1) / 2)
    
    if best_method.method_name == "Current (FAISS Cosine)":
        print("""
✓ Current implementation is performing well!
  
Suggestions for further improvement:
1. Consider using a multilingual model for Vietnamese content:
   - intfloat/multilingual-e5-small (better for Vietnamese)
   - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   
2. Add query preprocessing:
   - Normalize Vietnamese diacritics
   - Expand abbreviations (SQL → Structured Query Language)
   
3. Implement result caching for common queries
""")
    else:
        print(f"""
✓ Recommended method: {best_method.method_name}
  - MRR: {best_method.mrr:.3f}
  - P@1: {best_method.precision_at_1:.2%}
  - Avg Latency: {best_method.avg_latency_ms:.1f}ms

✓ Hybrid search has been integrated into search_chunks_by_prompt()!
  - Use use_hybrid=True (default) for best accuracy
  - Use use_hybrid=False for faster semantic-only search
  - Adjust semantic_weight (0-1) to balance keyword vs semantic matching
""")
    
    return evaluations


if __name__ == "__main__":
    main()
