
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed, falling back to semantic-only search")


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    semantic_score: float
    bm25_score: float
    hybrid_score: float
    section_path: List[str]
    metadata: Dict[str, Any]


class HybridSearcher:
    
    
    def __init__(
        self,
        embedding_model,
        faiss_index,
        semantic_weight: float = 0.7
    ):
        
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.semantic_weight = semantic_weight
        self._bm25_index = None
        self._chunks_cache = None
        
    def _tokenize(self, text: str) -> List[str]:
        
        if not text:
            return []
        
        # Lowercase
        text = text.lower()
        
        # Replace common HTML entities
        text = text.replace("&#x26;", "&").replace("&#x3c;", "<").replace("&#x3e;", ">")
        
        # Split on whitespace and punctuation, keep alphanumeric and Vietnamese chars
        tokens = re.findall(r'\b[\w\u00C0-\u024F\u1E00-\u1EFF]+\b', text, re.UNICODE)
        
        # Filter short tokens (noise)
        tokens = [t for t in tokens if len(t) >= 2]
        
        return tokens
    
    def build_bm25_index(self, chunks: List[Dict[str, Any]]):
        
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, skipping index build")
            return
        
        self._chunks_cache = chunks
        
        # Tokenize all chunks
        corpus = [self._tokenize(chunk.get("text", "")) for chunk in chunks]
        
        # Build BM25 index
        self._bm25_index = BM25Okapi(corpus)
        
        logger.info(f"Built BM25 index with {len(chunks)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        use_hybrid: bool = True
    ) -> List[SearchResult]:
        
        if not query:
            return []
        
        # Get semantic search results
        semantic_results = self._semantic_search(query, document_id)
        
        # If no BM25 or not using hybrid, return semantic results
        if not use_hybrid or not BM25_AVAILABLE or self._bm25_index is None:
            return self._format_semantic_results(semantic_results, top_k)
        
        # Get BM25 scores
        bm25_scores = self._bm25_search(query, document_id)
        
        # Combine scores
        hybrid_results = self._combine_scores(semantic_results, bm25_scores, document_id)
        
        # Sort by hybrid score and return top_k
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return hybrid_results[:top_k]
    
    def _semantic_search(
        self, 
        query: str, 
        document_id: Optional[str] = None
    ) -> List[Tuple[str, float, Dict]]:
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Search with extra results if filtering by document
        search_k = self.faiss_index.index.ntotal if self.faiss_index.index else 100
        results = self.faiss_index.search(query_embedding, k=search_k, deduplicate=True)
        
        # Filter by document_id if specified
        if document_id:
            results = [
                (cid, score, meta) 
                for cid, score, meta in results 
                if meta.get("document_id") == document_id
            ]
        
        return results
    
    def _bm25_search(
        self, 
        query: str, 
        document_id: Optional[str] = None
    ) -> Dict[str, float]:
        
        if self._bm25_index is None or self._chunks_cache is None:
            return {}
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1.0
        
        # Build score mapping
        bm25_scores = {}
        for i, chunk in enumerate(self._chunks_cache):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            
            # Filter by document_id if specified
            if document_id:
                chunk_doc_id = chunk.get("document_id") or chunk_id.rsplit("_chunk_", 1)[0]
                if chunk_doc_id != document_id:
                    continue
            
            bm25_scores[chunk_id] = scores[i] / max_score
        
        return bm25_scores
    
    def _combine_scores(
        self,
        semantic_results: List[Tuple[str, float, Dict]],
        bm25_scores: Dict[str, float],
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        
        # Create mapping from semantic results
        semantic_map = {cid: (score, meta) for cid, score, meta in semantic_results}
        
        # Get all unique chunk_ids
        all_chunk_ids = set(semantic_map.keys()) | set(bm25_scores.keys())
        
        # Combine scores
        results = []
        for chunk_id in all_chunk_ids:
            semantic_score = semantic_map.get(chunk_id, (0, {}))[0]
            metadata = semantic_map.get(chunk_id, (0, {}))[1]
            bm25_score = bm25_scores.get(chunk_id, 0)
            
            # Get text from metadata or chunks cache
            text = metadata.get("text", "")
            if not text and self._chunks_cache:
                for chunk in self._chunks_cache:
                    if chunk.get("chunk_id") == chunk_id:
                        text = chunk.get("text", "")[:200]
                        break
            
            # Compute hybrid score
            hybrid_score = (
                self.semantic_weight * semantic_score +
                (1 - self.semantic_weight) * bm25_score
            )
            
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=text,
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                hybrid_score=hybrid_score,
                section_path=metadata.get("section_path", []),
                metadata=metadata
            ))
        
        return results
    
    def _format_semantic_results(
        self, 
        semantic_results: List[Tuple[str, float, Dict]], 
        top_k: int
    ) -> List[SearchResult]:
        
        results = []
        for chunk_id, score, metadata in semantic_results[:top_k]:
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=metadata.get("text", ""),
                semantic_score=score,
                bm25_score=0,
                hybrid_score=score,
                section_path=metadata.get("section_path", []),
                metadata=metadata
            ))
        return results


def create_hybrid_searcher(
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    semantic_weight: float = 0.7
) -> HybridSearcher:
    
    from sentence_transformers import SentenceTransformer
    from app.database.faiss_index import get_faiss_index
    
    # Load embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Get FAISS index
    faiss_index = get_faiss_index(dimension=384)  # all-MiniLM-L6-v2 is 384-dim
    
    return HybridSearcher(
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        semantic_weight=semantic_weight
    )
