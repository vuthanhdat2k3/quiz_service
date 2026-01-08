"""
Query Relevance Detector - Kiểm tra độ liên quan của query với document.

Phát hiện khi query của người dùng không liên quan đến tài liệu
để tránh sinh câu hỏi từ các chunks không phù hợp.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class RelevanceResult:
    """Kết quả đánh giá độ liên quan"""
    is_relevant: bool  # Query có liên quan không?
    confidence: float  # Độ tin cậy (0-1)
    relevance_score: float  # Điểm liên quan tổng hợp (0-1)
    strategy: str  # Chiến lược đề xuất: "search", "hybrid", "representative"
    warning_message: Optional[str]  # Cảnh báo cho người dùng (nếu có)
    details: Dict[str, Any]  # Chi tiết phân tích


class QueryRelevanceDetector:
    """
    Phát hiện độ liên quan giữa query và document content.
    
    Sử dụng nhiều phương pháp:
    1. Semantic similarity với top chunks
    2. Semantic similarity với document overview
    3. Token overlap analysis
    4. Score distribution analysis
    """
    
    def __init__(
        self,
        high_relevance_threshold: float = 0.65,
        low_relevance_threshold: float = 0.35,
        min_top_score: float = 0.4,
        max_score_variance: float = 0.15
    ):
        """
        Args:
            high_relevance_threshold: Ngưỡng để xem là "highly relevant" (>= 0.65)
            low_relevance_threshold: Ngưỡng để xem là "low relevant" (< 0.35)
            min_top_score: Điểm tối thiểu của chunk top 1 để xem là relevant
            max_score_variance: Phương sai tối đa của scores (nếu cao = không rõ ràng)
        """
        self.high_relevance_threshold = high_relevance_threshold
        self.low_relevance_threshold = low_relevance_threshold
        self.min_top_score = min_top_score
        self.max_score_variance = max_score_variance
    
    def analyze_query_relevance(
        self,
        query: str,
        query_embedding: np.ndarray,
        search_results: List[Tuple[str, float, Dict]],
        document_overview: Optional[str] = None,
        chunks: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        embedding_model = None
    ) -> RelevanceResult:
        """
        Phân tích độ liên quan của query với document.
        
        Args:
            query: Query string từ người dùng
            query_embedding: Embedding của query
            search_results: Kết quả hybrid search [(chunk_id, score, metadata), ...]
            document_overview: Tóm tắt/overview của document (optional)
            chunks: List of all chunks (optional, for deeper analysis)
            embeddings: Embeddings của tất cả chunks (optional)
            embedding_model: Model để encode text (optional)
        
        Returns:
            RelevanceResult với phân tích chi tiết
        """
        logger.info(f"🔍 Analyzing query relevance: '{query[:50]}...'")
        
        details = {}
        relevance_scores = []
        
        # === 1. Phân tích Top Search Scores ===
        if search_results and len(search_results) > 0:
            top_scores = [score for _, score, _ in search_results[:5]]
            
            # Score của chunk tốt nhất
            top_score = top_scores[0]
            details["top_score"] = float(top_score)
            
            # Phương sai của top 5 scores (nếu cao = không có chunk nào nổi trội)
            if len(top_scores) > 1:
                score_variance = float(np.var(top_scores))
                details["score_variance"] = score_variance
            else:
                score_variance = 0
            
            # Đánh giá: top_score càng cao càng tốt, variance càng thấp càng rõ ràng
            score_relevance = top_score
            
            # Penalty nếu variance quá cao (không có chunk nào nổi bật)
            if score_variance > self.max_score_variance:
                score_relevance *= 0.8
                details["high_variance_penalty"] = True
            
            relevance_scores.append(("top_score", score_relevance, 0.4))  # weight 0.4
        
        # === 2. Semantic Similarity với Document Overview ===
        if document_overview and embedding_model:
            try:
                overview_embedding = embedding_model.encode([document_overview], convert_to_numpy=True)[0]
                
                # Normalize
                query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
                overview_norm = overview_embedding / (np.linalg.norm(overview_embedding) + 1e-9)
                
                # Cosine similarity
                overview_similarity = float(np.dot(query_norm, overview_norm))
                details["overview_similarity"] = overview_similarity
                
                relevance_scores.append(("overview", overview_similarity, 0.3))  # weight 0.3
                
            except Exception as e:
                logger.warning(f"Could not compute overview similarity: {e}")
        
        # === 3. Token Overlap Analysis ===
        token_overlap = self._compute_token_overlap(query, search_results)
        details["token_overlap"] = token_overlap
        relevance_scores.append(("token_overlap", token_overlap, 0.15))  # weight 0.15
        
        # === 4. Score Distribution Analysis ===
        if search_results and len(search_results) >= 5:
            # Kiểm tra xem có gap lớn giữa top chunks và các chunks khác không
            scores = [score for _, score, _ in search_results[:10]]
            score_drop = scores[0] - np.mean(scores[1:5]) if len(scores) > 1 else 0
            
            # Gap lớn = có chunk rõ ràng liên quan nhất
            distribution_score = min(1.0, score_drop * 3)  # Normalize
            details["score_drop"] = float(score_drop)
            details["distribution_score"] = distribution_score
            
            relevance_scores.append(("distribution", distribution_score, 0.15))  # weight 0.15
        
        # === Tính toán Relevance Score tổng hợp ===
        if relevance_scores:
            # Weighted average
            total_weight = sum(w for _, _, w in relevance_scores)
            weighted_sum = sum(score * w for _, score, w in relevance_scores)
            final_relevance = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            final_relevance = 0.5  # Default neutral score
        
        details["component_scores"] = {
            name: float(score) for name, score, _ in relevance_scores
        }
        details["final_relevance"] = float(final_relevance)
        
        # === Quyết định Strategy và Warning ===
        is_relevant, strategy, warning, confidence = self._determine_strategy(
            final_relevance, details
        )
        
        logger.info(
            f"✓ Relevance analysis: score={final_relevance:.3f}, "
            f"strategy={strategy}, relevant={is_relevant}"
        )
        
        return RelevanceResult(
            is_relevant=is_relevant,
            confidence=confidence,
            relevance_score=final_relevance,
            strategy=strategy,
            warning_message=warning,
            details=details
        )
    
    def _compute_token_overlap(
        self,
        query: str,
        search_results: List[Tuple[str, float, Dict]]
    ) -> float:
        """Tính token overlap giữa query và top chunks."""
        if not search_results:
            return 0.0
        
        # Tokenize query (simple word-based)
        query_tokens = set(query.lower().split())
        
        # Get text from top 3 results
        top_texts = []
        for _, _, meta in search_results[:3]:
            text = meta.get("text", "")
            if text:
                top_texts.append(text.lower())
        
        if not top_texts:
            return 0.0
        
        # Compute overlap
        combined_text = " ".join(top_texts)
        chunk_tokens = set(combined_text.split())
        
        if not query_tokens:
            return 0.0
        
        overlap = len(query_tokens.intersection(chunk_tokens))
        overlap_ratio = overlap / len(query_tokens)
        
        return min(1.0, overlap_ratio)
    
    def _determine_strategy(
        self,
        relevance_score: float,
        details: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str], float]:
        """
        Quyết định strategy dựa trên relevance score.
        
        Returns:
            (is_relevant, strategy, warning_message, confidence)
        """
        top_score = details.get("top_score", 0)
        
        # === HIGH RELEVANCE: Search-based ===
        if relevance_score >= self.high_relevance_threshold and top_score >= self.min_top_score:
            return (
                True,
                "search",
                None,
                0.9
            )
        
        # === MEDIUM RELEVANCE: Hybrid approach ===
        elif relevance_score >= self.low_relevance_threshold:
            warning = (
                "⚠️ Query có độ liên quan trung bình với tài liệu. "
                "Hệ thống sẽ kết hợp cả nội dung liên quan và nội dung đại diện."
            )
            return (
                True,
                "hybrid",
                warning,
                0.6
            )
        
        # === LOW RELEVANCE: Representative mode ===
        else:
            warning = (
                "⚠️ Query có vẻ KHÔNG liên quan đến tài liệu này. "
                "Hệ thống sẽ tự động sinh câu hỏi từ các phần quan trọng của tài liệu "
                "thay vì dựa vào query của bạn."
            )
            return (
                False,
                "representative",
                warning,
                0.3
            )
    
    def quick_check(
        self,
        query: str,
        top_search_score: float
    ) -> bool:
        """
        Quick check nhanh chỉ dựa vào top search score.
        Dùng khi không có đủ dữ liệu cho phân tích đầy đủ.
        
        Returns:
            True nếu có vẻ relevant, False nếu không
        """
        return top_search_score >= self.min_top_score


def create_relevance_detector(
    strict: bool = False
) -> QueryRelevanceDetector:
    """
    Factory function để tạo QueryRelevanceDetector.
    
    Args:
        strict: Nếu True, sử dụng ngưỡng chặt chẽ hơn
    
    Returns:
        QueryRelevanceDetector instance
    """
    if strict:
        return QueryRelevanceDetector(
            high_relevance_threshold=0.75,
            low_relevance_threshold=0.45,
            min_top_score=0.5,
            max_score_variance=0.12
        )
    else:
        return QueryRelevanceDetector(
            high_relevance_threshold=0.65,
            low_relevance_threshold=0.35,
            min_top_score=0.4,
            max_score_variance=0.15
        )
