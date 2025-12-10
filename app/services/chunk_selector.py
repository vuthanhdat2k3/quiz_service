

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ChunkSelectionConfig:
    max_total_tokens: int = 3000  # Max tokens to send to LLM
    tokens_per_question: int = 150  # Estimated tokens needed per question
    min_chunks: int = 3
    max_chunks: int = 30
    include_adjacent: bool = True  # Include chunks before/after search results
    semantic_weight: float = 0.7  # Weight for semantic vs BM25 in hybrid search


class ChunkSelector:
    
    def __init__(self, config: ChunkSelectionConfig = None):
        self.config = config or ChunkSelectionConfig()
    
    def calculate_chunks_needed(self, num_questions: int) -> int:
        
        if num_questions <= 5:
            multiplier = 1.5
        elif num_questions <= 15:
            multiplier = 1.3
        else:
            multiplier = 1.2
        
        chunks_by_questions = int(num_questions * multiplier)
        
        # Also calculate based on token budget
        # Assuming avg chunk is ~100 tokens
        avg_chunk_tokens = 100
        chunks_by_tokens = self.config.max_total_tokens // avg_chunk_tokens
        
        # Take minimum of both constraints
        num_chunks = min(chunks_by_questions, chunks_by_tokens)
        
        # Apply bounds
        num_chunks = max(self.config.min_chunks, min(num_chunks, self.config.max_chunks))
        
        logger.info(f"Calculated {num_chunks} chunks needed for {num_questions} questions")
        return num_chunks
    
    def select_by_search(
        self,
        query: str,
        chunks: List[Any],
        chunk_ids: List[str],
        embeddings: np.ndarray,
        faiss_index,
        embedding_model,
        num_chunks: int,
        document_id: str = None
    ) -> List[Dict[str, Any]]:
        
        logger.info(f"Selecting {num_chunks} chunks by search for query: '{query[:50]}...'")
        
        # Run hybrid search
        search_results = faiss_index.hybrid_search(
            query=query,
            embedding_model=embedding_model,
            k=num_chunks * 2,  # Get extra for filtering
            document_id=document_id,
            semantic_weight=self.config.semantic_weight,
            use_bm25=True
        )
        
        # Build chunk index map
        chunk_map = {cid: (idx, chunk) for idx, (cid, chunk) in enumerate(zip(chunk_ids, chunks))}
        
        # Collect selected chunks with scores
        selected = []
        selected_ids = set()
        total_tokens = 0
        
        # Add search results
        for chunk_id, score, metadata in search_results:
            if chunk_id in selected_ids:
                continue
            
            if chunk_id not in chunk_map:
                continue
                
            idx, chunk = chunk_map[chunk_id]
            token_count = chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.text.split())
            
            # Check token budget
            if total_tokens + token_count > self.config.max_total_tokens:
                break
            
            selected.append({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "text": chunk.text,
                "token_count": token_count,
                "section_path": chunk.section_path if hasattr(chunk, 'section_path') else [],
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {},
                "score": score,
                "selection_method": "search"
            })
            selected_ids.add(chunk_id)
            total_tokens += token_count
            
            if len(selected) >= num_chunks:
                break
        
        # Expand with adjacent chunks if enabled and have room
        if self.config.include_adjacent and len(selected) < num_chunks:
            selected = self._expand_with_adjacent(
                selected, chunks, chunk_ids, selected_ids, total_tokens
            )
        
        logger.info(f"Selected {len(selected)} chunks ({total_tokens} tokens) by search")
        return selected
    
    def select_representative(
        self,
        chunks: List[Any],
        chunk_ids: List[str],
        embeddings: np.ndarray,
        num_chunks: int
    ) -> List[Dict[str, Any]]:
        
        logger.info(f"Selecting {num_chunks} representative chunks")
        
        if len(chunks) <= num_chunks:
            # Return all chunks if fewer than needed
            return self._format_all_chunks(chunks, chunk_ids)
        
        # Compute global centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Score all chunks
        chunk_scores = []
        for idx, chunk in enumerate(chunks):
            score = self._compute_representative_score(
                chunk, idx, len(chunks), embeddings, centroid
            )
            chunk_scores.append((idx, score, chunk))
        
        # Use MMR for diversity-aware selection
        selected = self._mmr_selection(
            chunk_scores, embeddings, num_chunks, lambda_param=0.7
        )
        
        # Format results
        results = []
        total_tokens = 0
        
        for idx, score in selected:
            chunk = chunks[idx]
            token_count = chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.text.split())
            
            # Check token budget
            if total_tokens + token_count > self.config.max_total_tokens:
                break
            
            results.append({
                "chunk_id": chunk_ids[idx],
                "chunk_index": idx,
                "text": chunk.text,
                "token_count": token_count,
                "section_path": chunk.section_path if hasattr(chunk, 'section_path') else [],
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {},
                "score": score,
                "selection_method": "representative"
            })
            total_tokens += token_count
        
        logger.info(f"Selected {len(results)} representative chunks ({total_tokens} tokens)")
        return results
    
    def _compute_representative_score(
        self,
        chunk,
        idx: int,
        total_chunks: int,
        embeddings: np.ndarray,
        centroid: np.ndarray
    ) -> float:
        
        score = 0.0
        
        # 1. Centrality score (0-3 points)
        distance = np.linalg.norm(embeddings[idx] - centroid)
        max_dist = np.max([np.linalg.norm(e - centroid) for e in embeddings]) + 1e-9
        centrality = 1 - (distance / max_dist)
        score += centrality * 3
        
        # 2. Content quality (0-3 points)
        token_count = chunk.token_count if hasattr(chunk, 'token_count') else 50
        if 50 <= token_count <= 150:
            score += 2  # Ideal length
        elif 30 <= token_count <= 180:
            score += 1
        
        # Has heading (likely a key topic)
        if hasattr(chunk, 'metadata') and chunk.metadata.get("heading"):
            score += 1
        
        # 3. Position score (0-2 points)
        relative_pos = idx / total_chunks
        if 0.15 <= relative_pos <= 0.85:
            score += 2  # Middle content
        elif 0.05 <= relative_pos <= 0.95:
            score += 1  # Not too extreme
        
        # 4. Section variety bonus (prefer chunks that start sections)
        if hasattr(chunk, 'metadata'):
            if chunk.metadata.get("is_section_start", False):
                score += 1
        
        return score
    
    def _mmr_selection(
        self,
        chunk_scores: List[Tuple[int, float, Any]],
        embeddings: np.ndarray,
        num_select: int,
        lambda_param: float = 0.7
    ) -> List[Tuple[int, float]]:
        
        selected_indices = []
        selected_scores = []
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normalized = embeddings / norms
        
        while len(selected_indices) < num_select and len(selected_indices) < len(chunk_scores):
            best_score = -float('inf')
            best_idx = -1
            
            for idx, relevance, chunk in chunk_scores:
                if idx in selected_indices:
                    continue
                
                # Compute diversity penalty
                diversity_penalty = 0
                if selected_indices:
                    similarities = np.dot(normalized[selected_indices], normalized[idx])
                    diversity_penalty = np.max(similarities)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty * 5
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                # Find original relevance score
                orig_score = next(s for i, s, _ in chunk_scores if i == best_idx)
                selected_scores.append((best_idx, orig_score))
        
        return selected_scores
    
    def _expand_with_adjacent(
        self,
        selected: List[Dict],
        chunks: List[Any],
        chunk_ids: List[str],
        selected_ids: set,
        current_tokens: int
    ) -> List[Dict]:
        
        expanded = list(selected)
        total_tokens = current_tokens
        
        # Get indices of selected chunks
        selected_indices = {s["chunk_index"] for s in selected}
        
        for sel in selected:
            idx = sel["chunk_index"]
            section_path = sel.get("section_path", [])
            
            # Check previous chunk
            if idx > 0 and (idx - 1) not in selected_indices:
                prev_chunk = chunks[idx - 1]
                prev_section = prev_chunk.section_path if hasattr(prev_chunk, 'section_path') else []
                
                # Only add if same section
                if prev_section == section_path:
                    token_count = prev_chunk.token_count if hasattr(prev_chunk, 'token_count') else 50
                    if total_tokens + token_count <= self.config.max_total_tokens:
                        prev_id = chunk_ids[idx - 1]
                        if prev_id not in selected_ids:
                            expanded.append({
                                "chunk_id": prev_id,
                                "chunk_index": idx - 1,
                                "text": prev_chunk.text,
                                "token_count": token_count,
                                "section_path": prev_section,
                                "metadata": prev_chunk.metadata if hasattr(prev_chunk, 'metadata') else {},
                                "score": sel["score"] * 0.8,  # Slightly lower score
                                "selection_method": "context_expansion"
                            })
                            selected_ids.add(prev_id)
                            selected_indices.add(idx - 1)
                            total_tokens += token_count
            
            # Check next chunk
            if idx < len(chunks) - 1 and (idx + 1) not in selected_indices:
                next_chunk = chunks[idx + 1]
                next_section = next_chunk.section_path if hasattr(next_chunk, 'section_path') else []
                
                if next_section == section_path:
                    token_count = next_chunk.token_count if hasattr(next_chunk, 'token_count') else 50
                    if total_tokens + token_count <= self.config.max_total_tokens:
                        next_id = chunk_ids[idx + 1]
                        if next_id not in selected_ids:
                            expanded.append({
                                "chunk_id": next_id,
                                "chunk_index": idx + 1,
                                "text": next_chunk.text,
                                "token_count": token_count,
                                "section_path": next_section,
                                "metadata": next_chunk.metadata if hasattr(next_chunk, 'metadata') else {},
                                "score": sel["score"] * 0.8,
                                "selection_method": "context_expansion"
                            })
                            selected_ids.add(next_id)
                            selected_indices.add(idx + 1)
                            total_tokens += token_count
        
        # Sort by chunk_index for proper reading order
        expanded.sort(key=lambda x: x["chunk_index"])
        
        return expanded
    
    def _format_all_chunks(
        self,
        chunks: List[Any],
        chunk_ids: List[str]
    ) -> List[Dict[str, Any]]:
        
        return [
            {
                "chunk_id": chunk_ids[idx],
                "chunk_index": idx,
                "text": chunk.text,
                "token_count": chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.text.split()),
                "section_path": chunk.section_path if hasattr(chunk, 'section_path') else [],
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {},
                "score": 1.0,
                "selection_method": "all"
            }
            for idx, chunk in enumerate(chunks)
        ]
    
    def ensure_section_coverage(
        self,
        selected: List[Dict],
        chunks: List[Any],
        chunk_ids: List[str],
        embeddings: np.ndarray,
        min_coverage: float = 0.7
    ) -> List[Dict]:
        
        # Get all sections
        all_sections = set()
        section_chunks = {}  # section -> list of chunk indices
        
        for idx, chunk in enumerate(chunks):
            section = tuple(chunk.section_path) if hasattr(chunk, 'section_path') else ()
            all_sections.add(section)
            if section not in section_chunks:
                section_chunks[section] = []
            section_chunks[section].append(idx)
        
        # Get covered sections
        covered_sections = set()
        for sel in selected:
            section = tuple(sel.get("section_path", []))
            covered_sections.add(section)
        
        coverage = len(covered_sections) / len(all_sections) if all_sections else 1.0
        
        if coverage >= min_coverage:
            return selected
        
        logger.info(f"Section coverage {coverage:.1%} below {min_coverage:.1%}, adding more chunks")
        
        # Find uncovered sections
        uncovered = all_sections - covered_sections
        selected_ids = {s["chunk_id"] for s in selected}
        total_tokens = sum(s["token_count"] for s in selected)
        
        # Add one chunk from each uncovered section
        for section in uncovered:
            if section not in section_chunks:
                continue
            
            # Pick the most central chunk from this section
            section_indices = section_chunks[section]
            best_idx = section_indices[0]
            best_centrality = -1
            
            centroid = np.mean(embeddings, axis=0)
            for idx in section_indices:
                distance = np.linalg.norm(embeddings[idx] - centroid)
                centrality = 1.0 / (1 + distance)
                if centrality > best_centrality:
                    best_centrality = centrality
                    best_idx = idx
            
            chunk = chunks[best_idx]
            chunk_id = chunk_ids[best_idx]
            
            if chunk_id in selected_ids:
                continue
            
            token_count = chunk.token_count if hasattr(chunk, 'token_count') else 50
            if total_tokens + token_count > self.config.max_total_tokens:
                break
            
            selected.append({
                "chunk_id": chunk_id,
                "chunk_index": best_idx,
                "text": chunk.text,
                "token_count": token_count,
                "section_path": list(section),
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {},
                "score": best_centrality,
                "selection_method": "section_coverage"
            })
            selected_ids.add(chunk_id)
            total_tokens += token_count
        
        return selected
