import os
import pickle
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from loguru import logger

try:
    import faiss
except ImportError:
    logger.warning("FAISS not installed, vector search will not be available")
    faiss = None

from app.core.config import get_settings

settings = get_settings()


class FAISSIndex:
    def __init__(self, dimension: int = 768, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.index: Optional[faiss.Index] = None
        self.id_map: Dict[int, str] = {}  # FAISS index -> chunk_id
        self.metadata: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata

        self._initialize_index()

    def _initialize_index(self):
        if faiss is None:
            logger.warning("FAISS not available, skipping index initialization")
            return

        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")

        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                loaded_index = faiss.read_index(index_file)
                # Check if loaded index has compatible dimension
                if loaded_index.d != self.dimension:
                    logger.warning(
                        f"Loaded index dimension ({loaded_index.d}) differs from requested ({self.dimension}). "
                        f"Creating new index."
                    )
                    self._create_new_index()
                    return
                    
                self.index = loaded_index
                with open(metadata_file, "rb") as f:
                    data = pickle.load(f)
                    self.id_map = data["id_map"]
                    self.metadata = data["metadata"]
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors (dim={self.index.d})")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    def _create_new_index(self):
        if faiss is None:
            return

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = {}
        self.metadata = {}
        logger.info(f"Created new FAISS index with dimension {self.dimension}")

    def add_embedding(
        self, chunk_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ):
        if self.index is None:
            return

        # Normalize embedding for cosine similarity
        embedding = embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(embedding)

        # Add to index
        faiss_id = self.index.ntotal
        self.index.add(embedding)
        self.id_map[faiss_id] = chunk_id
        if metadata:
            self.metadata[chunk_id] = metadata

    def add_embeddings_batch(
        self, chunk_ids: List[str], embeddings: np.ndarray, metadata_list: Optional[List[Dict[str, Any]]] = None,
        already_normalized: bool = False
    ):
        """
        Add multiple embeddings to the index in a single batch operation.
        
        Args:
            chunk_ids: List of chunk IDs
            embeddings: NumPy array of embeddings (shape: [n_chunks, dimension])
            metadata_list: Optional list of metadata dicts for each chunk
            already_normalized: If True, skip normalization (for pre-normalized embeddings)
        """
        if self.index is None:
            return

        # Convert to float32 if needed (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        
        # Normalize embeddings only if not already normalized
        if not already_normalized:
            faiss.normalize_L2(embeddings)

        # Add to index in single operation (faster than individual adds)
        start_id = self.index.ntotal
        self.index.add(embeddings)

        # Batch update mappings using dict comprehension (faster than loop)
        new_id_map = {start_id + i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        self.id_map.update(new_id_map)
        
        # Track duplicates and update metadata
        existing_chunks = set(self.metadata.keys())
        new_chunks = 0
        duplicate_chunks = 0
        
        if metadata_list:
            # Batch update metadata
            new_metadata = {}
            for i, (chunk_id, meta) in enumerate(zip(chunk_ids, metadata_list)):
                if chunk_id in existing_chunks:
                    duplicate_chunks += 1
                else:
                    new_chunks += 1
                new_metadata[chunk_id] = meta
            self.metadata.update(new_metadata)
        else:
            new_chunks = len(chunk_ids)

        logger.info(
            f"Added {len(chunk_ids)} embeddings to FAISS index "
            f"({new_chunks} new, {duplicate_chunks} duplicates)"
        )

    def search(
        self, query_embedding: np.ndarray, k: int = 5, deduplicate: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Search with extra results if deduplication is needed
        search_k = k * 3 if deduplicate else k  # Get 3x results for deduplication
        search_k = min(search_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)

        # Build results
        if deduplicate:
            # Track seen chunk_ids to remove duplicates (keep best score)
            seen_chunks = set()
            results = []
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self.id_map:
                    chunk_id = self.id_map[idx]
                    
                    # Skip if already seen (duplicate)
                    if chunk_id in seen_chunks:
                        continue
                    
                    seen_chunks.add(chunk_id)
                    metadata = self.metadata.get(chunk_id, {})
                    results.append((chunk_id, float(dist), metadata))
                    
                    # Stop when we have enough unique results
                    if len(results) >= k:
                        break
        else:
            # No deduplication - return all results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self.id_map:
                    chunk_id = self.id_map[idx]
                    metadata = self.metadata.get(chunk_id, {})
                    results.append((chunk_id, float(dist), metadata))

        return results

    def save(self):
        if self.index is None:
            return

        os.makedirs(self.index_path, exist_ok=True)

        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")

        try:
            faiss.write_index(self.index, index_file)
            with open(metadata_file, "wb") as f:
                pickle.dump({"id_map": self.id_map, "metadata": self.metadata}, f)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def clear(self):
        self._create_new_index()
        logger.info("Cleared FAISS index")
    
    def get_chunk_indices(self, chunk_id: str) -> List[int]:
        return [idx for idx, cid in self.id_map.items() if cid == chunk_id]
    
    def has_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self.metadata
    
    def get_stats(self) -> Dict[str, Any]:
        if self.index is None:
            return {"total_vectors": 0, "unique_chunks": 0, "duplicates": 0}
        
        total_vectors = self.index.ntotal
        unique_chunks = len(set(self.id_map.values()))
        duplicates = total_vectors - unique_chunks
        
        return {
            "total_vectors": total_vectors,
            "unique_chunks": unique_chunks,
            "duplicates": duplicates,
            "duplicate_rate": f"{(duplicates / total_vectors * 100):.1f}%" if total_vectors > 0 else "0%"
        }
    
    def get_all_chunks_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        results = []
        for chunk_id, meta in self.metadata.items():
            if meta.get("document_id") == document_id or chunk_id.startswith(f"{document_id}_"):
                results.append({
                    "chunk_id": chunk_id,
                    "text": meta.get("text", ""),
                    "document_id": meta.get("document_id", document_id),
                    **meta
                })
        return results
    
    def hybrid_search(
        self,
        query: str,
        embedding_model,
        k: int = 5,
        document_id: Optional[str] = None,
        semantic_weight: float = 0.7,
        use_bm25: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        try:
            from app.search.hybrid_search import HybridSearcher
            
            # Create hybrid searcher
            searcher = HybridSearcher(
                embedding_model=embedding_model,
                faiss_index=self,
                semantic_weight=semantic_weight
            )
            
            # Build BM25 index from chunks if filtering by document
            if use_bm25:
                if document_id:
                    chunks = self.get_all_chunks_for_document(document_id)
                else:
                    # Get all chunks from metadata
                    chunks = [
                        {"chunk_id": cid, "text": meta.get("text", ""), **meta}
                        for cid, meta in self.metadata.items()
                    ]
                
                if chunks:
                    searcher.build_bm25_index(chunks)
            
            # Perform hybrid search
            results = searcher.search(
                query=query,
                top_k=k,
                document_id=document_id,
                use_hybrid=use_bm25 and len(self.metadata) > 0
            )
            
            # Convert SearchResult to tuple format
            return [
                (r.chunk_id, r.hybrid_score, r.metadata)
                for r in results
            ]
            
        except ImportError as e:
            logger.warning(f"Hybrid search not available: {e}. Using semantic search.")
            # Fallback to semantic search
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
            return self.search(query_embedding, k=k, deduplicate=True)


# Singleton instance
_faiss_index: Optional[FAISSIndex] = None


def get_faiss_index(dimension: int = 768) -> FAISSIndex:
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = FAISSIndex(dimension=dimension)
    return _faiss_index
