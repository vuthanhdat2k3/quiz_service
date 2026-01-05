"""
Embedding Cache and Optimization Utilities

This module provides caching and optimization for embedding operations to speed up
document processing in the quiz generation pipeline.
"""

import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
from loguru import logger


@dataclass
class CacheEntry:
    """Cache entry for embeddings."""
    embedding: np.ndarray
    timestamp: float
    access_count: int = 0


class EmbeddingCache:
    """
    LRU cache for embeddings to avoid recomputing the same text embeddings.
    
    This is particularly useful when:
    - Processing similar documents
    - Re-running pipelines with same content
    - Caching query embeddings for repeated searches
    """
    
    def __init__(
        self, 
        max_size: int = 10000,
        ttl_seconds: float = 3600  # 1 hour default TTL
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _compute_key(self, text: str, model_name: str = "") -> str:
        """Compute cache key from text and model name."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model_name: str = "") -> Optional[np.ndarray]:
        """Get embedding from cache if exists and not expired."""
        key = self._compute_key(text, model_name)
        
        if key in self._cache:
            entry = self._cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.access_count += 1
            self._hits += 1
            
            return entry.embedding.copy()
        
        self._misses += 1
        return None
    
    def put(
        self, 
        text: str, 
        embedding: np.ndarray, 
        model_name: str = ""
    ) -> None:
        """Store embedding in cache."""
        key = self._compute_key(text, model_name)
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = CacheEntry(
            embedding=embedding.copy(),
            timestamp=time.time()
        )
    
    def get_batch(
        self, 
        texts: List[str], 
        model_name: str = ""
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Get batch of embeddings from cache.
        
        Returns:
            Tuple of (cached_embeddings, uncached_indices)
            - cached_embeddings: List of embeddings for cached texts
            - uncached_indices: Indices of texts that need to be computed
        """
        cached = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, model_name)
            if embedding is not None:
                cached.append((i, embedding))
            else:
                uncached_indices.append(i)
        
        return cached, uncached_indices
    
    def put_batch(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        model_name: str = ""
    ) -> None:
        """Store batch of embeddings in cache."""
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding, model_name)
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2%}"
        }


# Global cache instance
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(
    max_size: int = 10000,
    ttl_seconds: float = 3600
) -> EmbeddingCache:
    """Get or create global embedding cache."""
    global _embedding_cache
    
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size, ttl_seconds)
    
    return _embedding_cache


async def encode_with_cache(
    texts: List[str],
    embedding_model,
    model_name: str = "",
    batch_size: int = 64,
    use_cache: bool = True
) -> np.ndarray:
    """
    Encode texts with caching for repeated texts.
    
    Args:
        texts: List of texts to encode
        embedding_model: SentenceTransformer model
        model_name: Model name for cache key
        batch_size: Batch size for encoding
        use_cache: Whether to use caching
    
    Returns:
        numpy array of embeddings
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    if not use_cache:
        # Direct encoding without cache
        def _encode():
            return embedding_model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=True
            )
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _encode)
    
    # Get cache
    cache = get_embedding_cache()
    
    # Check cache for each text
    cached, uncached_indices = cache.get_batch(texts, model_name)
    
    # If all cached, return immediately
    if not uncached_indices:
        logger.debug(f"All {len(texts)} embeddings retrieved from cache")
        embeddings = np.zeros((len(texts), cached[0][1].shape[0]))
        for idx, emb in cached:
            embeddings[idx] = emb
        return embeddings
    
    # Encode uncached texts
    uncached_texts = [texts[i] for i in uncached_indices]
    
    def _encode_uncached():
        return embedding_model.encode(
            uncached_texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=True
        )
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        uncached_embeddings = await loop.run_in_executor(executor, _encode_uncached)
    
    # Store in cache
    cache.put_batch(uncached_texts, uncached_embeddings, model_name)
    
    # Build result array
    dim = uncached_embeddings.shape[1] if len(uncached_embeddings) > 0 else (
        cached[0][1].shape[0] if cached else 768
    )
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)
    
    # Fill cached embeddings
    for idx, emb in cached:
        embeddings[idx] = emb
    
    # Fill uncached embeddings
    for i, idx in enumerate(uncached_indices):
        embeddings[idx] = uncached_embeddings[i]
    
    logger.debug(
        f"Embeddings: {len(cached)} cached, {len(uncached_indices)} computed. "
        f"Cache stats: {cache.get_stats()}"
    )
    
    return embeddings


class EmbeddingBatcher:
    """
    Utility for batching embedding requests efficiently.
    
    Useful for scenarios where embeddings are requested one at a time
    but should be computed in batches for efficiency.
    """
    
    def __init__(
        self,
        embedding_model,
        batch_size: int = 64,
        max_wait_time: float = 0.1  # Max time to wait for batch to fill
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self._pending: List[Tuple[str, Any]] = []  # (text, future)
        self._last_batch_time = time.time()
    
    async def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text, batching with other pending requests."""
        import asyncio
        
        future = asyncio.Future()
        self._pending.append((text, future))
        
        # Check if we should process batch
        should_process = (
            len(self._pending) >= self.batch_size or
            time.time() - self._last_batch_time > self.max_wait_time
        )
        
        if should_process:
            await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """Process all pending requests in a batch."""
        if not self._pending:
            return
        
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Collect pending
        batch = self._pending[:]
        self._pending = []
        self._last_batch_time = time.time()
        
        texts = [t for t, _ in batch]
        futures = [f for _, f in batch]
        
        # Encode batch
        def _encode():
            return self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            embeddings = await loop.run_in_executor(executor, _encode)
        
        # Resolve futures
        for future, embedding in zip(futures, embeddings):
            if not future.done():
                future.set_result(embedding)
