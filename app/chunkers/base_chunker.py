from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

    def _ensure_chunk_size(self, chunks: List[str]) -> List[str]:
        result = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                result.append(chunk)
            else:
                # Split large chunks
                result.extend(self._split_large_chunk(chunk))
        return result

    def _split_large_chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks
