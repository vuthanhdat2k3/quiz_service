from pathlib import Path
from typing import Optional

from loguru import logger

from app.chunkers.base_chunker import BaseChunker
from app.chunkers.markdown_chunker import MarkdownChunkerV2
from app.core.config import get_settings

settings = get_settings()


class ChunkerFactory:

    @classmethod
    def get_chunker(
        cls,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> MarkdownChunkerV2:
        
        file_ext = Path(file_path).suffix.lower()

        # Use max_tokens from parameter or config
        if max_tokens is None:
            max_tokens = settings.MAX_TOKENS_PER_CHUNK

        # All documents are parsed to markdown, use MarkdownChunkerV2 for all
        logger.info(f"Using MarkdownChunkerV2 for {file_ext} with max_tokens={max_tokens}")
        return MarkdownChunkerV2(max_tokens=max_tokens)
