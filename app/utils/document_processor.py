import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from app.chunkers.chunker_factory import ChunkerFactory
from app.chunkers.markdown_chunker import MarkdownChunker
from app.parsers.parser_factory import ParserFactory


class DocumentProcessor:
    def __init__(
        self,
        output_dir: str = "output",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens

        # Create subdirectories
        self.markdown_dir = self.output_dir / "markdown"
        self.chunks_dir = self.output_dir / "chunks"
        self.markdown_dir.mkdir(exist_ok=True)
        self.chunks_dir.mkdir(exist_ok=True)

    async def process_document(
        self, file_path: str, save_markdown: bool = True, save_chunks: bool = True
    ) -> Dict[str, any]:
        file_path_obj = Path(file_path)
        base_name = file_path_obj.stem

        logger.info(f"Processing document: {file_path}")

        # Step 1: Parse document to markdown
        markdown_content = await self._parse_to_markdown(file_path)

        # Step 2: Save markdown if requested
        markdown_file = None
        if save_markdown:
            markdown_file = await self._save_markdown(base_name, markdown_content)

        # Step 3: Chunk the markdown content
        chunks = await self._chunk_markdown(file_path, markdown_content)

        # Step 4: Save chunks if requested
        chunks_file = None
        if save_chunks:
            chunks_file = await self._save_chunks(base_name, chunks)

        # Prepare result
        result = {
            "markdown_content": markdown_content,
            "markdown_file": str(markdown_file) if markdown_file else None,
            "chunks": chunks,
            "chunks_file": str(chunks_file) if chunks_file else None,
            "metadata": {
                "original_file": str(file_path_obj.name),
                "markdown_length": len(markdown_content),
                "num_chunks": len(chunks),
                "max_tokens": self.max_tokens,
                "chunk_size": self.chunk_size,  # deprecated
                "chunk_overlap": self.chunk_overlap,  # deprecated
            },
        }

        logger.info(
            f"Document processed successfully: {len(chunks)} chunks created from {len(markdown_content)} chars"
        )

        return result

    async def _parse_to_markdown(self, file_path: str) -> str:
        logger.info(f"Parsing document to markdown: {file_path}")

        parser = ParserFactory.get_parser(file_path)
        if parser is None:
            raise ValueError(f"Unsupported file type: {file_path}")

        markdown_content = await parser.parse(file_path)

        logger.info(
            f"Document parsed successfully: {len(markdown_content)} characters"
        )

        return markdown_content

    async def _save_markdown(self, base_name: str, markdown_content: str) -> Path:
        markdown_file = self.markdown_dir / f"{base_name}.md"

        logger.info(f"Saving markdown to: {markdown_file}")

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"Markdown saved successfully: {markdown_file}")

        return markdown_file

    async def _chunk_markdown(self, file_path: str, markdown_content: str) -> List[Dict]:
        logger.info("Chunking markdown content with advanced MarkdownChunker")

        chunker = ChunkerFactory.get_chunker(
            file_path=file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_tokens=self.max_tokens,
        )

        # Use the advanced chunking method that returns formatted chunks with metadata
        if isinstance(chunker, MarkdownChunker):
            # Process through 3 steps and get formatted output
            _, _, step3_chunks = chunker.process_content(markdown_content)
            file_name = Path(file_path).name
            chunks = chunker.format_chunks_for_output(step3_chunks, file_name)
        else:
            # Fallback to simple chunk method for other chunkers
            raw_chunks = chunker.chunk(markdown_content)
            # Convert to same format for consistency
            chunks = [
                {
                    "chunk_id": str(i),
                    "chunk_text": chunk,
                    "lenght": str(len(chunk)),
                    "metadata": {"position": i + 1, "file_name": Path(file_path).name}
                }
                for i, chunk in enumerate(raw_chunks)
            ]

        logger.info(f"Created {len(chunks)} chunks with metadata")

        return chunks

    async def _save_chunks(self, base_name: str, chunks: List[Dict]) -> Path:
        chunks_file = self.chunks_dir / f"{base_name}_chunks.json"

        logger.info(f"Saving chunks to: {chunks_file}")

        # Chunks already have proper format from MarkdownChunker
        chunk_data = {
            "num_chunks": len(chunks),
            "chunks": chunks,
        }

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Chunks saved successfully: {chunks_file}")

        return chunks_file

    async def process_multiple_documents(
        self, file_paths: List[str], save_markdown: bool = True, save_chunks: bool = True
    ) -> List[Dict[str, any]]:
        results = []

        for file_path in file_paths:
            try:
                result = await self.process_document(
                    file_path=file_path,
                    save_markdown=save_markdown,
                    save_chunks=save_chunks,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results.append(
                    {
                        "error": str(e),
                        "file": file_path,
                    }
                )

        logger.info(
            f"Batch processing completed: {len(results)} documents processed"
        )

        return results
