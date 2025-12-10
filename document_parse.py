

from pathlib import Path
import json
from typing import List, Tuple, Dict, Any
import numpy as np
from loguru import logger
from dataclasses import asdict

from app.parsers import ParserFactory
from app.chunkers.markdown_chunker import MarkdownChunkerV2, MarkdownChunk
from app.database.faiss_index import FAISSIndex


async def document_parse(file_path: str) -> str:
    parser = ParserFactory.get_parser(file_path)
    if not parser:
        raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

    markdown_content = await parser.parse(file_path)
    
    if not markdown_content or len(markdown_content.strip()) < 50:
        raise ValueError("Insufficient content extracted from document")
    
    # Save markdown file
    with open(Path(file_path).with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"✓ Parsed: {len(markdown_content)} characters")
    return markdown_content


def chunk_document(file_path: str, max_tokens: int = 180, min_tokens: int = 12) -> List[MarkdownChunk]:
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    chunker = MarkdownChunkerV2(max_tokens=max_tokens, min_tokens=min_tokens)
    chunks = chunker.parse(markdown_content)
    
    # Save chunks to JSON file
    chunked_file = Path(file_path).with_name(Path(file_path).stem + "_chunks.json")
    with open(chunked_file, "w", encoding="utf-8") as f:
        json.dump([asdict(chunk) for chunk in chunks], f, ensure_ascii=False, indent=2)

    return chunks


def create_faiss_index_for_chunks(
    file_path: str, 
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[FAISSIndex, List[str]]:
    from sentence_transformers import SentenceTransformer
    
    logger.info(f"Creating FAISS index for: {file_path}")
    
    document_id = Path(file_path).stem
    chunks = chunk_document(file_path)
    
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_ids = [f"{document_id}_chunk_{idx}" for idx in range(len(chunks))]
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Generate embeddings
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(chunk_texts, convert_to_numpy=True)
    logger.info(f"Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
    
    # Create FAISS index
    faiss_index = FAISSIndex(dimension=embeddings.shape[1])
    
    # Add embeddings with metadata
    metadata_list = [
        {
            "document_id": document_id,
            "chunk_index": idx,
            "token_count": chunk.token_count,
            "chunk_type": chunk.chunk_type,
            "section_path": chunk.section_path if hasattr(chunk, 'section_path') else [],
            "text": chunk.text[:500]
        }
        for idx, chunk in enumerate(chunks)
    ]
    
    faiss_index.add_embeddings_batch(chunk_ids, embeddings, metadata_list)
    faiss_index.save()
    
    logger.info(f"✓ Created FAISS index with {len(chunk_ids)} chunks")
    return faiss_index, chunk_ids


def search_chunks_by_prompt(
    prompt: str,
    faiss_index: FAISSIndex = None,
    file_path: str = None,
    top_k: int = 5,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    document_id: str = None,
    use_hybrid: bool = True,
    semantic_weight: float = 0.7
) -> List[Dict[str, Any]]:
    from sentence_transformers import SentenceTransformer
    
    logger.info(f"Searching chunks for: '{prompt[:50]}...'")
    
    # Create index if not provided
    if faiss_index is None:
        if file_path is None:
            raise ValueError("Either faiss_index or file_path must be provided")
        faiss_index, _ = create_faiss_index_for_chunks(file_path, embedding_model_name)
    
    # Auto-detect document_id
    if document_id is None and file_path is not None:
        document_id = Path(file_path).stem
    
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Use hybrid search if available
    if use_hybrid:
        try:
            from app.search.hybrid_search import HybridSearcher
            
            searcher = HybridSearcher(
                embedding_model=embedding_model,
                faiss_index=faiss_index,
                semantic_weight=semantic_weight
            )
            
            # Build BM25 index from document chunks
            if file_path:
                chunks = chunk_document(file_path)
                chunks_dict = [
                    {
                        "chunk_id": f"{document_id}_chunk_{idx}",
                        "document_id": document_id,
                        **asdict(chunk)
                    }
                    for idx, chunk in enumerate(chunks)
                ]
                searcher.build_bm25_index(chunks_dict)
            
            results = searcher.search(
                query=prompt,
                top_k=top_k,
                document_id=document_id,
                use_hybrid=True
            )
            
            return [
                {
                    "chunk_id": r.chunk_id,
                    "similarity_score": float(r.hybrid_score),
                    "semantic_score": float(r.semantic_score),
                    "bm25_score": float(r.bm25_score),
                    "document_id": r.metadata.get("document_id", document_id),
                    "chunk_index": r.metadata.get("chunk_index"),
                    "token_count": r.metadata.get("token_count"),
                    "chunk_type": r.metadata.get("chunk_type"),
                    "section_path": r.section_path,
                    "text_preview": r.text
                }
                for r in results
            ]
            
        except ImportError:
            logger.warning("Hybrid search not available, using semantic-only")
    
    # Fallback to semantic search
    prompt_embedding = embedding_model.encode([prompt], convert_to_numpy=True)[0]
    search_k = top_k * 10 if document_id else top_k
    results = faiss_index.search(prompt_embedding, k=search_k)
    
    formatted_results = []
    for chunk_id, similarity_score, metadata in results:
        if document_id and metadata.get("document_id") != document_id:
            continue
            
        formatted_results.append({
            "chunk_id": chunk_id,
            "similarity_score": float(similarity_score),
            "semantic_score": float(similarity_score),
            "bm25_score": 0.0,
            "document_id": metadata.get("document_id"),
            "chunk_index": metadata.get("chunk_index"),
            "token_count": metadata.get("token_count"),
            "chunk_type": metadata.get("chunk_type"),
            "section_path": metadata.get("section_path", []),
            "text_preview": metadata.get("text", "")
        })
        
        if len(formatted_results) >= top_k:
            break
    
    return formatted_results


async def get_representative_chunks(
    file_path: str,
    num_chunks: int = 5,
    prompt: str = None
) -> List[Dict[str, Any]]:
    from app.services.chunk_selector import ChunkSelector, ChunkSelectionConfig
    from sentence_transformers import SentenceTransformer
    from app.core.config import get_settings
    
    document_id = Path(file_path).stem
    logger.info(f"Getting {num_chunks} representative chunks for: {document_id}")
    
    chunks = chunk_document(file_path)
    
    if len(chunks) <= num_chunks:
        return [
            {
                "chunk_id": f"{document_id}_chunk_{idx}",
                "chunk_index": idx,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "section_path": chunk.section_path if hasattr(chunk, 'section_path') else [],
                "score": 1.0,
                "selection_method": "all"
            }
            for idx, chunk in enumerate(chunks)
        ]
    
    # Create embeddings
    settings = get_settings()
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    embeddings = embedding_model.encode([c.text for c in chunks], convert_to_numpy=True)
    chunk_ids = [f"{document_id}_chunk_{idx}" for idx in range(len(chunks))]
    
    # Create chunk selector
    selector = ChunkSelector(ChunkSelectionConfig(
        max_total_tokens=3000,
        min_chunks=3,
        max_chunks=30
    ))
    
    if prompt:
        # Search-based selection
        faiss_index = FAISSIndex(dimension=embeddings.shape[1])
        metadata_list = [
            {
                "document_id": document_id,
                "chunk_index": idx,
                "token_count": c.token_count,
                "section_path": c.section_path,
                "text": c.text[:500]
            }
            for idx, c in enumerate(chunks)
        ]
        faiss_index.add_embeddings_batch(chunk_ids, embeddings, metadata_list)
        
        selected = selector.select_by_search(
            query=prompt,
            chunks=chunks,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            faiss_index=faiss_index,
            embedding_model=embedding_model,
            num_chunks=num_chunks,
            document_id=document_id
        )
    else:
        # Representative selection
        selected = selector.select_representative(
            chunks=chunks,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            num_chunks=num_chunks
        )
        
        # Ensure section coverage
        selected = selector.ensure_section_coverage(
            selected=selected,
            chunks=chunks,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            min_coverage=0.7
        )
    
    logger.info(f"✓ Selected {len(selected)} representative chunks")
    return selected


# CLI entry point
if __name__ == "__main__":
    import asyncio
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m document_parse <file_path> [--search <query>]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if "--search" in sys.argv:
        query_idx = sys.argv.index("--search") + 1
        query = sys.argv[query_idx] if query_idx < len(sys.argv) else ""
        
        faiss_index, chunk_ids = create_faiss_index_for_chunks(file_path)
        results = search_chunks_by_prompt(query, faiss_index=faiss_index, file_path=file_path, top_k=5)
        
        print(f"\nSearch results for: '{query}'")
        print("=" * 60)
        for i, r in enumerate(results, 1):
            section = " > ".join(r["section_path"][-2:]) if r["section_path"] else "N/A"
            print(f"\n{i}. Chunk #{r['chunk_index']} (Score: {r['similarity_score']:.4f})")
            print(f"   Section: {section}")
            print(f"   Text: {r['text_preview'][:150]}...")
    else:
        # Just parse and chunk
        chunks = chunk_document(file_path)
        print(f"✓ Created {len(chunks)} chunks from {file_path}")
