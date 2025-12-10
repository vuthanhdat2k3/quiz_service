
import asyncio
from pathlib import Path
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")


async def test_chunk_selector():
    
    from app.services.chunk_selector import ChunkSelector, ChunkSelectionConfig
    from app.chunkers.markdown_chunker import MarkdownChunkerV2
    from app.database.faiss_index import FAISSIndex
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    print("\n" + "=" * 80)
    print("TEST: ChunkSelector")
    print("=" * 80)
    
    # Load test document
    file_path = "sample_documents/Lesson15-Preclass-Reading-Pandas_SQL.md"
    document_id = Path(file_path).stem
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Chunk document
    chunker = MarkdownChunkerV2(max_tokens=180, min_tokens=12)
    chunks = chunker.parse(content)
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    
    print(f"\nDocument: {file_path}")
    print(f"Total chunks: {len(chunks)}")
    
    # Create embeddings
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedding_model.encode([c.text for c in chunks], convert_to_numpy=True)
    
    # Create FAISS index
    faiss_index = FAISSIndex(dimension=384)
    faiss_index.clear()
    
    metadata_list = [
        {
            "document_id": document_id,
            "chunk_index": i,
            "token_count": c.token_count,
            "section_path": c.section_path,
            "text": c.text[:500]
        }
        for i, c in enumerate(chunks)
    ]
    faiss_index.add_embeddings_batch(chunk_ids, embeddings, metadata_list)
    
    # Create selector
    config = ChunkSelectionConfig(
        max_total_tokens=3000,
        min_chunks=3,
        max_chunks=30,
        include_adjacent=True
    )
    selector = ChunkSelector(config)
    
    # Test 1: Calculate chunks needed for different question counts
    print("\n" + "-" * 40)
    print("Test 1: Chunks needed calculation")
    print("-" * 40)
    
    for num_q in [5, 10, 15, 20, 30]:
        num_chunks = selector.calculate_chunks_needed(num_q)
        print(f"  {num_q} questions → {num_chunks} chunks")
    
    # Test 2: Search-based selection
    print("\n" + "-" * 40)
    print("Test 2: Search-based selection (with prompt)")
    print("-" * 40)
    
    query = "Cách lọc dữ liệu theo điều kiện WHERE trong Pandas"
    selected = selector.select_by_search(
        query=query,
        chunks=chunks,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        faiss_index=faiss_index,
        embedding_model=embedding_model,
        num_chunks=5,
        document_id=document_id
    )
    
    print(f"\nQuery: {query}")
    print(f"Selected {len(selected)} chunks:")
    for i, c in enumerate(selected):
        section = " > ".join(c["section_path"][-2:]) if c["section_path"] else "N/A"
        print(f"  [{i+1}] {c['selection_method']:<20} Score: {c['score']:.3f} | Section: {section}")
        print(f"       Text: {c['text'][:80]}...")
    
    # Test 3: Representative selection
    print("\n" + "-" * 40)
    print("Test 3: Representative selection (no prompt)")
    print("-" * 40)
    
    representative = selector.select_representative(
        chunks=chunks,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        num_chunks=5
    )
    
    print(f"\nSelected {len(representative)} representative chunks:")
    for i, c in enumerate(representative):
        section = " > ".join(c["section_path"][-2:]) if c["section_path"] else "N/A"
        print(f"  [{i+1}] Score: {c['score']:.3f} | Section: {section}")
        print(f"       Text: {c['text'][:80]}...")
    
    # Test 4: Section coverage
    print("\n" + "-" * 40)
    print("Test 4: Section coverage check")
    print("-" * 40)
    
    # Get all sections
    all_sections = set()
    for chunk in chunks:
        if chunk.section_path:
            all_sections.add(tuple(chunk.section_path))
    
    covered_sections = set()
    for sel in representative:
        if sel["section_path"]:
            covered_sections.add(tuple(sel["section_path"]))
    
    coverage = len(covered_sections) / len(all_sections) if all_sections else 0
    print(f"Total sections: {len(all_sections)}")
    print(f"Covered sections: {len(covered_sections)}")
    print(f"Coverage: {coverage:.1%}")
    
    # Test with section coverage enforcement
    with_coverage = selector.ensure_section_coverage(
        selected=representative,
        chunks=chunks,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        min_coverage=0.8
    )
    
    new_covered = set()
    for sel in with_coverage:
        if sel["section_path"]:
            new_covered.add(tuple(sel["section_path"]))
    
    new_coverage = len(new_covered) / len(all_sections) if all_sections else 0
    print(f"\nAfter coverage enforcement:")
    print(f"Chunks: {len(representative)} → {len(with_coverage)}")
    print(f"Coverage: {coverage:.1%} → {new_coverage:.1%}")
    
    # Test 5: Token budget
    print("\n" + "-" * 40)
    print("Test 5: Token budget check")
    print("-" * 40)
    
    total_tokens = sum(c["token_count"] for c in with_coverage)
    print(f"Total tokens in selected chunks: {total_tokens}")
    print(f"Token budget: {config.max_total_tokens}")
    print(f"Within budget: {'✓' if total_tokens <= config.max_total_tokens else '✗'}")
    
    print("\n" + "=" * 80)
    print("ChunkSelector tests completed!")
    print("=" * 80)


async def test_quiz_modes():
    
    print("\n" + "=" * 80)
    print("TEST: Quiz Generation Modes")
    print("=" * 80)
    
    from app.services.quiz_service import QuizGenerationService
    
    service = QuizGenerationService()
    
    # Test Mode 1: Generate from prompt only (no document)
    print("\n" + "-" * 40)
    print("Mode 1: Generate from topic prompt (no document)")
    print("-" * 40)
    
    try:
        questions = await service.generate_from_prompt(
            prompt="Giải thích về DataFrame trong Pandas và cách sử dụng",
            num_questions=2,
            difficulty="medium",
            language="vi",
            question_types=[0]  # Single choice
        )
        
        print(f"Generated {len(questions)} questions from prompt")
        for i, q in enumerate(questions):
            print(f"\n  Q{i+1}: {q.questionText[:100]}...")
            for opt in q.options:
                mark = "✓" if opt.isCorrect else " "
                print(f"    [{mark}] {opt.optionText[:50]}...")
                
    except Exception as e:
        print(f"Mode 1 error (expected if no LLM configured): {e}")
    
    # Test Mode 2 & 3 require document
    file_path = "sample_documents/Lesson15-Preclass-Reading-Pandas_SQL.md"
    
    print("\n" + "-" * 40)
    print("Mode 2: Document + prompt (search-based)")
    print("-" * 40)
    
    try:
        questions, file_name, markdown, chunks, metadata = await service.generate_from_file_advanced(
            file_path=file_path,
            num_questions=3,
            difficulty="medium",
            language="vi",
            question_types=[0],
            additional_prompt="Cách lọc dữ liệu với WHERE và AND/OR trong Pandas"
        )
        
        print(f"Generated {len(questions)} questions")
        print(f"Pipeline steps: {len(metadata['steps'])}")
        
        # Show candidate selection step
        for step in metadata["steps"]:
            if step["name"] == "candidate_selection":
                print(f"Candidates selected: {step['num_candidates']}")
                
    except Exception as e:
        print(f"Mode 2 error: {e}")
    
    print("\n" + "-" * 40)
    print("Mode 3: Document only (representative selection)")
    print("-" * 40)
    
    try:
        questions, file_name, markdown, chunks, metadata = await service.generate_from_file_advanced(
            file_path=file_path,
            num_questions=3,
            difficulty="medium",
            language="vi",
            question_types=[0],
            additional_prompt=None  # No prompt = representative selection
        )
        
        print(f"Generated {len(questions)} questions")
        print(f"Pipeline steps: {len(metadata['steps'])}")
        
    except Exception as e:
        print(f"Mode 3 error: {e}")
    
    service.close()
    
    print("\n" + "=" * 80)
    print("Quiz modes test completed!")
    print("=" * 80)


if __name__ == "__main__":
    print("Starting Quiz Generation Tests...")
    
    # Test chunk selector
    asyncio.run(test_chunk_selector())
    
    # Optionally test full quiz generation (requires LLM)
    # asyncio.run(test_quiz_modes())
