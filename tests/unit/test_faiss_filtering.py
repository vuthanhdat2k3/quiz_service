import asyncio
from pathlib import Path
from document_parse import create_faiss_index_for_chunks, search_chunks_by_prompt
from loguru import logger

async def test_multi_document_search():
    
    # File paths
    file1 = "sample_documents/Lesson15-Preclass-Reading-Pandas_SQL.md"
    
    logger.info("="*80)
    logger.info("TEST: Multi-Document FAISS Search")
    logger.info("="*80)
    
    # Create FAISS index for document 1
    logger.info(f"\n1. Creating index for document: {Path(file1).stem}")
    faiss_index, chunk_ids1 = create_faiss_index_for_chunks(file1)
    logger.info(f"   Added {len(chunk_ids1)} chunks from {Path(file1).stem}")
    
    # Test 1: Search within document 1 only
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Search within Document 1 only")
    logger.info("="*80)
    results1 = search_chunks_by_prompt(
        prompt="SQL và Pandas",
        faiss_index=faiss_index,
        document_id=Path(file1).stem,
        top_k=3
    )
    
    print(f"\n✅ Found {len(results1)} results from document 1:")
    for i, r in enumerate(results1, 1):
        print(f"   {i}. [{r['document_id']}#{r['chunk_index']}] Score: {r['similarity_score']:.4f}")
    
    # Test 2: Search across all documents (no filter)
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Search across ALL documents (no filter)")
    logger.info("="*80)
    results_all = search_chunks_by_prompt(
        prompt="SQL và Pandas",
        faiss_index=faiss_index,
        document_id=None,  # No filter
        top_k=5
    )
    
    print(f"\n✅ Found {len(results_all)} results from all documents:")
    doc_counts = {}
    for i, r in enumerate(results_all, 1):
        doc_id = r['document_id']
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        print(f"   {i}. [{doc_id}#{r['chunk_index']}] Score: {r['similarity_score']:.4f}")
    
    print(f"\n📊 Results by document:")
    for doc_id, count in doc_counts.items():
        print(f"   - {doc_id}: {count} chunks")
    
    # Test 3: Search in non-existent document
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Search in non-existent document")
    logger.info("="*80)
    results_none = search_chunks_by_prompt(
        prompt="SQL và Pandas",
        faiss_index=faiss_index,
        document_id="NonExistentDocument",
        top_k=3
    )
    
    print(f"\n✅ Found {len(results_none)} results (should be 0)")
    
    logger.info("\n" + "="*80)
    logger.info("✅ All tests completed!")
    logger.info("="*80)

if __name__ == "__main__":
    asyncio.run(test_multi_document_search())
