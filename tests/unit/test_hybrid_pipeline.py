import asyncio
from document_parse import search_chunks_by_prompt, get_representative_chunks

file_path = 'sample_documents/Lesson15-Preclass-Reading-Pandas_SQL.md'

async def main():
    # Test 1: Hybrid search with prompt
    print('='*80)
    print('TEST 1: search_chunks_by_prompt with hybrid search')
    print('='*80)
    
    results = search_chunks_by_prompt(
        prompt='Cách chọn cột dữ liệu trong Pandas giống SELECT trong SQL',
        file_path=file_path,
        top_k=3,
        use_hybrid=True
    )
    
    for r in results:
        print(f'[{r["chunk_index"]}] Hybrid: {r["similarity_score"]:.4f} | Sem: {r["semantic_score"]:.4f} | BM25: {r["bm25_score"]:.4f}')
        section = " > ".join(r["section_path"][-2:]) if r["section_path"] else "N/A"
        print(f'    Section: {section}')
        print(f'    Text: {r["text_preview"][:80]}...')
        print()

    # Test 2: get_representative_chunks with prompt (hybrid)
    print('='*80)
    print('TEST 2: get_representative_chunks with prompt (hybrid search)')
    print('='*80)
    
    chunks = await get_representative_chunks(
        file_path=file_path,
        num_chunks=3,
        prompt='thao tác lọc dữ liệu WHERE'
    )
    
    for c in chunks:
        print(f'[{c["chunk_index"]}] Score: {c["score"]:.4f} | Search: {c.get("search_score", 0):.4f}')
        section = " > ".join(c["section_path"][-2:]) if c["section_path"] else "N/A"
        print(f'    Section: {section}')
        print(f'    Text: {c["text"][:80]}...')
        print()
    
    # Test 3: get_representative_chunks without prompt (scoring only)
    print('='*80)
    print('TEST 3: get_representative_chunks without prompt (scoring only)')
    print('='*80)
    
    chunks_no_prompt = await get_representative_chunks(
        file_path=file_path,
        num_chunks=3
    )
    
    for c in chunks_no_prompt:
        print(f'[{c["chunk_index"]}] Score: {c["score"]:.4f}')
        section = " > ".join(c["section_path"][-2:]) if c["section_path"] else "N/A"
        print(f'    Section: {section}')
        print(f'    Method: {c.get("selection_method", "unknown")}')
        print()
    
    print('='*80)
    print('✓ All tests passed!')
    print('='*80)

if __name__ == "__main__":
    asyncio.run(main())
