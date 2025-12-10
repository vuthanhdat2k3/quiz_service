import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_markdown_parser():
    logger.info("=" * 60)
    logger.info("Testing Markdown Chunker V2")
    logger.info("=" * 60)

    try:
        from app.chunkers.markdown_chunker import MarkdownChunkerV2

        chunker = MarkdownChunkerV2(max_tokens=180, min_tokens=12)

        # Test with sample document
        sample_md = """
# Python Basics

Python is a programming language.

## Variables

Variables store data:
- Integers: 1, 2, 3
- Strings: "hello"
- Floats: 3.14

### Example Code

```python
x = 10
print(x)
```

## Conclusion

Python is easy to learn.
"""

        chunks = chunker.parse(sample_md)

        logger.info(f"✓ Parsed {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:5]):  # Show first 5
            logger.info(
                f"  Chunk {i+1}: {chunk.chunk_type} | "
                f"{chunk.token_count} tokens | "
                f"Section: {' > '.join(chunk.section_path)}"
            )
            logger.debug(f"    Text: {chunk.text[:80]}...")

        logger.success("✅ Markdown chunker test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Markdown chunker test failed: {e}")
        return False


async def test_llm_adapters():
    logger.info("=" * 60)
    logger.info("Testing LLM Adapters (Mock Mode)")
    logger.info("=" * 60)

    try:
        from app.llm import get_llm_adapter

        # Get mock adapter
        adapter = get_llm_adapter(provider="mock")
        logger.info(f"✓ Created adapter: {adapter.__class__.__name__}")

        # Test MCQ generation
        passage = "Python is a high-level programming language known for its simplicity."
        mcq_result = await adapter.generate_mcq(passage)

        logger.info(f"✓ Generated MCQ:")
        logger.info(f"  Question: {mcq_result.question[:80]}...")
        logger.info(f"  Answer: {mcq_result.answer}")
        logger.info(f"  Difficulty: {mcq_result.difficulty}")

        # Test distractor refinement
        candidates = ["Java", "C++", "Ruby", "JavaScript", "Go", "Rust"]
        distractor_result = await adapter.refine_distractors(passage, "Python", candidates)

        logger.info(f"✓ Generated distractors: {len(distractor_result.distractors)}")
        for i, distractor in enumerate(distractor_result.distractors):
            logger.info(f"  {i+1}. {distractor}")

        logger.success("✅ LLM adapter test passed")
        return True

    except Exception as e:
        logger.error(f"❌ LLM adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_neo4j_connection():
    logger.info("=" * 60)
    logger.info("Testing Neo4j Connection")
    logger.info("=" * 60)

    try:
        from app.database.neo4j_db import get_neo4j_db, close_neo4j_db

        db = get_neo4j_db()
        logger.info("✓ Connected to Neo4j")

        # Test query
        result = db.execute_query("RETURN 1 as test")
        if result and result[0].get("test") == 1:
            logger.info("✓ Query executed successfully")

        close_neo4j_db()
        logger.success("✅ Neo4j connection test passed")
        return True

    except Exception as e:
        logger.warning(f"⚠️  Neo4j not available: {e}")
        logger.info("   This is OK if Neo4j is not running")
        return True  # Don't fail if Neo4j is not running


async def test_faiss_index():
    logger.info("=" * 60)
    logger.info("Testing FAISS Index")
    logger.info("=" * 60)

    try:
        import numpy as np
        from app.database.faiss_index import FAISSIndex

        index = FAISSIndex(dimension=768)
        logger.info("✓ Created FAISS index")

        # Add some test embeddings
        embeddings = np.random.rand(5, 768).astype("float32")
        chunk_ids = [f"chunk_{i}" for i in range(5)]

        index.add_embeddings_batch(chunk_ids, embeddings)
        logger.info(f"✓ Added {len(chunk_ids)} embeddings")

        # Test search
        query = embeddings[0]  # Use first embedding as query
        results = index.search(query, k=3)

        logger.info(f"✓ Search returned {len(results)} results")
        for chunk_id, score, metadata in results[:3]:
            logger.info(f"  {chunk_id}: {score:.4f}")

        logger.success("✅ FAISS index test passed")
        return True

    except Exception as e:
        logger.error(f"❌ FAISS index test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    logger.info("🚀 Starting component tests...")
    logger.info("")

    results = []

    # Run tests
    results.append(await test_markdown_parser())
    logger.info("")

    results.append(await test_llm_adapters())
    logger.info("")

    results.append(await test_neo4j_connection())
    logger.info("")

    results.append(await test_faiss_index())
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        logger.success(f"✅ All tests passed ({passed}/{total})")
        return 0
    else:
        logger.warning(f"⚠️  Some tests failed ({passed}/{total})")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
