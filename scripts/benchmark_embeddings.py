"""
Benchmark script for measuring document processing performance.

Usage:
    python scripts/benchmark_embeddings.py --file <path_to_document>
    python scripts/benchmark_embeddings.py --text "sample text" --count 100
"""

import argparse
import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


async def benchmark_embedding_loading():
    """Benchmark embedding model loading time."""
    from app.services.quiz_service import _load_embedding_model, _get_device
    from app.core.config import get_settings
    
    settings = get_settings()
    device = _get_device()
    
    print(f"\n{'='*60}")
    print("📊 EMBEDDING MODEL LOADING BENCHMARK")
    print(f"{'='*60}")
    print(f"Model: {settings.EMBEDDING_MODEL}")
    print(f"Device: {device}")
    
    # Clear cache to get fresh load time
    from app.services import quiz_service
    quiz_service._embedding_model_cache.clear()
    
    # Measure cold load time
    start = time.perf_counter()
    model = _load_embedding_model(settings.EMBEDDING_MODEL, device)
    cold_load_time = time.perf_counter() - start
    
    print(f"\n⏱️  Cold load time: {format_time(cold_load_time)}")
    
    # Measure cached load time
    start = time.perf_counter()
    model = _load_embedding_model(settings.EMBEDDING_MODEL, device)
    cached_load_time = time.perf_counter() - start
    
    print(f"⏱️  Cached load time: {format_time(cached_load_time)}")
    print(f"🚀 Speedup: {cold_load_time / cached_load_time:.1f}x faster")
    
    return model


async def benchmark_encoding(model, texts: list, batch_sizes: list = None):
    """Benchmark encoding with different batch sizes."""
    from app.services.quiz_service import _get_device
    
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64, 128, 256]
    
    device = _get_device()
    
    print(f"\n{'='*60}")
    print("📊 ENCODING BENCHMARK")
    print(f"{'='*60}")
    print(f"Number of texts: {len(texts)}")
    print(f"Device: {device}")
    print(f"Avg text length: {np.mean([len(t) for t in texts]):.0f} chars")
    
    results = []
    
    # Warmup
    print("\n🔥 Warming up...")
    _ = model.encode(texts[:min(10, len(texts))], convert_to_numpy=True)
    
    print("\n📈 Testing batch sizes...")
    for batch_size in batch_sizes:
        if batch_size > len(texts):
            continue
            
        # Run multiple times for accuracy
        times = []
        for _ in range(3):
            start = time.perf_counter()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        throughput = len(texts) / avg_time
        
        results.append({
            "batch_size": batch_size,
            "time": avg_time,
            "throughput": throughput
        })
        
        print(f"  Batch {batch_size:>4}: {format_time(avg_time):>10} | {throughput:.1f} texts/sec")
    
    # Find optimal batch size
    if results:
        best = max(results, key=lambda x: x["throughput"])
        print(f"\n✅ Optimal batch size: {best['batch_size']} ({best['throughput']:.1f} texts/sec)")
    
    return results


async def benchmark_faiss_operations(embeddings: np.ndarray):
    """Benchmark FAISS index operations."""
    from app.database.faiss_index import FAISSIndex
    
    print(f"\n{'='*60}")
    print("📊 FAISS INDEX BENCHMARK")
    print(f"{'='*60}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    dimension = embeddings.shape[1]
    
    # Create fresh index
    faiss_index = FAISSIndex(dimension=dimension)
    faiss_index.clear()
    
    # Benchmark batch add
    chunk_ids = [f"chunk_{i}" for i in range(len(embeddings))]
    metadata_list = [{"idx": i, "text": f"text_{i}"} for i in range(len(embeddings))]
    
    # Test with normalization
    start = time.perf_counter()
    faiss_index.add_embeddings_batch(chunk_ids, embeddings.copy(), metadata_list, already_normalized=False)
    add_time_norm = time.perf_counter() - start
    
    # Clear and test without normalization
    faiss_index.clear()
    
    start = time.perf_counter()
    faiss_index.add_embeddings_batch(chunk_ids, embeddings.copy(), metadata_list, already_normalized=True)
    add_time_no_norm = time.perf_counter() - start
    
    print(f"\n⏱️  Add {len(embeddings)} embeddings:")
    print(f"   With normalization: {format_time(add_time_norm)}")
    print(f"   Pre-normalized:     {format_time(add_time_no_norm)}")
    print(f"   🚀 Speedup: {add_time_norm / add_time_no_norm:.2f}x")
    
    # Benchmark search
    query = embeddings[0]
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        results = faiss_index.search(query, k=10)
        times.append(time.perf_counter() - start)
    
    avg_search_time = np.mean(times)
    print(f"\n⏱️  Search (k=10): {format_time(avg_search_time)} avg over 100 queries")
    print(f"   Queries/sec: {1 / avg_search_time:.0f}")
    
    faiss_index.clear()


async def run_full_benchmark(file_path: str = None, sample_texts: list = None):
    """Run full benchmark suite."""
    print("\n" + "="*60)
    print("🎯 QUIZ SERVICE PERFORMANCE BENCHMARK")
    print("="*60)
    
    # 1. Benchmark model loading
    model = await benchmark_embedding_loading()
    
    # 2. Prepare texts
    if file_path:
        from app.parsers import ParserFactory
        from app.chunkers.markdown_chunker import MarkdownChunkerV2
        
        print(f"\n📄 Processing file: {file_path}")
        
        parser = ParserFactory.get_parser(file_path)
        content = await parser.parse(file_path)
        
        chunker = MarkdownChunkerV2()
        chunks = chunker.parse(content)
        
        texts = [c.text for c in chunks]
        print(f"📦 Created {len(chunks)} chunks")
    elif sample_texts:
        texts = sample_texts
    else:
        # Generate sample texts
        texts = [f"This is sample text number {i} with some content." for i in range(100)]
    
    # 3. Benchmark encoding
    results = await benchmark_encoding(model, texts)
    
    # 4. Get embeddings for FAISS benchmark
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    # 5. Benchmark FAISS
    await benchmark_faiss_operations(embeddings)
    
    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding performance")
    parser.add_argument("--file", type=str, help="Path to document file to process")
    parser.add_argument("--text", type=str, help="Sample text to encode")
    parser.add_argument("--count", type=int, default=100, help="Number of samples")
    
    args = parser.parse_args()
    
    if args.file:
        asyncio.run(run_full_benchmark(file_path=args.file))
    elif args.text:
        texts = [args.text] * args.count
        asyncio.run(run_full_benchmark(sample_texts=texts))
    else:
        asyncio.run(run_full_benchmark())


if __name__ == "__main__":
    main()
