# Performance Optimizations for Quiz Service

## Tổng quan các tối ưu đã thực hiện

### 1. ⚡ Embedding Model Loading (Tiết kiệm ~3-5 giây mỗi request)

**Vấn đề gốc:** `SentenceTransformer` được khởi tạo mới mỗi lần xử lý document, tốn ~3-5 giây cho mỗi lần load.

**Giải pháp:**
- Singleton pattern với global cache cho embedding model
- Pre-load model khi server khởi động (background task)
- Model được tái sử dụng cho tất cả requests

```python
# Trước
embedding_model = SentenceTransformer(self.settings.EMBEDDING_MODEL)  # Load mỗi request

# Sau
@property
def embedding_model(self):
    """Lazy-load and cache embedding model."""
    if self._embedding_model is None:
        self._embedding_model = _load_embedding_model(...)
    return self._embedding_model
```

### 2. 🖥️ GPU/Device Optimization

**Cải tiến:**
- Tự động detect và sử dụng GPU (CUDA) nếu có
- Hỗ trợ Apple Silicon (MPS)
- Fallback về CPU nếu không có GPU

```python
def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"
```

### 3. 🔄 Async Encoding (Non-blocking)

**Vấn đề gốc:** Encoding embeddings block event loop, làm chậm toàn bộ service.

**Giải pháp:**
- Sử dụng `ThreadPoolExecutor` để chạy encoding trong background thread
- Event loop không bị block, có thể xử lý requests khác

```python
async def _encode_texts_async(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(self._executor, _encode)
```

### 4. 📦 Optimized Batch Processing

**Cải tiến:**
- Dynamic batch size dựa trên device (128 cho GPU, 64 cho CPU)
- Pre-normalize embeddings trong encode step, skip trong FAISS add
- Sử dụng dict comprehension thay vì loops cho metadata

```python
# Batch size tối ưu theo device
batch_size = 128 if self._device == "cuda" else 64

# Skip double normalization
self.faiss_index.add_embeddings_batch(chunk_ids, embeddings, metadata_list, already_normalized=True)
```

### 5. 🗃️ FAISS Index Optimization

**Cải tiến:**
- Skip normalization nếu embeddings đã được normalize
- Batch update metadata với dict.update() thay vì loop
- Giảm memory copies không cần thiết

```python
def add_embeddings_batch(self, ..., already_normalized: bool = False):
    if not already_normalized:
        faiss.normalize_L2(embeddings)  # Skip nếu đã normalize
    
    # Batch update thay vì loop
    new_id_map = {start_id + i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
    self.id_map.update(new_id_map)
```

### 6. 🚀 Server Startup Preloading

**Cải tiến:**
- Pre-load embedding model khi server khởi động
- Background task để không block server startup
- First request không cần chờ model loading

```python
async def preload_embedding_model():
    """Pre-load embedding model in background to avoid cold start latency."""
    from app.services.quiz_service import _load_embedding_model, _get_device
    _load_embedding_model(settings.EMBEDDING_MODEL, _get_device())

# Trong lifespan
asyncio.create_task(preload_embedding_model())
```

## 📊 Kết quả ước tính

| Optimization | Tiết kiệm thời gian |
|--------------|---------------------|
| Model caching | ~3-5 giây/request |
| GPU utilization | 2-5x faster encoding |
| Async encoding | Non-blocking IO |
| Skip normalization | ~10-20% faster FAISS add |
| Server preload | Eliminate first-request latency |

## 🧪 Cách test performance

```bash
# Chạy benchmark
python scripts/benchmark_embeddings.py

# Với file cụ thể
python scripts/benchmark_embeddings.py --file path/to/document.pdf

# Với sample text
python scripts/benchmark_embeddings.py --text "Sample text" --count 100
```

## 📝 Config tối ưu thêm

Trong `.env`:

```env
# Tăng batch size nếu có nhiều VRAM
EMBEDDING_BATCH_SIZE=128

# Model nhỏ hơn cho tốc độ (trade-off accuracy)
# EMBEDDING_MODEL=all-MiniLM-L6-v2  # 384 dim, faster
EMBEDDING_MODEL=all-mpnet-base-v2   # 768 dim, more accurate
```

## 🔧 Tối ưu thêm có thể thực hiện

1. **ONNX Runtime**: Convert model sang ONNX để inference nhanh hơn
2. **Quantization**: INT8 quantization cho model nhỏ hơn, nhanh hơn
3. **Redis Caching**: Cache embeddings trong Redis cho multiple workers
4. **Parallel Chunking**: Song song hóa bước chunking
5. **Streaming Processing**: Process chunks theo batches thay vì tất cả cùng lúc
