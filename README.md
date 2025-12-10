# Quiz Generation Service

AI-powered quiz generation from documents with advanced chunk selection and hybrid search.

## Features

- **3 Quiz Generation Modes**:
  1. **Topic-based**: Generate quizzes from a text prompt (no document)
  2. **Document + Prompt**: Hybrid search (BM25 + Semantic) for relevant chunks
  3. **Document only**: Representative chunk selection with section coverage

- **Document Processing**:
  - Multi-format support: PDF, DOC, DOCX, PPT, PPTX, TXT
  - LlamaParse integration for accurate document parsing
  - Markdown AST chunking (≤180 tokens per chunk)

- **Intelligent Chunk Selection**:
  - Hybrid BM25 + Semantic search (MRR: 0.719)
  - Context expansion (adjacent chunks)
  - Section coverage enforcement (≥70%)
  - Token budget control (max 3000 tokens)

- **Quiz Generation**:
  - Single choice / Multiple choice / Mixed
  - Difficulty levels: Easy, Medium, Hard
  - Multi-language support (Vietnamese, English)
  - Graph-based distractor enhancement

## Architecture

```
Document → Parse → Chunk → Embed → Select → Generate
                    ↓         ↓
                 Neo4j     FAISS
```

### Pipeline Steps

1. **Parse**: Document → Markdown (LlamaParse)
2. **Chunk**: Markdown AST chunking (≤180 tokens)
3. **Store**: Neo4j graph (Document → Section → Chunk)
4. **Embed**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
5. **Index**: FAISS vector index
6. **Select**: ChunkSelector (hybrid search / representative)
7. **Generate**: LLM question generation (Gemini 2.0)
8. **Enhance**: Graph-based distractor refinement

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Start services
docker-compose up -d  # Neo4j + Redis

# 4. Run API
uvicorn app.main:app --reload
```

## Project Structure

```
quiz-service/
├── app/
│   ├── api/              # FastAPI routes
│   ├── chunkers/         # Markdown chunking
│   │   └── markdown_chunker.py  # MarkdownChunkerV2
│   ├── database/         # Neo4j + FAISS
│   │   ├── neo4j_db.py
│   │   └── faiss_index.py
│   ├── search/           # Hybrid search
│   │   └── hybrid_search.py
│   ├── services/         # Business logic
│   │   ├── quiz_service.py      # Main quiz generation
│   │   └── chunk_selector.py    # Intelligent chunk selection
│   ├── llm/              # LLM adapters
│   ├── parsers/          # Document parsers
│   └── graph/            # Graph builders
├── tests/                # Unit tests
├── sample_documents/     # Test documents
├── document_parse.py     # CLI document processing
└── requirements.txt
```

## API Endpoints

### Quiz Generation

```http
POST /api/quiz/generate
Content-Type: multipart/form-data

file: <document>
num_questions: 10
difficulty: "medium"
language: "vi"
question_types: [0]  # 0=single, 1=multiple, 2=mix
additional_prompt: "Focus on SQL concepts"  # optional
```

### Document Processing

```http
POST /api/document/parse
POST /api/document/chunk
POST /api/document/search
```

## Configuration

```env
# LLM
GOOGLE_API_KEY=your_key
LLM_PROVIDER=gemini

# LlamaParse
LLAMA_CLOUD_API_KEY=your_key

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chunk Selection
MAX_TOTAL_TOKENS=3000
MAX_CHUNKS=30
```

## Chunk Selection Algorithm

### Search-based (with prompt)
1. Run hybrid search (BM25 + Semantic, weight: 0.7 semantic)
2. Expand with adjacent chunks (same section)
3. Respect token budget (max 3000 tokens)

### Representative (no prompt)
1. Score chunks by: centrality, length, position, heading
2. Use MMR (λ=0.7) for diversity
3. Ensure section coverage (≥70%)
4. Respect token budget

### Chunks Calculation
- 1-5 questions: ~1.5x chunks
- 6-15 questions: ~1.3x chunks
- 16-30 questions: ~1.2x chunks
- Max: 30 chunks or token budget

## Testing

```bash
# Run chunk selector tests
python tests/unit/test_quiz_modes.py

# Run search accuracy tests
python tests/unit/test_search_accuracy.py

# Test document processing
python document_parse.py sample_documents/doc.md --search "query"
```

## Performance

| Metric | Value |
|--------|-------|
| Hybrid Search MRR | 0.719 |
| Hybrid Search P@5 | 45.71% |
| Pure Semantic MRR | 0.643 |
| Search Latency | ~8ms |

## Docker

```bash
# Start full stack
docker-compose up -d

# Services:
# - quiz-service: http://localhost:8000
# - neo4j: bolt://localhost:7687, http://localhost:7474
# - redis: localhost:6379
```

## License

MIT
