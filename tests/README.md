# Quiz Service Pipeline Test Suite

## 📋 Mô tả

Thư mục này chứa các script test cho từng bước trong pipeline sinh câu hỏi trắc nghiệm tự động. Mỗi file chạy một bước riêng lẻ và lưu kết quả chi tiết (JSON + Markdown) vào folder `results/`.

## 🚀 Cách sử dụng

### Yêu cầu
- Python 3.9+
- Đã cài đặt dependencies: `pip install -r requirements.txt`
- Neo4j database đang chạy (cho Step 3)
- File `.env` đã cấu hình (LLM API keys)

### Chạy từng bước riêng lẻ

```bash
# Di chuyển vào thư mục tests
cd G:\quiz_service\tests

# Step 1: Document Parsing (PDF → Markdown)
python step1_document_parser.py

# Step 2: Markdown Chunking (Markdown → Chunks)
python step2_markdown_chunker.py

# Step 3: Graph Building (Chunks → Neo4j Graph)
python step3_graph_builder.py

# Step 4: Embedding Computation (Chunks → FAISS Index)
python step4_embedding_computation.py

# Step 5: Chunk Selection (Embeddings → Selected Chunks)
python step5_chunk_selection.py

# Step 6: Question Generation (Selected Chunks → Questions)
python step6_question_generation.py
```

### Chạy toàn bộ pipeline

```bash
python run_full_pipeline.py
```

## 📁 Cấu trúc file

```
tests/
├── results/                          # Kết quả output
│   ├── step1_*_parsed.json          # Kết quả parsing
│   ├── step1_*_parsed.md            
│   ├── step2_*_chunks.json          # Kết quả chunking
│   ├── step2_*_chunks.md
│   ├── step3_*_graph.json           # Kết quả graph building
│   ├── step3_*_graph.md
│   ├── step4_*_embeddings.json      # Kết quả embeddings
│   ├── step4_*_embeddings.md
│   ├── step5_*_selection.json       # Kết quả chunk selection
│   ├── step5_*_selection.md
│   ├── step6_*_questions.json       # Kết quả question generation
│   ├── step6_*_questions.md
│   └── full_pipeline_*.json/md      # Kết quả full pipeline
│
├── step1_document_parser.py         # Step 1: Parse PDF → Markdown
├── step2_markdown_chunker.py        # Step 2: Chunk Markdown
├── step3_graph_builder.py           # Step 3: Build Neo4j Graph
├── step4_embedding_computation.py   # Step 4: Compute Embeddings
├── step5_chunk_selection.py         # Step 5: Select Chunks
├── step6_question_generation.py     # Step 6: Generate Questions
├── run_full_pipeline.py             # Chạy toàn bộ pipeline
└── README.md                         # File này
```

## 📊 Mô tả từng bước

### Step 1: Document Parsing
- **Input:** File PDF (`documents/Reading-Matplotlib_List.pdf`)
- **Output:** Nội dung Markdown
- **Tool:** LlamaParse API
- **Đánh giá:** 
  - Content richness (số ký tự)
  - Structure organization (heading distribution)
  - Hierarchy quality (H1 → H2 → H3)

### Step 2: Markdown Chunking
- **Input:** Markdown content từ Step 1
- **Output:** Danh sách chunks với metadata
- **Algorithm:** MarkdownChunkerV2 (AST-based)
- **Config:** max_tokens=400, min_tokens=50
- **Đánh giá:**
  - Token size distribution
  - Section coverage
  - Type diversity (text, table, code)

### Step 3: Graph Building
- **Input:** Chunks từ Step 2
- **Output:** Knowledge Graph trong Neo4j
- **Nodes:** Document → Section (hierarchy) → Chunk
- **Đánh giá:**
  - Section hierarchy depth
  - Chunk coverage
  - Relationship density

### Step 4: Embedding Computation
- **Input:** Chunks từ Step 2
- **Output:** Embeddings trong FAISS index
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Đánh giá:**
  - Completeness
  - Normalization quality
  - Embedding diversity

### Step 5: Chunk Selection
- **Input:** Chunks + Embeddings
- **Output:** Selected chunks cho question generation
- **Methods:**
  - Representative Selection (document coverage)
  - Search-based Selection (query relevance)
- **Đánh giá:**
  - Selection efficiency
  - Section coverage
  - Relevance scores

### Step 6: Question Generation
- **Input:** Selected chunks từ Step 5
- **Output:** Quiz questions (Single/Multiple choice)
- **LLM:** Gemini/OpenAI/OpenRouter
- **Đánh giá:**
  - Completion rate
  - Format quality
  - Type diversity
  - Content richness

## 📈 Evaluation Metrics

Mỗi bước có các metrics đánh giá riêng với thang điểm 0-100:

| Score Range | Assessment |
|-------------|------------|
| 80-100 | Excellent |
| 60-79 | Good |
| 40-59 | Fair |
| 0-39 | Poor |

## ⚙️ Cấu hình

Có thể thay đổi các tham số trong mỗi file:

```python
# Input file
INPUT_FILE = r"G:\quiz_service\documents\Reading-Matplotlib_List.pdf"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results"

# Question generation config
NUM_QUESTIONS = 5
DIFFICULTY = "medium"  # easy, medium, hard
LANGUAGE = "en"        # en, vi
```

## 📝 Lưu ý

1. **Neo4j:** Step 3 cần Neo4j database. Đảm bảo Neo4j đang chạy trước khi test.

2. **API Keys:** Step 1 (LlamaParse) và Step 6 (LLM) cần API keys trong `.env`:
   ```
   LLAMA_PARSE_API_KEY=your_key
   GEMINI_API_KEY=your_key
   ```

3. **GPU:** Step 4 sẽ tự động sử dụng GPU nếu có sẵn (CUDA/MPS).

4. **Cleanup:** Mỗi script tự động cleanup data sau khi chạy xong.

## 🔍 Xem kết quả

Sau khi chạy, kiểm tra folder `results/`:

- **JSON files:** Chứa dữ liệu đầy đủ để phân tích tự động
- **MD files:** Chứa báo cáo human-readable để xem và chụp ảnh

## 📷 Chụp ảnh báo cáo

Mở các file `.md` trong VS Code hoặc preview Markdown để chụp ảnh:
1. Right-click → "Open Preview" (VS Code)
2. Hoặc mở trong browser với Markdown viewer
