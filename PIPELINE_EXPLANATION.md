# 🧠 Giải Thích Chi Tiết Pipeline & Thuật Toán Quiz Service

Tài liệu này mô tả chi tiết từng bước trong quy trình xử lý tài liệu để tạo câu hỏi trắc nghiệm (Quiz Generation Pipeline).

## 📋 Tổng Quan Pipeline

1.  **Parsing**: Chuyển đổi tài liệu (PDF, Docx...) sang Markdown chuẩn hóa.
2.  **Chunking**: Chia nhỏ văn bản thành các đoạn (chunks) có ý nghĩa ngữ nghĩa.
3.  **Graph Construction**: Xây dựng đồ thị kiến thức (Knowledge Graph) trong Neo4j.
4.  **Indexing**: Tạo vector index để tìm kiếm ngữ nghĩa (Semantic Search).
5.  **Selection**: Lựa chọn các chunk tốt nhất để gửi cho LLM (Hybrid Search / MMR).
6.  **Generation**: Tạo câu hỏi từ các chunk đã chọn.

---

## 1. Parsing (Phân tích tài liệu)
**File**: `app/parsers/llama_parser.py`

Chúng ta sử dụng **LlamaParse API** để trích xuất nội dung, nhưng có thêm lớp xử lý hậu kỳ (post-processing) mạnh mẽ để làm sạch dữ liệu.

### Thuật toán Post-processing:
1.  **Heading Analysis (Phân tích chiều cao heading)**:
    *   Hệ thống quét qua tất cả text block và chiều cao (font height) của chúng.
    *   Sử dụng thống kê để phân biệt đâu là **Heading thực sự** (H1, H2...) và đâu là **UI Noise** (text to nhưng là nút bấm, menu, footer...).
    *   *Logic*: Nếu text có numbering (1.1, A.) hoặc pattern tiêu đề ("Chapter 1", "Introduction"), nó được coi là heading hợp lệ để làm chuẩn chiều cao.

2.  **UI Element Filtering (Lọc nhiễu UI)**:
    *   Loại bỏ các thành phần giao diện như "Login", "Share", "Menu", "© Copyright".
    *   Sử dụng regex và danh sách từ khóa (keyword list) để phát hiện nhiễu.

3.  **Hierarchy Normalization (Chuẩn hóa phân cấp)**:
    *   Nhiều tài liệu bị nhảy cóc level (ví dụ: đang H1 nhảy xuống H3).
    *   Thuật toán sẽ map lại để đảm bảo cấu trúc cây liền mạch: H1 -> H2 -> H3.

---

## 2. Chunking (Chia nhỏ văn bản)
**File**: `app/chunkers/markdown_chunker.py`

Thay vì chia cắt máy móc theo số ký tự (fixed-size chunking), hệ thống sử dụng thuật toán **3-Step Hierarchical Chunking** để giữ trọn vẹn ngữ cảnh.

### Step 1: Header-based Splitting
*   Sử dụng `MarkdownHeaderTextSplitter` để cắt văn bản tại các điểm H1-H6.
*   Kết quả là các block thô tương ứng với từng tiểu mục.

### Step 2: Hierarchical Merging (Gộp theo cây)
*   **Vấn đề**: Nhiều heading con (H3, H4) có nội dung quá ngắn (ví dụ chỉ vài dòng). Nếu để riêng sẽ mất ngữ cảnh.
*   **Giải pháp**: Dựng lại cây phân cấp (Tree structure).
*   **Thuật toán**: Duyệt từ lá lên gốc (Bottom-up). Nếu một node con quá nhỏ (< `min_tokens`), nó sẽ được gộp (merge) vào node cha hoặc gộp với node anh em (sibling) liền kề.

### Step 3: Intelligent Splitting (Cắt thông minh)
*   Xử lý các chunk vẫn còn quá lớn (> `max_tokens` ~ 500 tokens).
*   **Table Handling**: Nếu chunk là bảng biểu, thuật toán sẽ cắt bảng theo hàng (row-based split) nhưng **tự động lặp lại header** cho mỗi phần cắt mới. Điều này giúp LLM luôn hiểu cột số liệu là gì.
*   **Content Splitting**: Tìm điểm cắt tối ưu dựa trên độ ưu tiên:
    1. Header (cao nhất)
    2. Dòng kẻ ngang (---)
    3. Kết thúc đoạn văn (Paragraph break)
    4. Dấu chấm câu.

---

## 3. Graph Construction (Xây dựng Đồ thị)
**File**: `app/graph/*`

Dữ liệu được lưu trữ trong **Neo4j** dưới dạng đồ thị để phục vụ truy vấn cấu trúc.

### Schema:
*   `(:Document) -[:HAS_SECTION]-> (:Section)`
*   `(:Section) -[:HAS_CHILD]-> (:Section)` (Tạo thành cây mục lục)
*   `(:Section) -[:HAS_CHUNK]-> (:Chunk)`
*   `(:Chunk) -[:NEXT]-> (:Chunk)` (Linked List để duyệt tuần tự)

### Thuật toán Section Hierarchy Builder:
*   Sử dụng **Stack** để tái tạo cấu trúc cha-con từ một danh sách section phẳng.
*   Khi gặp một section level N, nó sẽ tìm trong stack section gần nhất có level N-1 để làm cha.

---

## 4. Indexing (Đánh chỉ mục)
**File**: `app/database/faiss_index.py`

*   **Model**: Sử dụng `sentence-transformers/all-mpnet-base-v2` (hoặc model nhỏ hơn tùy cấu hình) để tạo vector (embedding).
*   **Store**: Sử dụng **FAISS** (Facebook AI Similarity Search).
*   **Metric**: Cosine Similarity (Vector được Normalization L2 trước khi đưa vào index `IndexFlatIP`).

---

## 5. Candidate Selection (Lựa chọn nội dung)
**File**: `app/services/chunk_selector.py`

Đây là bước quan trọng nhất để đảm bảo chất lượng câu hỏi. Có 2 chế độ:

### Mode A: Search-based (Khi người dùng nhập Prompt)
*   Sử dụng **Hybrid Search**: Kết hợp Vector Search (Semantic) và BM25 (Keyword Match).
*   `Score = 0.7 * Semantic_Score + 0.3 * Keyword_Score`
*   **Context Expansion**: Sau khi tìm được chunk tốt nhất, hệ thống tự động lấy thêm chunk liền trước và liền sau (dựa trên quan hệ `NEXT` hoặc index) để mở rộng ngữ cảnh.

### Mode B: Representative (Khi tạo Quiz tổng hợp)
Nếu không có prompt, làm sao chọn được phần quan trọng nhất? Hệ thống dùng thuật toán **MMR (Maximal Marginal Relevance)** kết hợp chấm điểm đa tiêu chí:

1.  **Centrality Score**: Chunk nào có nội dung "trung tâm" nhất (gần vector trung bình của cả bài) được điểm cao.
2.  **Structural Score**: Chunk có chứa Heading hoặc nằm ở đầu Section được cộng điểm.
3.  **Coverage Guarantee**:
    *   Hệ thống kiểm tra xem các chunk được chọn đã bao phủ đủ các Section chính chưa.
    *   Nếu một Section quan trọng chưa có chunk nào được chọn, thuật toán sẽ ép buộc chọn chunk có điểm centrality cao nhất trong section đó.

---

## 6. Question Generation (Tạo câu hỏi)
**File**: `app/services/quiz_service.py`

*   Gom các chunk đã chọn (Candidate Chunks) thành một context lớn.
*   Gửi một **Batch Prompt** duy nhất tới LLM (Gemini/OpenAI) để tạo nhiều câu hỏi một lúc.
*   Format output thành JSON để dễ dàng parse và lưu vào Database.
