# 🚀 Hướng Dẫn Deploy Quiz Service MIỄN PHÍ

## 📋 Tổng Quan Project

Project này là một Quiz Generation Service bao gồm:
- **FastAPI** - Web API server
- **Neo4j Aura** - Graph database (cloud - FREE)
- ~~Redis~~ - Message queue (tạm thời disabled)
- ~~Celery Worker~~ - Background task (tạm thời disabled)

---

## 🚂 Deploy lên Railway.app (Chi tiết từng bước)

Railway cung cấp **$5 credit miễn phí hàng tháng** - đủ để chạy project này!

### 📌 Bước 1: Chuẩn bị trước khi deploy

#### 1.1. Lấy API Keys (miễn phí)

**LlamaParse API Key:**
1. Truy cập https://cloud.llamaindex.ai
2. Đăng ký tài khoản (dùng Google/GitHub)
3. Vào Dashboard → API Keys → Create new key
4. Copy API key (Free: 1000 pages/ngày)

**Google Gemini API Key:**
1. Truy cập https://makersuite.google.com
2. Đăng nhập bằng Google account
3. Click "Get API Key" → "Create API key"
4. Copy API key (Free: 60 requests/phút)

#### 1.2. Tạo Neo4j Aura Database (FREE)

1. Truy cập https://neo4j.com/cloud/aura/
2. Click **"Start Free"** → Đăng ký
3. Click **"New Instance"** → Chọn **"AuraDB Free"**
4. Chọn region: **Singapore** (gần Việt Nam)
5. Đặt tên: `quiz-service-db`
6. Click **"Create"**
7. **⚠️ QUAN TRỌNG**: Lưu ngay credentials:
   - Connection URI: `neo4j+s://xxxxxx.databases.neo4j.io`
   - Username: `neo4j`
   - Password: `xxxxxxxxx` (chỉ hiện 1 lần!)

#### 1.3. Push code lên GitHub

```powershell
# Đảm bảo đã có .gitignore
cd F:\quiz-service

# Kiểm tra status
git status

# Add và commit
git add .
git commit -m "Prepare for Railway deployment"

# Push lên GitHub
git push origin main
```

---

### 📌 Bước 2: Tạo Project trên Railway

1. Truy cập https://railway.app
2. Click **"Login"** → Đăng nhập bằng GitHub
3. Click **"New Project"**
4. Chọn **"Deploy from GitHub repo"**
5. Authorize Railway truy cập GitHub
6. Chọn repository **quiz_service**

---

### 📌 Bước 3: Configure Environment Variables

Sau khi tạo project, Railway sẽ tự động detect Dockerfile.

1. Click vào service vừa tạo
2. Click tab **"Variables"**
3. Click **"+ New Variable"** và thêm từng biến sau:

```
# Required API Keys
LLAMA_PARSE_API_KEY=<your_llamaparse_key>
GOOGLE_API_KEY=<your_google_gemini_key>

# Neo4j Aura (từ Bước 1.2)
NEO4J_URI=neo4j+s://xxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your_neo4j_password>

# App Config
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000

# Embedding (dùng model nhỏ để tiết kiệm RAM)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

**💡 Tips**: Có thể add nhiều variables cùng lúc bằng cách click **"RAW Editor"**

---

### 📌 Bước 4: Configure Deployment Settings

1. Click tab **"Settings"**
2. Scroll xuống **"Deploy"** section:
   - **Root Directory**: `/` (để mặc định)
   - **Build Command**: (để trống - dùng Dockerfile)
   - **Start Command**: (để trống - dùng CMD trong Dockerfile)

3. Scroll xuống **"Networking"**:
   - Click **"Generate Domain"** để tạo public URL
   - URL sẽ có dạng: `https://quiz-service-xxx.up.railway.app`

---

### 📌 Bước 5: Deploy!

1. Railway sẽ tự động deploy khi bạn:
   - Push code mới lên GitHub
   - Thay đổi environment variables
   
2. Xem logs: Click vào service → **"Deployments"** → Click deployment mới nhất

3. Kiểm tra status: Truy cập URL đã generate
   ```
   https://quiz-service-xxx.up.railway.app/health
   ```
   
   Response mong đợi:
   ```json
   {"status": "healthy", "service": "quiz-generation-service"}
   ```

---

### 📌 Bước 6: Test API

Truy cập Swagger docs:
```
https://quiz-service-xxx.up.railway.app/docs
```

Test generate quiz:
```bash
curl -X POST "https://quiz-service-xxx.up.railway.app/api/quiz/generate" \
  -H "Content-Type: multipart/form-data" \
  -F "num_questions=5" \
  -F "difficulty=medium" \
  -F "prompt=Tạo câu hỏi về lập trình Python"
```

---

## 🔧 Troubleshooting

### Lỗi: "Service crashed" hoặc "OOM"
- Railway Free chỉ có 512MB RAM
- Giảm `EMBEDDING_MODEL` xuống `all-MiniLM-L6-v2` (nhỏ hơn)
- Hoặc upgrade lên Hobby plan ($5/tháng)

### Lỗi: "Failed to connect to Neo4j"
- Kiểm tra `NEO4J_URI` có prefix `neo4j+s://` (có `+s`)
- Kiểm tra password đúng
- Kiểm tra Neo4j Aura instance đang "Running"

### Lỗi: "LLAMA_PARSE_API_KEY invalid"
- Kiểm tra key từ https://cloud.llamaindex.ai/api-key
- Đảm bảo không có khoảng trắng

### Xem logs chi tiết
```bash
# Cài Railway CLI (optional)
npm install -g @railway/cli

# Login
railway login

# Xem logs
railway logs
```

---

## 💰 Chi phí ước tính

| Service | Free Limit | Sau Free |
|---------|------------|----------|
| Railway | $5/tháng | Pay as you go |
| Neo4j Aura | 200MB, 50k nodes | $65/tháng |
| LlamaParse | 1000 pages/ngày | $0.003/page |
| Gemini | 60 req/phút | Pay per token |

**Với free tier, bạn có thể:**
- Chạy ~720 giờ/tháng
- Xử lý ~1000 documents/ngày
- Lưu trữ ~50,000 quiz chunks

---

## 🔄 Cập nhật code

Mỗi lần push code mới, Railway tự động redeploy:

```powershell
git add .
git commit -m "Update feature"
git push origin main
```

Railway sẽ tự động build và deploy trong 2-5 phút.

---

## 📞 Hỗ Trợ

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Neo4j Aura Docs: https://neo4j.com/docs/aura

**Chúc bạn deploy thành công! 🎉**
