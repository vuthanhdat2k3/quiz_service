# 🚀 Hướng Dẫn Deploy Quiz Service MIỄN PHÍ

## 📋 Tổng Quan Project

Project này là một Quiz Generation Service bao gồm:
- **FastAPI** - Web API server
- **Neo4j** - Graph database
- **Redis** - Message queue cho Celery
- **Celery Worker** - Background task processing

## 🎯 Các Tùy Chọn Deploy Miễn Phí

### Option 1: Railway.app (Khuyên dùng - Dễ nhất)
### Option 2: Render.com + Neo4j Aura + Upstash Redis
### Option 3: Fly.io + Neo4j Aura + Upstash Redis

---

## 🚂 Option 1: Railway.app (Khuyên dùng)

Railway cung cấp $5 credit miễn phí hàng tháng - đủ để chạy các service nhỏ.

### Bước 1: Chuẩn bị

1. Đăng ký tài khoản tại [railway.app](https://railway.app)
2. Tạo file `.env.example` (để tham khảo):

```env
# API Keys (Required)
LLAMA_PARSE_API_KEY=your_llama_parse_key
GOOGLE_API_KEY=your_google_api_key

# Optional APIs
OPENAI_API_KEY=
OPENROUTER_API_KEY=

# Environment
ENVIRONMENT=production
```

### Bước 2: Tạo Project trên Railway

1. Click "New Project" → "Deploy from GitHub repo"
2. Connect GitHub repository của bạn
3. Railway sẽ tự động detect Dockerfile

### Bước 3: Thêm Neo4j Database

1. Trong project, click "+ New" → "Database" → "Add Neo4j"
2. Railway sẽ tự động tạo Neo4j instance
3. Copy connection string từ Railway

### Bước 4: Thêm Redis

1. Click "+ New" → "Database" → "Add Redis"
2. Copy Redis URL

### Bước 5: Configure Environment Variables

Trong service settings, thêm các variables:
```
LLAMA_PARSE_API_KEY=<your_key>
GOOGLE_API_KEY=<your_key>
NEO4J_URI=<from_railway>
NEO4J_USER=<from_railway>
NEO4J_PASSWORD=<from_railway>
REDIS_URL=<from_railway>
ENVIRONMENT=production
```

### Bước 6: Deploy Worker (Optional)

1. Click "+ New" → "Empty Service"
2. Connect cùng GitHub repo
3. Set Start Command: `celery -A app.worker.celery_app worker --loglevel=info`

---

## 🎨 Option 2: Render.com + Free Databases

### Bước 1: Setup Free Neo4j (Neo4j Aura)

1. Đăng ký tại [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)
2. Tạo **Free Instance** (AuraDB Free)
3. Lưu lại credentials:
   - Connection URI: `neo4j+s://xxxxx.databases.neo4j.io`
   - Username: `neo4j`
   - Password: (generated)

### Bước 2: Setup Free Redis (Upstash)

1. Đăng ký tại [upstash.com](https://upstash.com)
2. Tạo Redis database (Free tier: 10,000 commands/day)
3. Copy Redis URL: `redis://default:xxxxx@xxxxx.upstash.io:6379`

### Bước 3: Deploy trên Render

1. Đăng ký tại [render.com](https://render.com)
2. Tạo **render.yaml** trong project:

```yaml
services:
  - type: web
    name: quiz-service
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: NEO4J_URI
        sync: false
      - key: NEO4J_USER
        value: neo4j
      - key: NEO4J_PASSWORD
        sync: false
      - key: REDIS_URL
        sync: false
      - key: LLAMA_PARSE_API_KEY
        sync: false
      - key: GOOGLE_API_KEY
        sync: false
    healthCheckPath: /health
```

3. Connect GitHub repo
4. Add environment variables trong Dashboard
5. Deploy!

---

## 🪰 Option 3: Fly.io

### Bước 1: Install Fly CLI

```bash
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# hoặc dùng scoop
scoop install flyctl
```

### Bước 2: Login và Initialize

```bash
flyctl auth login
flyctl launch
```

### Bước 3: Tạo file fly.toml

```toml
app = "quiz-service"
primary_region = "sin"  # Singapore

[build]
  dockerfile = "Dockerfile"

[env]
  ENVIRONMENT = "production"
  API_PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 512
```

### Bước 4: Set Secrets

```bash
flyctl secrets set LLAMA_PARSE_API_KEY=your_key
flyctl secrets set GOOGLE_API_KEY=your_key
flyctl secrets set NEO4J_URI=your_neo4j_uri
flyctl secrets set NEO4J_PASSWORD=your_password
flyctl secrets set REDIS_URL=your_redis_url
```

### Bước 5: Deploy

```bash
flyctl deploy
```

---

## 🔑 Lấy API Keys (Miễn Phí)

### 1. LlamaParse API Key
1. Đăng ký tại [cloud.llamaindex.ai](https://cloud.llamaindex.ai)
2. Free tier: 1000 pages/day

### 2. Google Gemini API Key
1. Đăng ký tại [makersuite.google.com](https://makersuite.google.com)
2. Free tier: 60 requests/minute

### 3. OpenRouter (Alternative cho Gemini)
1. Đăng ký tại [openrouter.ai](https://openrouter.ai)
2. Có free credits để bắt đầu

---

## 📁 Files Cần Tạo/Sửa

### 1. Tạo .env.example (cho reference)

```env
# Required
LLAMA_PARSE_API_KEY=
GOOGLE_API_KEY=

# Database (sẽ được cung cấp bởi hosting)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

# Redis
REDIS_URL=redis://localhost:6379/0

# Optional
OPENAI_API_KEY=
OPENROUTER_API_KEY=
ENVIRONMENT=development
```

### 2. Sửa Dockerfile cho production (nếu cần)

Dockerfile hiện tại đã OK cho production.

### 3. Tạo Procfile (cho Heroku/Render)

```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
worker: celery -A app.worker.celery_app worker --loglevel=info
```

---

## 🎯 Deploy Đơn Giản Nhất (Không cần Worker)

Nếu bạn muốn deploy đơn giản nhất mà không cần Celery Worker:

### Sử dụng Render với Free Databases

1. **Neo4j Aura Free** - Graph database
2. **Upstash Redis Free** - Chỉ cần cho cache (không bắt buộc)
3. **Render Free** - Web service

**Chi phí: $0/tháng**

### Các bước:

```bash
# 1. Push code lên GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/quiz-service.git
git push -u origin main

# 2. Setup Neo4j Aura (free)
# - Đăng ký tại neo4j.com/cloud/aura
# - Tạo Free instance
# - Lưu credentials

# 3. Setup Upstash Redis (free) - Optional
# - Đăng ký tại upstash.com
# - Tạo free Redis database

# 4. Deploy lên Render
# - Connect GitHub repo
# - Add environment variables
# - Deploy!
```

---

## ⚠️ Lưu Ý Quan Trọng

### Giới Hạn Free Tier:

| Service | Free Limit |
|---------|------------|
| Railway | $5 credit/tháng |
| Render | 750 hours/tháng, auto sleep sau 15 phút inactive |
| Fly.io | 3 shared-cpu VMs |
| Neo4j Aura | 200MB storage |
| Upstash Redis | 10,000 commands/day |

### Tips để tối ưu:

1. **Render**: Service sẽ sleep sau 15 phút - request đầu tiên sẽ chậm
2. **Railway**: Giám sát usage để không vượt quá $5
3. **Embedding Model**: Có thể dùng model nhỏ hơn như `all-MiniLM-L6-v2` thay vì `all-mpnet-base-v2` để giảm memory

### Nếu không cần Worker:

Bạn có thể tắt Celery worker và xử lý đồng bộ bằng cách:
- Không deploy worker service
- Documents sẽ được xử lý trực tiếp (có thể chậm hơn với file lớn)

---

## 🔧 Debug Commands

```bash
# Check logs trên Railway
railway logs

# Check logs trên Render
# Xem trong Dashboard → Service → Logs

# Check logs trên Fly.io
flyctl logs

# Test API locally
curl http://localhost:8000/health

# Test API sau khi deploy
curl https://your-app.railway.app/health
```

---

## 📞 Hỗ Trợ

- Railway Docs: https://docs.railway.app
- Render Docs: https://render.com/docs
- Fly.io Docs: https://fly.io/docs
- Neo4j Aura: https://neo4j.com/docs/aura

**Chúc bạn deploy thành công! 🎉**
