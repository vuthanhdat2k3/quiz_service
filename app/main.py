from contextlib import asynccontextmanager
from pathlib import Path
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api import quiz_router
from app.api.admin_routes import router as admin_router
from app.core.config import get_settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
settings = get_settings()


async def preload_embedding_model():
    """Pre-load embedding model in background to avoid cold start latency."""
    try:
        logger.info("🔄 Pre-loading embedding model in background...")
        
        # Import and load the model
        from app.services.quiz_service import _load_embedding_model, _get_device
        
        device = _get_device()
        _load_embedding_model(settings.EMBEDDING_MODEL, device)
        
        logger.info("✅ Embedding model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to pre-load embedding model: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Quiz Generation Service")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Gemini Model: {settings.GEMINI_MODEL}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    
    # Pre-load embedding model in background (non-blocking)
    asyncio.create_task(preload_embedding_model())
    
    yield
    logger.info("Shutting down Quiz Generation Service")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="AI-powered quiz generation service with document parsing",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(quiz_router, prefix="/api")
app.include_router(admin_router, prefix="/api")

# Mount static files for UI
public_dir = Path(__file__).parent.parent / "public"
if public_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(public_dir), html=True), name="ui")
    logger.info(f"Mounted UI at /ui from {public_dir}")


@app.get("/")
async def root():
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "quiz-generation-service"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.ENVIRONMENT == "development",
    )
