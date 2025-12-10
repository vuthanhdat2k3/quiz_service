from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Quiz Generation Service"
    API_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # CORS Configuration
    ALLOWED_ORIGINS: str = "*"

    # LlamaParse API
    LLAMA_PARSE_API_KEY: str

    # Google Gemini API
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"

    # OpenAI API
    OPENAI_API_KEY: str = ""

    # OpenRouter API (for Gemini via OpenRouter)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "google/gemini-2.0-flash-001"

    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768
    FAISS_INDEX_PATH: str = "/app/faiss_index"

    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_EXTENSIONS: str = ".pdf,.doc,.docx,.ppt,.pptx,.txt,.png,.jpg,.jpeg"

    # Chunking Configuration
    CHUNK_TOKEN_CAP: int = 180
    MIN_CHUNK_TOKENS: int = 12
    DEFAULT_CHUNK_SIZE: int = 8000
    DEFAULT_CHUNK_OVERLAP: int = 500
    MAX_TOKENS_PER_CHUNK: int = 8000

    # Candidate Selection Configuration
    CANDIDATE_M_SLIDES: int = 150
    CANDIDATE_M_CHAPTERS: int = 300
    CLUSTERING_K_FACTOR: int = 5
    DEDUP_THRESHOLD: float = 0.88

    # Quiz Generation Configuration
    DEFAULT_NUM_QUESTIONS: int = 10
    MAX_NUM_QUESTIONS: int = 30
    DEFAULT_DIFFICULTY: str = "medium"
    DEFAULT_LANGUAGE: str = "vi"
    LLM_CONCURRENCY: int = 8
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_RETRIES: int = 3
    CONFIDENCE_AUTO_ACCEPT: float = 0.8

    # Mock mode for testing
    MOCK_LLM: bool = False
    MOCK_LLAMAPARSE: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    @property
    def allowed_origins_list(self) -> List[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_FILE_EXTENSIONS.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()
