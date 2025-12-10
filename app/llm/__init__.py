from typing import Optional
from loguru import logger

from app.core.config import get_settings
from app.llm.base_adapter import LLMAdapter
from app.llm.mock_adapter import MockLLMAdapter
from app.llm.openrouter_adapter import OpenRouterAdapter

settings = get_settings()


def get_llm_adapter(
    provider: Optional[str] = None, model_name: Optional[str] = None, **kwargs
) -> LLMAdapter:
    # Check if mock mode is enabled
    if settings.MOCK_LLM or provider == "mock":
        logger.info("Using MockLLMAdapter (no API calls)")
        return MockLLMAdapter(**kwargs)

    # Determine provider
    if provider is None:
        # Auto-detect based on available API keys - prioritize OpenRouter for Gemini
        if settings.OPENROUTER_API_KEY:
            provider = "openrouter"
        elif settings.OPENAI_API_KEY:
            provider = "openai"
        elif settings.GOOGLE_API_KEY:
            provider = "gemini"
        else:
            logger.warning("No API keys found, falling back to mock adapter")
            return MockLLMAdapter(**kwargs)

    # Create adapter
    if provider == "openrouter":
        from app.llm.openrouter_adapter import OpenRouterAdapter

        model_name = model_name or settings.OPENROUTER_MODEL
        logger.info(f"Using OpenRouterAdapter with model {model_name}")
        return OpenRouterAdapter(model_name=model_name, **kwargs)

    elif provider == "openai":
        from app.llm.openai_adapter import OpenAIAdapter

        model_name = model_name or "gpt-4-turbo-preview"
        logger.info(f"Using OpenAIAdapter with model {model_name}")
        return OpenAIAdapter(model_name=model_name, **kwargs)

    elif provider == "gemini":
        from app.llm.gemini_adapter import GeminiAdapter

        model_name = model_name or settings.GEMINI_MODEL
        logger.info(f"Using GeminiAdapter with model {model_name}")
        return GeminiAdapter(model_name=model_name, **kwargs)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = [
    "LLMAdapter",
    "MockLLMAdapter",
    "OpenRouterAdapter",
    "get_llm_adapter",
]
