import sys
from loguru import logger
from pathlib import Path


from app.core.config import get_settings

settings = get_settings()


def setup_logging():
    # Remove default handler
    logger.remove()

    # Add custom handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # Add file handler for errors
    try:
        log_path = Path("logs/error.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="10 MB",
            retention="1 week",
        )
    except Exception as e:
        print(f"Failed to setup file logging: {e}")

    return logger
