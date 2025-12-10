import os
from pathlib import Path


def cleanup_temp_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass  # Silently fail on cleanup


def validate_file_extension(file_name: str, allowed_extensions: list[str]) -> bool:
    file_ext = Path(file_name).suffix.lower()
    return file_ext in allowed_extensions


def format_file_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
