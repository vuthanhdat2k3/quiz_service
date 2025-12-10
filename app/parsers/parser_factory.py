from pathlib import Path
from typing import Optional

from loguru import logger

from app.parsers.base_parser import BaseParser
from app.parsers.llama_parser import PDFParser, PowerPointParser, WordParser
from app.parsers.text_parser import TextParser


class ParserFactory:

    _parsers: dict[str, type[BaseParser]] = {
        ".pdf": PDFParser,
        ".doc": WordParser,
        ".docx": WordParser,
        ".ppt": PowerPointParser,
        ".pptx": PowerPointParser,
        ".txt": TextParser,
    }

    @classmethod
    def get_parser(cls, file_path: str) -> Optional[BaseParser]:
        file_ext = Path(file_path).suffix.lower()

        parser_class = cls._parsers.get(file_ext)

        if parser_class is None:
            logger.warning(f"No parser found for file extension: {file_ext}")
            return None

        logger.info(f"Creating {parser_class.__name__} for {file_ext}")
        return parser_class()

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        file_ext = Path(file_path).suffix.lower()
        return file_ext in cls._parsers

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        return list(cls._parsers.keys())
