from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseParser(ABC):

    @abstractmethod
    async def parse(self, file_path: str) -> str:
        pass

    @abstractmethod
    def get_file_extensions(self) -> list[str]:
        pass

    async def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        return {
            "file_name": file_path.split("/")[-1],
            "file_type": file_path.split(".")[-1],
        }
