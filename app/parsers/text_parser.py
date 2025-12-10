import aiofiles
from loguru import logger

from app.parsers.base_parser import BaseParser


class TextParser(BaseParser):

    async def parse(self, file_path: str) -> str:
        try:
            logger.info(f"Reading text file: {file_path}")

            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                content = await f.read()

            logger.info(f"Successfully read text file. Length: {len(content)} chars")

            # Convert to basic markdown format
            markdown_content = f"# Document Content\n\n{content}"

            return markdown_content

        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise

    def get_file_extensions(self) -> list[str]:
        return [".txt"]
