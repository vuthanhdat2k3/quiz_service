import os
import re
from typing import Optional, List, Tuple, Dict, Any

from llama_parse import LlamaParse
from loguru import logger

from app.core.config import get_settings
from app.parsers.base_parser import BaseParser

settings = get_settings()


class LlamaParseParser(BaseParser):

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.LLAMA_PARSE_API_KEY
        if not self.api_key:
            raise ValueError("LLAMA_PARSE_API_KEY is required")

        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            verbose=True,
        )

    async def parse(self, file_path: str) -> str:
        try:
            logger.info(f"Parsing document with LlamaParse: {file_path}")

            # Get JSON result with layout information
            json_result = await self.parser.aget_json(file_path)

            if not json_result or not json_result[0].get('pages'):
                raise ValueError("No content extracted from document")

            # Build markdown using custom heading rules
            markdown_content = self._build_markdown_from_json(json_result[0])

            logger.info(
                f"Successfully parsed document. Content length: {len(markdown_content)} chars"
            )

            return markdown_content

        except Exception as e:
            logger.error(f"Error parsing document with LlamaParse: {str(e)}")
            raise
    
    def _build_markdown_from_json(self, json_data: Dict[str, Any]) -> str:
        pages = json_data.get('pages', [])
        
        # First pass: analyze heading heights to establish thresholds
        heading_stats = self._analyze_heading_heights(pages)
        
        # Second pass: build markdown with correct heading levels
        markdown_lines = []
        
        for page in pages:
            items = page.get('items', [])
            
            for item in items:
                item_type = item.get('type', '')
                
                if item_type == 'heading':
                    line = self._process_heading_item(item, heading_stats)
                    if line:
                        markdown_lines.append(line)
                        
                elif item_type == 'text':
                    text = item.get('value', '').strip()
                    if text:
                        markdown_lines.append(text)
                        markdown_lines.append('')  # Empty line after paragraph
                elif item_type == 'table':
                    # Keep table markdown as-is
                    md = item.get('md', '')
                    if md:
                        markdown_lines.append(md)
                        markdown_lines.append('')
        
        # Normalize heading levels to remove gaps (H1->H3 becomes H1->H2)
        result = '\n'.join(markdown_lines)
        return self._normalize_heading_hierarchy(result)
    
    def _normalize_heading_hierarchy(self, markdown_content: str) -> str:
        lines = markdown_content.split('\n')
        heading_regex = re.compile(r'^(#+)\s+(.+)$')
        
        # First pass: collect all heading levels used
        heading_levels_used = set()
        for line in lines:
            match = heading_regex.match(line.strip())
            if match:
                level = len(match.group(1))
                heading_levels_used.add(level)
        
        if not heading_levels_used:
            return markdown_content
        
        # Create mapping from old levels to new normalized levels
        sorted_levels = sorted(heading_levels_used)
        level_mapping = {}
        for new_level, old_level in enumerate(sorted_levels, start=1):
            level_mapping[old_level] = new_level
        
        # Second pass: apply the mapping
        normalized_lines = []
        for line in lines:
            match = heading_regex.match(line.strip())
            if match:
                old_level = len(match.group(1))
                heading_text = match.group(2)
                new_level = level_mapping.get(old_level, old_level)
                normalized_lines.append('#' * new_level + ' ' + heading_text)
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _analyze_heading_heights(self, pages: List[Dict]) -> Dict[str, Any]:
        # Collect all heading heights and their content
        headings = []
        for page in pages:
            for item in page.get('items', []):
                if item.get('type') == 'heading':
                    height = item.get('bBox', {}).get('h', 0)
                    value = item.get('value', '').strip()
                    headings.append({
                        'height': height,
                        'value': value,
                        'has_numbering': self._has_numbering_pattern(value),
                        'is_title': self._is_title_pattern(value),
                        'is_ui_element': self._is_likely_ui_element(value),
                    })
        
        if not headings:
            return {'height_thresholds': [], 'min_valid_height': 10}
        
        # Group headings by structural validity
        valid_headings = [h for h in headings if h['has_numbering'] or h['is_title']]
        invalid_headings = [h for h in headings if h['is_ui_element']]
        
        # Determine height threshold
        if valid_headings:
            valid_heights = [h['height'] for h in valid_headings]
            min_valid = min(valid_heights)
            # Set threshold slightly below minimum valid height
            min_valid_height = min_valid * 0.85
        else:
            # Fallback: use 10pt as minimum
            min_valid_height = 10
        
        # If we have invalid headings, use their max height as additional filter
        if invalid_headings:
            invalid_heights = [h['height'] for h in invalid_headings]
            max_invalid = max(invalid_heights)
            # If invalid heights overlap with valid, prioritize content analysis
            if max_invalid >= min_valid_height:
                min_valid_height = max(min_valid_height, 9)  # At least 9pt
        
        # Build height-to-level mapping based on unique heights of valid headings
        height_levels = {}
        if valid_headings:
            unique_heights = sorted(set(h['height'] for h in valid_headings), reverse=True)
            for i, height in enumerate(unique_heights[:6]):  # Max 6 levels
                height_levels[height] = i + 1
        
        return {
            'min_valid_height': min_valid_height,
            'height_levels': height_levels,
            'valid_headings': valid_headings,
        }
    
    def _process_heading_item(self, item: Dict, stats: Dict) -> Optional[str]:
        value = item.get('value', '').strip()
        height = item.get('bBox', {}).get('h', 0)
        
        if not value:
            return None
        
        # Rule 1: Filter by height - too small is likely UI element
        min_valid_height = stats.get('min_valid_height', 10)
        if height < min_valid_height and not self._has_numbering_pattern(value):
            # Small text without numbering - likely UI element, return as plain text
            return value
        
        # Rule 2: Check if it's definitely a UI element by content
        if self._is_likely_ui_element(value):
            return value  # Return as plain text
        
        # Rule 3: Determine heading level
        level = self._determine_heading_level(value, height, stats)
        
        if level:
            return '#' * level + ' ' + value
        else:
            return value  # Cannot determine level, return as plain text
    
    def _determine_heading_level(self, value: str, height: float, stats: Dict) -> Optional[int]:
        # Check numbering pattern first - most reliable
        numbering_depth = self._get_numbering_depth(value)
        if numbering_depth > 0:
            # Map depth to level: depth 1 -> H2, depth 2 -> H3, etc.
            return min(numbering_depth + 1, 6)
        
        # Check if it's a title pattern
        if self._is_title_pattern(value):
            return 1  # Top-level heading
        
        # Use height-based level if available
        height_levels = stats.get('height_levels', {})
        if height in height_levels:
            return height_levels[height]
        
        # Find closest height
        if height_levels:
            closest_height = min(height_levels.keys(), key=lambda h: abs(h - height))
            if abs(closest_height - height) < 3:  # Within 3pt tolerance
                return height_levels[closest_height]
        
        # Default: if it passed all filters, assign H2
        return 2
    
    def _has_numbering_pattern(self, text: str) -> bool:
        # Remove bracket prefixes like [ok]
        clean = re.sub(r'^\[[\w\s]+\]\s*', '', text)
        
        patterns = [
            r'^\d+\.',           # 1.
            r'^\d+\.\d+',        # 1.1
            r'^[A-Z]\.',         # A.
            r'^[IVX]+\.',        # I., II., III.
            r'^\(\d+\)',        # (1)
            r'^\([a-z]\)',       # (a)
        ]
        
        for pattern in patterns:
            if re.match(pattern, clean):
                return True
        return False
    
    def _get_numbering_depth(self, text: str) -> int:
        clean = re.sub(r'^\[[\w\s]+\]\s*', '', text)
        
        # Check decimal numbering: 1.2.3
        match = re.match(r'^(\d+(?:\.\d+)*)\s*\.?\s+', clean)
        if match:
            return len(match.group(1).split('.'))
        
        # Single number or letter
        if re.match(r'^(\d+|[A-Za-z]|[IVXivx]+)\s*[\.\)\:]\s+', clean):
            return 1
        
        return 0
    
    def _is_title_pattern(self, text: str) -> bool:
        text_lower = text.lower()
        
        title_patterns = [
            r'^(module|chapter|chương|phần|part|unit|bài|mục)\s*\d*',
            r'^(introduction|conclusion|summary|overview|abstract)',
            r'^(giới thiệu|kết luận|tổng quan|mô tả|tóm tắt)',
            r'^(requirements?|specifications?|mục tiêu|yêu cầu)',
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, text_lower):
                return True
        return False
    
    def _is_likely_ui_element(self, text: str) -> bool:
        text_lower = text.lower().strip()
        original = text.strip()
        
        # 0. Check for bullet/symbol prefixes (these are list items, not headings)
        if re.match(r'^[◼◻●○•◦▪▸►⇒→\-\*]\s*', original):
            return True
        
        # 1. Exact match UI keywords
        ui_keywords = {
            'ok', 'cancel', 'save', 'add', 'edit', 'delete', 'create', 'join',
            'yes', 'no', 'confirm', 'apply', 'reset', 'close', 'back', 'next',
            'home', 'menu', 'settings', 'profile', 'logout', 'login',
            'chat', 'meet', 'call', 'share', 'post', 'send', 'reply',
            'members', 'owners', 'admins', 'users', 'channels', 'rooms',
            'notifications', 'alerts', 'messages', 'inbox', 'activity', 'feed',
            'name', 'title', 'description', 'status', 'type', 'date', 'time',
            'calendar', 'events', 'posts', 'school', 'communities',
        }
        
        if text_lower in ui_keywords:
            return True
        
        # 2. Patterns like "Word Word Number" (e.g., "Lop lon 3")
        if re.match(r'^[A-Za-z]+\s+[A-Za-z]+\s*\d*$', original):
            if not self._is_title_pattern(text):
                return True
        
        # 3. "Number + Word" pattern (e.g., "8 Events")
        if re.match(r'^\d+\s+[A-Za-z]+s?$', original):
            return True
        
        # 4. Very short text (1-2 words) without structure
        words = text.split()
        if len(words) <= 2:
            if not self._has_numbering_pattern(text) and not self._is_title_pattern(text):
                return True
        
        # 5. Common UI phrases
        ui_phrases = [
            r'^add\s+\w+', r'^edit\s+\w+', r'^delete\s+\w+', r'^create\s+\w+',
            r'^invite\s+\w+', r'^share\s+\w+', r'^manage\s+\w+',
            r'^welcome\s+to', r"let's\s+start", r'^give\s+your',
            r'^post\s+in', r'^channel\s+name',
        ]
        
        for pattern in ui_phrases:
            if re.match(pattern, text_lower):
                return True
        
        return False

    def get_file_extensions(self) -> list[str]:
        return [".pdf", ".doc", ".docx", ".ppt", ".pptx"]


class PDFParser(LlamaParseParser):

    def get_file_extensions(self) -> list[str]:
        return [".pdf"]


class WordParser(LlamaParseParser):

    def get_file_extensions(self) -> list[str]:
        return [".doc", ".docx"]


class PowerPointParser(LlamaParseParser):

    def get_file_extensions(self) -> list[str]:
        return [".ppt", ".pptx"]
