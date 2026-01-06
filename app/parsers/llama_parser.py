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
        
        # Track current heading context for proper hierarchy
        self._current_heading_context = []  # Stack of (level, depth)

    async def parse(self, file_path: str) -> str:
        try:
            logger.info(f"Parsing document with LlamaParse: {file_path}")

            # Reset heading context for each parse
            self._current_heading_context = []

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
        
        # Reset heading context
        self._current_heading_context = []
        
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
        
        result = '\n'.join(markdown_lines)
        
        # Normalize to remove level gaps while preserving semantic meaning
        return self._normalize_heading_gaps_only(result)
    
    def _normalize_heading_gaps_only(self, markdown_content: str) -> str:
        """
        Normalize headings to remove level gaps while PRESERVING semantic hierarchy.
        
        This only compresses gaps (e.g., H1->H3 becomes H1->H2) without changing
        the relative meaning assigned by _determine_heading_level_contextual.
        
        Example: If we have H1, H2, H4 (no H3), it becomes H1, H2, H3.
        But H2-H2-H2 stays as siblings, not restructured.
        """
        lines = markdown_content.split('\n')
        heading_regex = re.compile(r'^(#+)\s+(.+)$')
        
        # Collect all heading levels used
        levels_used = set()
        for line in lines:
            match = heading_regex.match(line.strip())
            if match:
                level = len(match.group(1))
                levels_used.add(level)
        
        if not levels_used:
            return markdown_content
        
        # Create gap-compression mapping
        # This preserves relative order but removes unused levels
        sorted_levels = sorted(levels_used)
        level_mapping = {old_level: new_level for new_level, old_level in enumerate(sorted_levels, start=1)}
        
        # Apply mapping
        result_lines = []
        for line in lines:
            match = heading_regex.match(line.strip())
            if match:
                old_level = len(match.group(1))
                heading_text = match.group(2)
                new_level = level_mapping.get(old_level, old_level)
                result_lines.append('#' * new_level + ' ' + heading_text)
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _analyze_heading_heights(self, pages: List[Dict]) -> Dict[str, Any]:
        """Analyze heading heights and patterns to establish classification rules."""
        headings = []
        for page in pages:
            for item in page.get('items', []):
                if item.get('type') == 'heading':
                    height = item.get('bBox', {}).get('h', 0)
                    value = item.get('value', '').strip()
                    numbering_info = self._analyze_numbering(value)
                    headings.append({
                        'height': height,
                        'value': value,
                        'numbering_info': numbering_info,
                        'has_numbering': numbering_info['has_numbering'],
                        'numbering_depth': numbering_info['depth'],
                        'is_title': self._is_title_pattern(value),
                        'is_ui_element': self._is_likely_ui_element(value),
                    })
        
        if not headings:
            return {'min_valid_height': 10, 'height_levels': {}}
        
        # Group by validity
        valid_headings = [h for h in headings if h['has_numbering'] or h['is_title']]
        
        # Determine min valid height
        min_valid_height = 10
        if valid_headings:
            valid_heights = [h['height'] for h in valid_headings if h['height'] > 0]
            if valid_heights:
                min_valid_height = min(valid_heights) * 0.85
        
        # Build height-to-level mapping for non-numbered headings
        height_levels = {}
        if valid_headings:
            unique_heights = sorted(set(h['height'] for h in valid_headings if h['height'] > 0), reverse=True)
            for i, height in enumerate(unique_heights[:6]):
                height_levels[height] = i + 1
        
        return {
            'min_valid_height': min_valid_height,
            'height_levels': height_levels,
            'valid_headings': valid_headings,
        }
    
    def _analyze_numbering(self, text: str) -> Dict[str, Any]:
        """Analyze numbering pattern and return detailed info."""
        clean = re.sub(r'^\[[\w\s]+\]\s*', '', text.strip())
        
        result = {
            'has_numbering': False,
            'type': None,
            'depth': 0,
            'parts': []
        }
        
        # Pattern 1: Roman numerals with dots (I., II., III., IV., etc.)
        roman_match = re.match(r'^([IVXLCDMivxlcdm]+)\s*\.\s+(.+)$', clean)
        if roman_match:
            roman = roman_match.group(1).upper()
            # Validate it's a proper Roman numeral
            if self._is_valid_roman(roman):
                result['has_numbering'] = True
                result['type'] = 'roman'
                result['depth'] = 1
                result['parts'] = [roman]
                return result
        
        # Pattern 2: Roman.Arabic (I.1., II.2., etc.)
        roman_arabic_match = re.match(r'^([IVXLCDMivxlcdm]+)\.(\d+)\s*\.?\s+(.+)$', clean)
        if roman_arabic_match:
            roman = roman_arabic_match.group(1).upper()
            arabic = roman_arabic_match.group(2)
            if self._is_valid_roman(roman):
                result['has_numbering'] = True
                result['type'] = 'roman_arabic'
                result['depth'] = 2
                result['parts'] = [roman, arabic]
                return result
        
        # Pattern 3: Arabic decimal numbering (1., 1.1, 1.1.1, etc.)
        decimal_match = re.match(r'^(\d+(?:\.\d+)*)\s*\.?\s+(.+)$', clean)
        if decimal_match:
            parts = decimal_match.group(1).split('.')
            result['has_numbering'] = True
            result['type'] = 'decimal'
            result['depth'] = len(parts)
            result['parts'] = parts
            return result
        
        # Pattern 4: Single letter (A., B., etc.)
        letter_match = re.match(r'^([A-Z])\s*\.\s+(.+)$', clean)
        if letter_match:
            result['has_numbering'] = True
            result['type'] = 'letter'
            result['depth'] = 1
            result['parts'] = [letter_match.group(1)]
            return result
        
        # Pattern 5: Parenthetical numbering ((1), (a), etc.)
        paren_match = re.match(r'^\((\d+|[a-z])\)\s+(.+)$', clean)
        if paren_match:
            result['has_numbering'] = True
            result['type'] = 'parenthetical'
            result['depth'] = 3  # Usually used for sub-sub items
            result['parts'] = [paren_match.group(1)]
            return result
        
        return result
    
    def _is_valid_roman(self, text: str) -> bool:
        """Check if text is a valid Roman numeral."""
        if not text:
            return False
        
        text = text.upper()
        # Simple validation: only valid Roman numeral characters
        if not re.match(r'^[IVXLCDM]+$', text):
            return False
        
        # Common valid patterns
        valid_romans = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                        'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']
        return text in valid_romans or len(text) <= 4
    
    def _process_heading_item(self, item: Dict, stats: Dict) -> Optional[str]:
        value = item.get('value', '').strip()
        height = item.get('bBox', {}).get('h', 0)
        
        if not value:
            return None
        
        # Rule 1: Filter out UI elements
        if self._is_likely_ui_element(value):
            return value  # Return as plain text
        
        # Rule 2: Check height threshold (but don't filter numbered headings)
        numbering_info = self._analyze_numbering(value)
        min_valid_height = stats.get('min_valid_height', 10)
        
        if height < min_valid_height and not numbering_info['has_numbering']:
            return value  # Return as plain text
        
        # Rule 3: Determine heading level based on structure
        level = self._determine_heading_level_contextual(value, height, numbering_info, stats)
        
        if level:
            # Update heading context
            self._update_heading_context(level, numbering_info)
            return '#' * level + ' ' + value
        else:
            return value
    
    def _determine_heading_level_contextual(self, value: str, height: float, 
                                             numbering_info: Dict, stats: Dict) -> Optional[int]:
        """Determine heading level using context and structure."""
        
        # Priority 1: Numbered headings
        if numbering_info['has_numbering']:
            numbering_type = numbering_info['type']
            depth = numbering_info['depth']
            
            if numbering_type == 'roman':
                # I., II., III. -> H2 (main sections)
                return 2
            elif numbering_type == 'roman_arabic':
                # I.1, I.2 -> H3 (sub-sections)
                return 3
            elif numbering_type == 'decimal':
                # 1. -> H2, 1.1 -> H3, 1.1.1 -> H4
                return min(depth + 1, 6)
            elif numbering_type == 'letter':
                # A., B. -> H2
                return 2
            elif numbering_type == 'parenthetical':
                # (1), (a) -> H4 or context-based
                if self._current_heading_context:
                    return min(self._current_heading_context[-1][0] + 1, 6)
                return 4
        
        # Priority 2: Title patterns
        if self._is_title_pattern(value):
            return 1
        
        # Priority 3: Context-based level (for non-numbered sub-headings)
        # If we have a current context, this is likely a sub-heading
        if self._current_heading_context:
            parent_level = self._current_heading_context[-1][0]
            # Non-numbered heading after a numbered one is typically one level deeper
            return min(parent_level + 1, 6)
        
        # Priority 4: Height-based level
        height_levels = stats.get('height_levels', {})
        if height in height_levels:
            return height_levels[height]
        
        # Find closest height
        if height_levels:
            closest_height = min(height_levels.keys(), key=lambda h: abs(h - height))
            if abs(closest_height - height) < 3:
                return height_levels[closest_height]
        
        # Default: H3 for unclassified headings (not H2 to avoid conflicts with main sections)
        return 3
    
    def _update_heading_context(self, level: int, numbering_info: Dict):
        """Update the heading context stack."""
        depth = numbering_info.get('depth', 0)
        
        # Pop levels that are >= current (siblings or children of siblings)
        while self._current_heading_context and self._current_heading_context[-1][0] >= level:
            self._current_heading_context.pop()
        
        # Push current heading
        self._current_heading_context.append((level, depth))
    
    def _has_numbering_pattern(self, text: str) -> bool:
        """Quick check if text has any numbering pattern."""
        return self._analyze_numbering(text)['has_numbering']
    
    def _get_numbering_depth(self, text: str) -> int:
        """Get the depth of numbering (for compatibility)."""
        return self._analyze_numbering(text)['depth']
    
    def _is_title_pattern(self, text: str) -> bool:
        text_lower = text.lower().strip()
        
        title_patterns = [
            r'^(module|chapter|chương|phần|part|unit|bài|mục)\s*\d*',
            r'^(introduction|conclusion|summary|overview|abstract)',
            r'^(giới thiệu|kết luận|tổng quan|mô tả|tóm tắt)',
            r'^(requirements?|specifications?|mục tiêu|yêu cầu)',
            r'^(appendix|phụ lục|references?|tài liệu)',
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, text_lower):
                return True
        return False
    
    def _is_likely_ui_element(self, text: str) -> bool:
        text_lower = text.lower().strip()
        original = text.strip()
        
        # 0. Check for bullet/symbol prefixes (these are list items, not headings)
        if re.match(r'^[◼◻●○•◦▪▸►⇒→¢¶§]\s*', original):
            # But allow if it's followed by substantive content that looks like a heading
            remaining = re.sub(r'^[◼◻●○•◦▪▸►⇒→¢¶§]\s*', '', original)
            if len(remaining) < 10 or not any(c.isupper() for c in remaining[:5]):
                return True  # Likely a bullet point
        
        # 1. Exact match UI keywords
        ui_keywords = {
            'ok', 'cancel', 'save', 'add', 'edit', 'delete', 'create', 'join',
            'yes', 'no', 'confirm', 'apply', 'reset', 'close', 'back', 'next',
            'home', 'menu', 'settings', 'profile', 'logout', 'login',
            'chat', 'meet', 'call', 'share', 'post', 'send', 'reply',
            'members', 'owners', 'admins', 'users', 'channels', 'rooms',
            'notifications', 'alerts', 'messages', 'inbox', 'activity', 'feed',
        }
        
        if text_lower in ui_keywords:
            return True
        
        # 2. "Number + Word" pattern (e.g., "8 Events") - likely UI counter
        if re.match(r'^\d+\s+[A-Za-z]+s?$', original):
            return True
        
        # 3. Very short text (1-2 words) that doesn't look like a heading
        words = text.split()
        if len(words) <= 2 and len(text) < 20:
            # But allow if it has heading-like patterns
            if not self._has_numbering_pattern(text) and not self._is_title_pattern(text):
                # Check if it looks like a section name
                if not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', original):
                    return True
        
        # 4. Footer/header patterns
        footer_patterns = [
            r'^(page\s+)?\d+$',  # Page numbers
            r'^\d+\s*/\s*\d+$',  # Page X/Y
            r'^AI\s+VIET\s+NAM',  # Document footer
            r'@.*\.edu\.vn',  # Email
            r'^AIO\d+',  # Course code
        ]
        
        for pattern in footer_patterns:
            if re.match(pattern, original, re.IGNORECASE):
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
