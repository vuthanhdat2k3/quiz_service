import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

@dataclass
class MarkdownChunk:
    chunk_id: str
    text: str
    token_count: int
    chunk_type: str  # 'heading', 'content', 'table', 'code', 'mixed'
    section_path: List[str] = field(default_factory=list)  # [H1, H2, H3, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

class MarkdownChunkerV2:
    
    # Header patterns for Markdown
    HEADER_PATTERNS = [
        (r'^######\s+(.+)$', 'H6', 6),
        (r'^#####\s+(.+)$', 'H5', 5),
        (r'^####\s+(.+)$', 'H4', 4),
        (r'^###\s+(.+)$', 'H3', 3),
        (r'^##\s+(.+)$', 'H2', 2),
        (r'^#\s+(.+)$', 'H1', 1),
    ]
    
    # Split point patterns with priorities
    SPLIT_PATTERNS = [
        (r'\n#{1,6}\s+', 5, 'header'),           # Headers
        (r'\n\*{3,}\n|\n-{3,}\n|\n_{3,}\n', 4, 'horizontal_rule'),  # Horizontal rules
        (r'\n\n(?=\*\*[^*]+\*\*)', 3, 'bold_start'),  # Bold paragraph starts
        (r'\n\n+', 2, 'paragraph'),              # Paragraph breaks
        (r'\n(?=[-*+]\s)', 1, 'list_item'),      # List items
    ]
    
    # TOC patterns
    TOC_PATTERNS = [
        r'^\s*[-*]\s+\[.+\]\(#.+\)\s*$',  # [Title](#anchor)
        r'^\s*\d+\.\s+\[.+\]\(#.+\)\s*$', # 1. [Title](#anchor)
        r'^\s*[-*]\s+.+\.{2,}\s*\d+\s*$', # Title...... 1
    ]
    
    def __init__(
        self,
        max_tokens: int = 500,
        min_tokens: int = 100,
        overlap_tokens: int = 0,
        model_name: str = "gpt-4"
    ):

        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.model_name = model_name
        
        # Cache for section IDs
        self._section_id_cache: Dict[str, str] = {}
        
        logger.info(f"MarkdownChunkerV2 initialized: max_tokens={max_tokens}, min_tokens={min_tokens}")
    
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        
        # Simple approximation: ~4 chars per token (English)
        # More accurate than word count for mixed content
        words = len(text.split())
        chars = len(text)
        
        # Use average of word-based and char-based estimates
        word_estimate = words * 1.3  # English words ~1.3 tokens
        char_estimate = chars / 4
        
        return int((word_estimate + char_estimate) / 2)
    
######## Step1: Header-Based Splitting ########
    
    def _step1_basic_chunking(self, content: str) -> List[Dict[str, Any]]:

        logger.info("🔄 Step 1: Header-based splitting")
        
        # Define headers to split on
        headers_to_split = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
            ("#####", "H5"),
            ("######", "H6"),
        ]
        
        # Use langchain splitter
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split,
            strip_headers=False
        )
        
        try:
            splits = splitter.split_text(content)
        except Exception as e:
            logger.warning(f"Header splitting failed: {e}, falling back to simple split")
            return self._fallback_split(content)
        
        # Convert to internal format
        chunks = []
        for i, split in enumerate(splits):
            chunk_content = split.page_content
            token_count = self.count_tokens(chunk_content)
            
            # Skip empty chunks
            if not chunk_content.strip():
                continue
            
            # Detect chunk type
            chunk_type = self._detect_chunk_type(chunk_content)
            
            # Extract metadata
            metadata = dict(split.metadata) if hasattr(split, 'metadata') else {}
            
            # Mark tables
            if self._contains_table(chunk_content):
                metadata['is_table'] = True
                if chunk_type == 'content':
                    chunk_type = 'table'
            
            # Mark code blocks
            if self._contains_code_block(chunk_content):
                metadata['has_code'] = True
                if chunk_type == 'content':
                    chunk_type = 'code'
            
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'content': chunk_content,
                'length': token_count,
                'chunk_type': chunk_type,
                'metadata': metadata
            })
        
        logger.info(f"✅ Step 1: {len(chunks)} chunks from header split")
        return chunks
    
    def _fallback_split(self, content: str) -> List[Dict[str, Any]]:
        paragraphs = content.split('\n\n')
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'content': para,
                'length': self.count_tokens(para),
                'chunk_type': self._detect_chunk_type(para),
                'metadata': {}
            })
        
        return chunks
    
    def _detect_chunk_type(self, content: str) -> str:
        content_stripped = content.strip()
        
        # Check for header
        for pattern, _, _ in self.HEADER_PATTERNS:
            if re.match(pattern, content_stripped, re.MULTILINE):
                return 'heading'
        
        # Check for table
        if self._contains_table(content_stripped):
            return 'table'
        
        # Check for code block
        if self._contains_code_block(content_stripped):
            return 'code'
        
        return 'content'
    
    def _contains_table(self, content: str) -> bool:
        lines = content.split('\n')
        
        # Pattern 1: Pipe tables (| col1 | col2 |)
        pipe_rows = [l for l in lines if '|' in l and l.strip().startswith('|')]
        if len(pipe_rows) >= 2:
            return True
        
        # Pattern 2: Simple pipe tables (col1 | col2)
        simple_pipe_rows = [l for l in lines if re.search(r'\S\s*\|\s*\S', l)]
        if len(simple_pipe_rows) >= 2:
            return True
        
        # Pattern 3: Grid tables (+---+---+)
        grid_markers = [l for l in lines if re.match(r'^\+[-=+]+\+$', l.strip())]
        if len(grid_markers) >= 2:
            return True
        
        return False
    
    def _contains_code_block(self, content: str) -> bool:
        return bool(re.search(r'```[\s\S]*?```', content))
    
######## Step2: Hierarchical Merging ########
    
    def _step2_merge_hierarchical(self, step1_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        logger.info("🔄 Step 2: Hierarchical merging")
        
        if not step1_chunks:
            return []
        
        # Build header hierarchy tree
        tree = self._build_header_tree(step1_chunks)
        
        # Bottom-up merge
        merged = self._merge_bottom_up(tree)
        
        # Filter empty header chunks
        merged = [c for c in merged if not self._is_empty_header_chunk(c)]
        
        logger.info(f"✅ Step 2: {len(merged)} chunks after merge")
        return merged
    
    def _build_header_tree(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if not chunks:
            return []
        
        # Add level and parent tracking
        for chunk in chunks:
            chunk['level'] = self._get_chunk_level(chunk)
            chunk['children'] = []
            chunk['parent_idx'] = None
        
        # Build parent-child relationships
        for i, chunk in enumerate(chunks):
            if i == 0:
                continue
            
            # Find parent (nearest preceding chunk with lower level)
            parent_idx = self._find_parent(chunks, i)
            if parent_idx is not None:
                chunk['parent_idx'] = parent_idx
                chunks[parent_idx]['children'].append(i)
        
        return chunks
    
    def _get_chunk_level(self, chunk: Dict[str, Any]) -> int:

        metadata = chunk.get('metadata', {})
        
        # Check for explicit header in metadata
        for i in range(6, 0, -1):
            if f'H{i}' in metadata:
                return i
        
        # Check content for header
        content = chunk.get('content', '')
        for pattern, _, level in self.HEADER_PATTERNS:
            if re.match(pattern, content.strip(), re.MULTILINE):
                return level
        
        return 0  # No header
    
    def _get_effective_chunk_level(self, chunk: Dict[str, Any]) -> int:

        metadata = chunk.get('metadata', {})
        effective_level = 0
        
        # Find the deepest header level in metadata
        for i in range(1, 7):
            if f'H{i}' in metadata:
                effective_level = i
        
        # Use content level if no metadata
        if effective_level == 0:
            effective_level = self._get_chunk_level(chunk)
        
        return effective_level
    
    def _find_parent(self, chunks: List[Dict[str, Any]], current_idx: int) -> Optional[int]:

        current_level = self._get_effective_chunk_level(chunks[current_idx])
        
        # If no header level, attach to nearest preceding chunk with any header
        if current_level == 0:
            for i in range(current_idx - 1, -1, -1):
                if self._get_effective_chunk_level(chunks[i]) > 0:
                    return i
            return None
        
        # Find nearest preceding chunk with lower level
        for i in range(current_idx - 1, -1, -1):
            parent_level = self._get_effective_chunk_level(chunks[i])
            if 0 < parent_level < current_level:
                return i
        
        return None
    
    def _merge_bottom_up(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if not chunks:
            return []
        
        # Process from deepest level up
        max_level = max(c.get('level', 0) for c in chunks)
        
        for level in range(max_level, 0, -1):
            chunks = self._merge_level(chunks, level)
        
        # Merge siblings at each level
        chunks = self._merge_siblings(chunks)
        
        return chunks
    
    def _merge_level(self, chunks: List[Dict[str, Any]], level: int) -> List[Dict[str, Any]]:

        merged = []
        skip_indices = set()
        
        for i, chunk in enumerate(chunks):
            if i in skip_indices:
                continue
            
            if chunk.get('level', 0) == level:
                # Try to merge with parent if small
                parent_idx = chunk.get('parent_idx')
                if parent_idx is not None and chunk['length'] < self.min_tokens:
                    parent = chunks[parent_idx]
                    combined_length = parent['length'] + chunk['length']
                    
                    if combined_length <= self.max_tokens:
                        # Merge into parent
                        parent['content'] = parent['content'] + '\n\n' + chunk['content']
                        parent['length'] = combined_length
                        skip_indices.add(i)
                        continue
            
            merged.append(chunk)
        
        return merged
    
    def _merge_siblings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if len(chunks) <= 1:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # Try to merge with next if both are small
            if current['length'] < self.min_tokens and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                # Check if same level (siblings)
                if current.get('level', 0) == next_chunk.get('level', 0):
                    combined_length = current['length'] + next_chunk['length']
                    
                    if combined_length <= self.max_tokens:
                        # Merge
                        merged_chunk = {
                            'chunk_id': str(uuid.uuid4()),
                            'content': current['content'] + '\n\n' + next_chunk['content'],
                            'length': combined_length,
                            'chunk_type': 'mixed',
                            'metadata': self._merge_metadata(current['metadata'], next_chunk['metadata']),
                            'level': current.get('level', 0)
                        }
                        merged.append(merged_chunk)
                        i += 2
                        continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _merge_metadata(self, meta1: Dict[str, Any], meta2: Dict[str, Any]) -> Dict[str, Any]:

        result = dict(meta1)
        
        for key, value in meta2.items():
            if key not in result:
                result[key] = value
            elif key.startswith('H') and key[1:].isdigit():
                # Keep the first header
                pass
            else:
                # Merge other metadata
                if isinstance(result[key], list) and isinstance(value, list):
                    result[key] = result[key] + value
                elif isinstance(result[key], dict) and isinstance(value, dict):
                    result[key].update(value)
        
        return result
    
    def _is_empty_header_chunk(self, chunk: Dict[str, Any]) -> bool:

        content = chunk.get('content', '').strip()
        
        # Remove headers and check if anything remains
        content_without_headers = re.sub(r'^#{1,6}\s+.+$', '', content, flags=re.MULTILINE)
        content_without_headers = content_without_headers.strip()
        
        return len(content_without_headers) < 10
    
####### Step3: Splitting Oversized Chunks ########
    
    def _step3_split_oversized(self, step2_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        logger.info("🔄 Step 3: Splitting oversized chunks")
        
        result = []
        split_count = 0
        table_split_count = 0
        
        for chunk in step2_chunks:
            if chunk['length'] <= self.max_tokens:
                result.append(chunk)
                continue
            
            # Oversized - need to split
            if chunk.get('metadata', {}).get('is_table', False):
                # Table splitting
                parts = self._split_table_by_rows(chunk['content'])
                table_split_count += 1
            else:
                # Content splitting
                parts = self._split_content_intelligently(chunk['content'])
            
            if len(parts) > 1:
                split_count += 1
                for j, part in enumerate(parts):
                    split_chunk = self._create_split_chunk(chunk, part, j, len(parts))
                    result.append(split_chunk)
            else:
                result.append(chunk)
        
        if table_split_count > 0:
            logger.info(f"📊 Split {table_split_count} table chunks")
        
        logger.info(f"✅ Step 3: {len(result)} chunks (split {split_count} oversized)")
        return result
    
    def _split_table_by_rows(self, content: str) -> List[str]:

        lines = content.split('\n')
        
        # Find table structure
        table_start = None
        header_rows = []
        separator_idx = None
        data_rows = []
        non_table_before = []
        non_table_after = []
        
        in_table = False
        for i, line in enumerate(lines):
            is_table_line = '|' in line or re.match(r'^\+[-=+]+\+$', line.strip())
            is_separator = bool(re.match(r'^[\|\+\s:-]+$', line.strip()) and ('-' in line or '=' in line))
            
            if is_table_line and not in_table:
                in_table = True
                table_start = i
            
            if in_table:
                if is_separator and separator_idx is None:
                    separator_idx = len(header_rows)
                    header_rows.append(line)
                elif is_table_line:
                    if separator_idx is None:
                        header_rows.append(line)
                    else:
                        data_rows.append(line)
                else:
                    # End of table
                    non_table_after = lines[i:]
                    break
            else:
                non_table_before.append(line)
        
        if not data_rows:
            # Can't split - return as is
            return [content]
        
        # Calculate how many rows per part
        header_text = '\n'.join(header_rows)
        header_tokens = self.count_tokens(header_text)
        
        target_tokens = self.max_tokens - header_tokens - 50  # Buffer
        if target_tokens < 50:
            target_tokens = 50
        
        # Split data rows
        parts = []
        current_rows = []
        current_tokens = 0
        
        non_table_prefix = '\n'.join(non_table_before).strip()
        
        for row in data_rows:
            row_tokens = self.count_tokens(row)
            
            if current_tokens + row_tokens > target_tokens and current_rows:
                # Create part
                part_content = []
                if parts == [] and non_table_prefix:
                    part_content.append(non_table_prefix)
                    part_content.append('')
                part_content.append(header_text)
                part_content.extend(current_rows)
                
                parts.append('\n'.join(part_content))
                current_rows = []
                current_tokens = 0
            
            current_rows.append(row)
            current_tokens += row_tokens
        
        # Last part
        if current_rows:
            part_content = []
            if parts == [] and non_table_prefix:
                part_content.append(non_table_prefix)
                part_content.append('')
            part_content.append(header_text)
            part_content.extend(current_rows)
            
            non_table_suffix = '\n'.join(non_table_after).strip()
            if non_table_suffix:
                part_content.append('')
                part_content.append(non_table_suffix)
            
            parts.append('\n'.join(part_content))
        
        return parts if parts else [content]
    
    def _split_content_intelligently(self, content: str) -> List[str]:

        if self.count_tokens(content) <= self.max_tokens:
            return [content]
        
        parts = []
        remaining = content
        
        while self.count_tokens(remaining) > self.max_tokens:
            target_tokens = int(self.max_tokens * 0.75)
            
            # Find best split point
            split_pos = self._find_best_split(remaining, target_tokens)
            
            if split_pos is None or split_pos <= 0:
                # Fallback to character-based split
                ratio = target_tokens / self.count_tokens(remaining)
                split_pos = int(len(remaining) * ratio)
                split_pos = self._adjust_to_line_boundary(remaining, split_pos)
            
            first_part = remaining[:split_pos].strip()
            
            if first_part:
                # Verify size
                if self.count_tokens(first_part) > self.max_tokens:
                    # Reduce target
                    target_tokens = int(self.max_tokens * 0.5)
                    split_pos = self._find_best_split(remaining, target_tokens)
                    
                    if split_pos is None or split_pos <= 0:
                        ratio = target_tokens / self.count_tokens(remaining)
                        split_pos = max(1, int(len(remaining) * ratio))
                    
                    first_part = remaining[:split_pos].strip()
                
                if first_part:
                    parts.append(first_part)
            
            remaining = remaining[split_pos:].strip()
            
            # Safety limit
            if len(parts) > 20:
                break
        
        if remaining.strip():
            parts.append(remaining.strip())
        
        return parts if parts else [content]
    
    def _find_best_split(self, content: str, target_tokens: int) -> Optional[int]:
        # Estimate target character position
        ratio = target_tokens / max(1, self.count_tokens(content))
        target_pos = int(len(content) * ratio)
        
        # Search window
        window_start = max(0, target_pos - 500)
        window_end = min(len(content), target_pos + 500)
        window = content[window_start:window_end]
        
        # Find split points in window
        split_points = self._find_split_points(window)
        
        if not split_points:
            return None
        
        # Score each split point
        best_split = None
        best_score = -1
        
        for pos, priority, _ in split_points:
            abs_pos = window_start + pos
            distance = abs(abs_pos - target_pos)
            max_distance = max(500, target_pos)
            distance_factor = 1 - (distance / max_distance)
            
            score = priority * distance_factor
            
            if score > best_score:
                best_score = score
                best_split = abs_pos
        
        if best_split is not None:
            best_split = self._adjust_to_line_boundary(content, best_split)
        
        return best_split
    
    def _find_split_points(self, content: str) -> List[Tuple[int, int, str]]:

        points = []
        
        for pattern, priority, split_type in self.SPLIT_PATTERNS:
            for match in re.finditer(pattern, content):
                pos = match.start()
                
                # Skip TOC sections
                if split_type == 'list_item' and self._is_toc_item(content, pos):
                    continue
                
                points.append((pos, priority, split_type))
        
        return points
    
    def _is_toc_item(self, content: str, position: int) -> bool:

        # Get the line containing this position
        line_start = content.rfind('\n', 0, position)
        line_end = content.find('\n', position)
        
        if line_start == -1:
            line_start = 0
        if line_end == -1:
            line_end = len(content)
        
        line = content[line_start:line_end]
        
        # Check TOC patterns
        for pattern in self.TOC_PATTERNS:
            if re.match(pattern, line.strip()):
                return True
        
        return False
    
    def _adjust_to_line_boundary(self, content: str, position: int) -> int:

        if position <= 0 or position >= len(content):
            return position
        
        prev_newline = content.rfind('\n', 0, position)
        next_newline = content.find('\n', position)
        
        if prev_newline == -1:
            return 0 if next_newline == -1 else next_newline + 1
        
        if next_newline == -1:
            return len(content)
        
        # Choose closer boundary
        if position - prev_newline <= next_newline - position:
            return prev_newline + 1
        else:
            return next_newline + 1
    
    def _create_split_chunk(
        self, 
        original: Dict[str, Any], 
        content: str, 
        part_idx: int, 
        total_parts: int
    ) -> Dict[str, Any]:

        new_metadata = dict(original.get('metadata', {}))
        new_metadata['split_from'] = original['chunk_id']
        new_metadata['part'] = f"{part_idx + 1}/{total_parts}"
        new_metadata['original_length'] = original['length']
        
        return {
            'chunk_id': str(uuid.uuid4()),
            'content': content,
            'length': self.count_tokens(content),
            'chunk_type': original.get('chunk_type', 'content'),
            'metadata': new_metadata,
            'level': original.get('level', 0)
        }
    
    def _convert_to_markdown_chunks(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[MarkdownChunk]:

        result = []
        
        for chunk in chunks:
            # Build section path
            section_path = []
            metadata = chunk.get('metadata', {})
            
            for i in range(1, 7):
                key = f'H{i}'
                if key in metadata:
                    section_path.append(metadata[key])
            
            # Generate section_id from path
            section_id = self._get_section_id(tuple(section_path))
            
            # Get heading (last header in path)
            heading = section_path[-1] if section_path else ''
            level = len(section_path)
            
            # Build enhanced metadata
            enhanced_metadata = dict(metadata)
            enhanced_metadata['section_id'] = section_id
            enhanced_metadata['level'] = level
            enhanced_metadata['heading'] = heading
            enhanced_metadata['section_path'] = section_path
            
            md_chunk = MarkdownChunk(
                chunk_id=chunk['chunk_id'],
                text=chunk['content'],
                token_count=chunk['length'],
                chunk_type=chunk.get('chunk_type', 'content'),
                section_path=section_path,
                metadata=enhanced_metadata
            )
            
            result.append(md_chunk)
        
        return result
    
    def _get_section_id(self, path: tuple) -> str:

        if not path:
            return str(uuid.uuid4())
        
        if path not in self._section_id_cache:
            self._section_id_cache[path] = str(uuid.uuid4())
        
        return self._section_id_cache[path]
    

    def parse(self, content: str) -> List[MarkdownChunk]:

        logger.info(f"🚀 Parsing content ({len(content)} chars)")
        
        # Clear section cache for new document
        self._section_id_cache.clear()
        
        # 3-step processing
        step1 = self._step1_basic_chunking(content)
        step2 = self._step2_merge_hierarchical(step1)
        step3 = self._step3_split_oversized(step2)
        
        # Convert to MarkdownChunk format
        result = self._convert_to_markdown_chunks(step3)
        
        logger.info(f"✅ Parsed: {len(result)} chunks")
        return result
    
    def process_content(
        self, 
        content: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:

        logger.info(f"🚀 Processing content ({len(content)} chars)")
        
        step1 = self._step1_basic_chunking(content)
        step2 = self._step2_merge_hierarchical(step1)
        step3 = self._step3_split_oversized(step2)
        
        logger.info(f"✅ Completed: {len(step3)} final chunks")
        return step1, step2, step3
    
    def chunk(self, text: str) -> List[str]:
        chunks = self.parse(text)
        return [c.text for c in chunks]
    
    def format_chunks_for_output(
        self, 
        chunks: List[MarkdownChunk], 
        file_name: str
    ) -> List[Dict[str, Any]]:

        result = []
        
        for i, chunk in enumerate(chunks):
            # Build metadata
            meta = {}
            for j, header in enumerate(chunk.section_path, 1):
                meta[f'H{j}'] = header
            
            meta['file_name'] = file_name
            meta['position'] = i + 1
            meta['section_id'] = chunk.metadata.get('section_id', '')
            
            formatted = {
                'chunk_id': chunk.chunk_id,
                'embedding': 'None',
                'length': str(chunk.token_count),
                'metadata': meta,
                'chunk_text': chunk.text
            }
            
            result.append(formatted)
        
        return result
    
    def get_section_metadata(self, chunks: List[MarkdownChunk]) -> List[Dict[str, Any]]:
        sections = {}
        
        for chunk in chunks:
            section_id = chunk.metadata.get('section_id')
            if not section_id or section_id in sections:
                continue
            
            path = chunk.section_path
            level = len(path)
            heading = path[-1] if path else ''
            
            # Find parent section
            parent_section_id = None
            if len(path) > 1:
                parent_path = tuple(path[:-1])
                parent_section_id = self._section_id_cache.get(parent_path)
            
            sections[section_id] = {
                'section_id': section_id,
                'heading': heading,
                'level': level,
                'parent_section_id': parent_section_id,
                'path': path
            }
        
        return list(sections.values())

def create_chunker(
    max_tokens: int = 500,
    min_tokens: int = 100
) -> MarkdownChunkerV2:
    return MarkdownChunkerV2(
        max_tokens=max_tokens,
        min_tokens=min_tokens
    )


def chunk_markdown(
    content: str,
    max_tokens: int = 500,
    min_tokens: int = 100
) -> List[MarkdownChunk]:
    chunker = create_chunker(max_tokens, min_tokens)
    return chunker.parse(content)

