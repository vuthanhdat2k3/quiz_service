from typing import List, Optional, Dict
from loguru import logger

from .models import SectionNode, ChunkNode


class ChunkAssigner:
    
    def __init__(self, document_id: str):
        self.document_id = document_id
    
    def assign_chunks_to_sections(
        self, 
        chunks: List[ChunkNode], 
        sections: List[SectionNode],
        default_section_id: Optional[str] = None
    ) -> List[ChunkNode]:
        
        if not sections:
            logger.warning("No sections provided, using default section for all chunks")
            for chunk in chunks:
                chunk.section_id = default_section_id or "default"
            return chunks
        
        # Sort sections by order for efficient lookup
        sorted_sections = sorted(sections, key=lambda s: s.order)
        
        logger.info(f"Assigning {len(chunks)} chunks to {len(sections)} sections")
        
        for chunk in chunks:
            section = self._find_nearest_section(chunk, sorted_sections)
            
            if section:
                chunk.section_id = section.id
                logger.debug(
                    f"Chunk {chunk.order} assigned to section '{section.title}' "
                    f"(section order: {section.order})"
                )
            else:
                chunk.section_id = default_section_id or "default"
                logger.warning(
                    f"Chunk {chunk.order} has no matching section, "
                    f"assigned to default: {chunk.section_id}"
                )
        
        return chunks
    
    def _find_nearest_section(
        self, 
        chunk: ChunkNode, 
        sorted_sections: List[SectionNode]
    ) -> Optional[SectionNode]:
        
        # Find last section where section.order < chunk.order
        nearest_section = None
        
        for section in sorted_sections:
            if section.order < chunk.order:
                nearest_section = section
            else:
                # Since sections are sorted, no need to continue
                break
        
        return nearest_section
    
    def create_chunk_chain(self, chunks: List[ChunkNode]) -> List[tuple]:
        
        # Sort by order
        sorted_chunks = sorted(chunks, key=lambda c: c.order)
        
        chain = []
        for i in range(len(sorted_chunks) - 1):
            current_chunk = sorted_chunks[i]
            next_chunk = sorted_chunks[i + 1]
            chain.append((current_chunk.id, next_chunk.id))
        
        logger.info(f"Created chunk chain with {len(chain)} NEXT relationships")
        return chain
    
    def get_section_statistics(
        self, 
        chunks: List[ChunkNode], 
        sections: List[SectionNode]
    ) -> Dict[str, Dict]:
        
        from collections import defaultdict
        
        stats = defaultdict(lambda: {
            "title": "",
            "level": 0,
            "chunk_count": 0,
            "total_tokens": 0
        })
        
        # Initialize with section info
        for section in sections:
            stats[section.id]["title"] = section.title
            stats[section.id]["level"] = section.level
        
        # Count chunks and tokens
        for chunk in chunks:
            stats[chunk.section_id]["chunk_count"] += 1
            stats[chunk.section_id]["total_tokens"] += chunk.token_length
        
        return dict(stats)
