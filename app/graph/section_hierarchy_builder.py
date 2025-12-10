from typing import List, Optional, Dict
from loguru import logger

from .models import SectionNode


class SectionHierarchyBuilder:
    
    def __init__(self, document_id: str):
        self.document_id = document_id
        self.level_stack: Dict[int, SectionNode] = {}  # level -> last section at that level
        
    def build_hierarchy(self, sections: List[SectionNode]) -> List[SectionNode]:
        # Sort by order to process sequentially
        sorted_sections = sorted(sections, key=lambda s: s.order)
        
        logger.info(f"Building hierarchy for {len(sorted_sections)} sections")
        
        for section in sorted_sections:
            # Determine parent
            parent_id = self._find_parent(section)
            section.parent_id = parent_id
            
            # Update stack: this section is now the most recent at its level
            self.level_stack[section.level] = section
            
            # Clear deeper levels (they can't be parents anymore)
            self._clear_deeper_levels(section.level)
            
            logger.debug(
                f"Section '{section.title}' (level {section.level}, order {section.order}) "
                f"-> parent: {parent_id or 'Document'}"
            )
        
        return sorted_sections
    
    def _find_parent(self, section: SectionNode) -> Optional[str]:
        if section.level == 1:
            # H1 sections are children of Document
            return None
        
        # Find parent at level-1
        parent_level = section.level - 1
        parent_section = self.level_stack.get(parent_level)
        
        if parent_section:
            return parent_section.id
        
        # If no parent found at level-1, search upwards
        # This handles cases where heading levels are skipped
        for level in range(parent_level - 1, 0, -1):
            if level in self.level_stack:
                logger.warning(
                    f"Section '{section.title}' (level {section.level}) "
                    f"skipped level {parent_level}, using level {level} as parent"
                )
                return self.level_stack[level].id
        
        # No parent found, attach to Document
        logger.warning(
            f"Section '{section.title}' (level {section.level}) "
            f"has no parent, attaching to Document"
        )
        return None
    
    def _clear_deeper_levels(self, current_level: int):
        levels_to_remove = [level for level in self.level_stack.keys() if level > current_level]
        for level in levels_to_remove:
            del self.level_stack[level]
    
    def get_hierarchy_tree(self, sections: List[SectionNode]) -> Dict:
        # Build children map
        children_map: Dict[Optional[str], List[SectionNode]] = {}
        for section in sections:
            parent_id = section.parent_id
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(section)
        
        def build_tree(parent_id: Optional[str]) -> List[Dict]:
            """Recursively build tree."""
            if parent_id not in children_map:
                return []
            
            tree = []
            for section in sorted(children_map[parent_id], key=lambda s: s.order):
                node = {
                    "id": section.id,
                    "title": section.title,
                    "level": section.level,
                    "order": section.order,
                    "children": build_tree(section.id)
                }
                tree.append(node)
            return tree
        
        # Start from root (sections with parent_id = None)
        return {
            "document_id": self.document_id,
            "sections": build_tree(None)
        }
