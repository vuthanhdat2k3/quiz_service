from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class DocumentNode:
    id: str
    title: str
    source: str
    language: str = "vi"
    llamaparse_used: bool = False
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class SectionNode:
    id: str
    document_id: str
    title: str
    level: int  # 1=H1, 2=H2, ..., 6=H6
    order: int  # Sequential order in document
    number: Optional[str] = None  # e.g., "1", "1.1", "1.1.1"
    path: List[str] = field(default_factory=list)  # e.g., ["Chương 1", "1.1 Khái niệm"]
    parent_id: Optional[str] = None  # Parent section ID
    
    def get_path_string(self) -> str:
        return " > ".join(self.path) if self.path else ""


@dataclass
class ChunkNode:
    id: str
    document_id: str
    section_id: str
    text: str
    token_length: int
    order: int  # Sequential order in document
    chunk_type: str = "paragraph"
    features: dict = field(default_factory=dict)
    embedding_id: Optional[str] = None
