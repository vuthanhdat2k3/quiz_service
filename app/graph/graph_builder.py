from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import json
import re

from app.database.neo4j_db import Neo4jDatabase
from .models import DocumentNode, SectionNode, ChunkNode
from .section_hierarchy_builder import SectionHierarchyBuilder
from .chunk_assignment import ChunkAssigner
from .graph_schema import GraphSchema


class DocumentGraphBuilder:
    
    def __init__(self, db: Optional[Neo4jDatabase] = None):
        self.db = db or Neo4jDatabase()
        self.schema = GraphSchema(self.db)
    
    @staticmethod
    def extract_nodes_from_chunks(
        chunks: List[Any],
        document_id: str,
        file_name: str,
        file_path: str,
        language: str = "en",
        llamaparse_used: bool = False
    ) -> Tuple[DocumentNode, List[SectionNode], List[ChunkNode]]:
        
        # Create DocumentNode
        document = DocumentNode(
            id=document_id,
            title=file_name,
            source=file_path,
            language=language,
            llamaparse_used=llamaparse_used
        )
        
        # Extract sections from chunks with proper ordering
        sections_map = {}  # section_id -> section info
        section_order = 0
        
        for idx, chunk in enumerate(chunks):
            section_id = chunk.metadata.get("section_id")
            
            if section_id and section_id not in sections_map:
                section_path = chunk.section_path if hasattr(chunk, 'section_path') else []
                
                # Get level from metadata (for heading chunks) or calculate from path depth
                level = chunk.metadata.get("level")
                if level is None:
                    # If no level in metadata, infer from section_path depth
                    level = len(section_path) if section_path else 1
                
                # Get header from section_path or metadata
                header = section_path[-1] if section_path else chunk.metadata.get("heading", "Untitled Section")
                
                # Extract numbering from header if present (e.g., "1.1", "2.3.4")
                number_match = re.match(r'^([\d\.]+)', header)
                number = number_match.group(1) if number_match else None
                
                sections_map[section_id] = {
                    "section_id": section_id,
                    "header": header,
                    "level": level,
                    "path": section_path,
                    "number": number,
                    "order": section_order
                }
                section_order += 1
                
                logger.debug(
                    f"Extracted section: {header} (level={level}, order={section_order-1})"
                )
        
        # Convert to SectionNode objects
        section_nodes = []
        for section_id, section_info in sections_map.items():
            section_node = SectionNode(
                id=f"{document_id}_section_{section_id}",
                document_id=document_id,
                title=section_info["header"],
                level=section_info["level"],
                order=section_info["order"],
                number=section_info["number"],
                path=section_info["path"],
                parent_id=None  # Will be set by hierarchy builder
            )
            section_nodes.append(section_node)
            
            logger.debug(
                f"Section extracted: {section_info['header']} "
                f"(level={section_info['level']}, order={section_info['order']})"
            )
        
        logger.info(f"Extracted {len(section_nodes)} sections")
        
        # Create mapping from chunk metadata section_id to section node id
        section_id_map = {}  # chunk_metadata_section_id -> section_node_id
        for section_id, section_info in sections_map.items():
            section_node_id = f"{document_id}_section_{section_id}"
            section_id_map[section_id] = section_node_id
        
        # Convert chunks to ChunkNode objects with correct section_id
        chunk_nodes = []
        for idx, chunk in enumerate(chunks):
            # Get section_id from chunk metadata and map to section node id
            chunk_section_id = chunk.metadata.get("section_id", "")
            section_node_id = section_id_map.get(chunk_section_id, "")
            
            # Build features dict
            exclude_keys = {'section_id', 'chunk_id', 'section_path'}
            features = {"position": idx}
            
            for key, value in chunk.metadata.items():
                if key not in exclude_keys and isinstance(value, (str, int, float, bool)):
                    features[key] = value
            
            chunk_node = ChunkNode(
                id=f"{document_id}_chunk_{idx}",
                document_id=document_id,
                section_id=section_node_id,  # Use mapped section node id
                text=chunk.text,
                token_length=chunk.token_count,
                order=idx,
                chunk_type=chunk.chunk_type,
                features=features,
                embedding_id=None
            )
            chunk_nodes.append(chunk_node)
            
            logger.debug(
                f"Chunk {idx} mapped to section: {section_node_id if section_node_id else 'NO SECTION'}"
            )
        
        logger.info(f"Prepared {len(chunk_nodes)} chunks")
        
        return document, section_nodes, chunk_nodes
    
    def build_document_graph(
        self,
        document: DocumentNode,
        sections: List[SectionNode],
        chunks: List[ChunkNode]
    ) -> Dict[str, Any]:
        
        logger.info(f"Building graph for document: {document.title}")
        logger.info(f"  - Sections: {len(sections)}")
        logger.info(f"  - Chunks: {len(chunks)}")
        
        stats = {
            "document_id": document.id,
            "sections_created": 0,
            "chunks_created": 0,
            "relationships_created": 0
        }
        
        try:
            # Step 1: Create Document node
            logger.info("Step 1: Creating Document node...")
            self._create_document_node(document)
            
            # Step 2: Build section hierarchy
            logger.info("Step 2: Building section hierarchy...")
            hierarchy_builder = SectionHierarchyBuilder(document.id)
            sections_with_parents = hierarchy_builder.build_hierarchy(sections)
            
            # Log hierarchy tree for debugging
            tree = hierarchy_builder.get_hierarchy_tree(sections_with_parents)
            logger.debug(f"Hierarchy tree: {json.dumps(tree, indent=2, ensure_ascii=False)}")
            
            # Step 3: Create Section nodes and relationships
            logger.info("Step 3: Creating Section nodes...")
            self._create_section_nodes_with_hierarchy(
                document.id, 
                sections_with_parents
            )
            stats["sections_created"] = len(sections_with_parents)
            
            # Step 4: Verify chunk-section mapping and log statistics
            logger.info("Step 4: Verifying chunk-section mapping...")
            chunk_assigner = ChunkAssigner(document.id)
            
            # Chunks already have section_id from extract_nodes_from_chunks,
            # just use them directly
            chunks_with_sections = chunks
            
            # Log chunk statistics
            section_stats = chunk_assigner.get_section_statistics(
                chunks_with_sections, 
                sections_with_parents
            )
            
            # Step 5: Create Chunk nodes
            logger.info("Step 5: Creating Chunk nodes...")
            self._create_chunk_nodes(chunks_with_sections)
            stats["chunks_created"] = len(chunks_with_sections)
            
            # Step 6: Create NEXT chain
            logger.info("Step 6: Creating chunk NEXT chain...")
            chunk_chain = chunk_assigner.create_chunk_chain(chunks_with_sections)
            self._create_chunk_chain(chunk_chain)
            stats["relationships_created"] = len(chunk_chain)
            
            logger.info(f"✓ Graph building complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise
    
    def _create_document_node(self, document: DocumentNode):
        query = """
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.source = $source,
            d.language = $language,
            d.llamaparse_used = $llamaparse_used,
            d.created_at = datetime($created_at)
        RETURN d
        """
        
        self.db.execute_write(query, {
            "id": document.id,
            "title": document.title,
            "source": document.source,
            "language": document.language,
            "llamaparse_used": document.llamaparse_used,
            "created_at": document.created_at.isoformat() if document.created_at else None
        })
        
        logger.info(f"✓ Created Document: {document.title}")
    
    def _create_section_nodes_with_hierarchy(
        self, 
        document_id: str, 
        sections: List[SectionNode]
    ):
        
        for section in sections:
            # Create section node
            self._create_section_node(section)
            
            # Create relationship to parent
            if section.parent_id is None:
                # H1: Link to Document
                self._link_section_to_document(document_id, section.id)
                logger.debug(f"Linked H1 section '{section.title}' to Document")
            else:
                # Hn: Link to parent section
                self._link_section_to_parent(section.parent_id, section.id)
                logger.debug(
                    f"Created HAS_CHILD: parent={section.parent_id} -> child={section.id} "
                    f"('{section.title}', level {section.level})"
                )
    
    def _create_section_node(self, section: SectionNode):
        query = """
        MERGE (s:Section {id: $id})
        SET s.document_id = $document_id,
            s.title = $title,
            s.level = $level,
            s.order = $order,
            s.number = $number,
            s.path = $path,
            s.parent_id = $parent_id
        RETURN s
        """
        
        self.db.execute_write(query, {
            "id": section.id,
            "document_id": section.document_id,
            "title": section.title,
            "level": section.level,
            "order": section.order,
            "number": section.number,
            "path": section.path,
            "parent_id": section.parent_id
        })
    
    def _link_section_to_document(self, document_id: str, section_id: str):
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (s:Section {id: $section_id})
        MERGE (d)-[:HAS_SECTION]->(s)
        """
        
        self.db.execute_write(query, {
            "document_id": document_id,
            "section_id": section_id
        })
    
    def _link_section_to_parent(self, parent_id: str, child_id: str):
        query = """
        MATCH (parent:Section {id: $parent_id})
        MATCH (child:Section {id: $child_id})
        MERGE (parent)-[:HAS_CHILD]->(child)
        """
        
        self.db.execute_write(query, {
            "parent_id": parent_id,
            "child_id": child_id
        })
    
    def _create_chunk_nodes(self, chunks: List[ChunkNode]):
        for chunk in chunks:
            self._create_chunk_node(chunk)
            # Only link if chunk has a valid section_id (not "default" or empty)
            if chunk.section_id and chunk.section_id != "default":
                self._link_chunk_to_section(chunk.section_id, chunk.id)
            else:
                logger.warning(
                    f"Chunk {chunk.id} has no valid section_id: '{chunk.section_id}', "
                    f"skipping HAS_CHUNK relationship"
                )
    
    def _create_chunk_node(self, chunk: ChunkNode):
        query = """
        MERGE (c:Chunk {id: $id})
        SET c.document_id = $document_id,
            c.section_id = $section_id,
            c.text = $text,
            c.token_length = $token_length,
            c.order = $order,
            c.chunk_type = $chunk_type,
            c.features_json = $features_json,
            c.embedding_id = $embedding_id
        RETURN c
        """
        
        features_json = json.dumps(chunk.features, ensure_ascii=False)
        
        self.db.execute_write(query, {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "section_id": chunk.section_id,
            "text": chunk.text,
            "token_length": chunk.token_length,
            "order": chunk.order,
            "chunk_type": chunk.chunk_type,
            "features_json": features_json,
            "embedding_id": chunk.embedding_id
        })
    
    def _link_chunk_to_section(self, section_id: str, chunk_id: str):
        query = """
        MATCH (s:Section {id: $section_id})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (s)-[:HAS_CHUNK]->(c)
        """
        
        self.db.execute_write(query, {
            "section_id": section_id,
            "chunk_id": chunk_id
        })
    
    def _create_chunk_chain(self, chain: List[tuple]):
        for from_id, to_id in chain:
            query = """
            MATCH (from:Chunk {id: $from_id})
            MATCH (to:Chunk {id: $to_id})
            MERGE (from)-[:NEXT]->(to)
            """
            
            self.db.execute_write(query, {
                "from_id": from_id,
                "to_id": to_id
            })
    
    def delete_document_graph(self, document_id: str):
        logger.warning(f"Deleting graph for document: {document_id}")
        
        query = """
        MATCH (d:Document {id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:HAS_CHILD*]->(child:Section)
        OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (child)-[:HAS_CHUNK]->(cc:Chunk)
        DETACH DELETE d, s, child, c, cc
        """
        
        self.db.execute_write(query, {"document_id": document_id})
        logger.info(f"✓ Deleted graph for document: {document_id}")
    
    def initialize_schema(self):
        self.schema.initialize_schema()
