from typing import List, Dict, Any, Optional
from loguru import logger
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable

from app.core.config import get_settings

settings = get_settings()


class Neo4jDatabase:

    def __init__(self):
        self.driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def execute_write(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        parameters = parameters or {}

        def _write_tx(tx):
            result = tx.run(query, parameters)
            return [dict(record) for record in result]

        try:
            with self.driver.session() as session:
                return session.execute_write(_write_tx)
        except Exception as e:
            logger.error(f"Error executing write transaction: {e}")
            raise

    # Document operations
    def create_document(
        self,
        doc_id: str,
        title: str,
        source: str,
        language: str = "vi",
        llamaparse_used: bool = False,
    ) -> Dict[str, Any]:
        query = """
        CREATE (d:Document {
            id: $doc_id,
            title: $title,
            source: $source,
            language: $language,
            llamaparse_used: $llamaparse_used,
            created_at: datetime()
        })
        RETURN d
        """
        result = self.execute_write(
            query,
            {
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "language": language,
                "llamaparse_used": llamaparse_used,
            },
        )
        return result[0]["d"] if result else None

    def create_section(
        self, section_id: str, doc_id: str, header: str, level: int, path: List[str]
    ) -> Dict[str, Any]:
        query = """
        MATCH (d:Document {id: $doc_id})
        CREATE (s:Section {
            id: $section_id,
            header: $header,
            level: $level,
            path: $path
        })
        CREATE (d)-[:HAS_SECTION]->(s)
        RETURN s
        """
        result = self.execute_write(
            query,
            {
                "doc_id": doc_id,
                "section_id": section_id,
                "header": header,
                "level": level,
                "path": path,
            },
        )
        return result[0]["s"] if result else None

    def create_chunk(
        self,
        chunk_id: str,
        section_id: str,
        text: str,
        token_count: int,
        chunk_type: str,
        features_json: str,
        embedding_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = """
        MATCH (s:Section {id: $section_id})
        CREATE (c:Chunk {
            id: $chunk_id,
            text: $text,
            token_count: $token_count,
            chunk_type: $chunk_type,
            embedding_id: $embedding_id,
            features_json: $features_json
        })
        CREATE (s)-[:HAS_CHUNK]->(c)
        RETURN c
        """
        result = self.execute_write(
            query,
            {
                "section_id": section_id,
                "chunk_id": chunk_id,
                "text": text,
                "token_count": token_count,
                "chunk_type": chunk_type,
                "embedding_id": embedding_id,
                "features_json": features_json,
            },
        )
        return result[0]["c"] if result else None

    def create_concept(
        self, concept_id: str, text: str, canonical_form: str, embedding_id: Optional[str] = None
    ) -> Dict[str, Any]:
        query = """
        MERGE (c:Concept {id: $concept_id})
        ON CREATE SET 
            c.text = $text,
            c.canonical_form = $canonical_form,
            c.embedding_id = $embedding_id,
            c.created_at = datetime()
        RETURN c
        """
        result = self.execute_write(
            query,
            {
                "concept_id": concept_id,
                "text": text,
                "canonical_form": canonical_form,
                "embedding_id": embedding_id,
            },
        )
        return result[0]["c"] if result else None

    def link_chunk_to_concept(self, chunk_id: str, concept_id: str):
        query = """
        MATCH (chunk:Chunk {id: $chunk_id})
        MATCH (concept:Concept {id: $concept_id})
        MERGE (chunk)-[:MENTIONS]->(concept)
        """
        self.execute_write(query, {"chunk_id": chunk_id, "concept_id": concept_id})

    def create_question(
        self,
        question_id: str,
        chunk_id: str,
        question: str,
        question_type: str,
        choices: List[str],
        answer: str,
        explanation: str,
        difficulty: str,
        confidence: float,
        metadata_json: Optional[str],
        status: str = "generated",
    ) -> Dict[str, Any]:
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        CREATE (q:Question {
            id: $question_id,
            question: $question,
            type: $question_type,
            choices: $choices,
            answer: $answer,
            explanation: $explanation,
            difficulty: $difficulty,
            confidence: $confidence,
            status: $status,
            metadata_json: $metadata_json,
            created_at: datetime()
        })
        CREATE (q)-[:BASED_ON]->(c)
        RETURN q
        """
        result = self.execute_write(
            query,
            {
                "chunk_id": chunk_id,
                "question_id": question_id,
                "question": question,
                "question_type": question_type,
                "choices": choices,
                "answer": answer,
                "explanation": explanation,
                "difficulty": difficulty,
                "confidence": confidence,
                "status": status,
                "metadata_json": metadata_json,
            },
        )
        return result[0]["q"] if result else None

    def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
        RETURN c, s
        ORDER BY s.level, c.id
        """
        return self.execute_query(query, {"doc_id": doc_id})

    def get_related_concepts(self, chunk_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = """
        MATCH (chunk:Chunk {id: $chunk_id})-[:MENTIONS]->(c:Concept)
        OPTIONAL MATCH (c)-[:RELATED_TO]->(related:Concept)
        RETURN c, collect(distinct related) as related_concepts
        LIMIT $limit
        """
        return self.execute_query(query, {"chunk_id": chunk_id, "limit": limit})

    def get_neighbor_chunks(
        self, chunk_id: str, same_section: bool = True, limit: int = 5
    ) -> List[Dict[str, Any]]:
        if same_section:
            query = """
            MATCH (target:Chunk {id: $chunk_id})<-[:HAS_CHUNK]-(s:Section)-[:HAS_CHUNK]->(neighbor:Chunk)
            WHERE target.id <> neighbor.id
            RETURN neighbor
            LIMIT $limit
            """
        else:
            query = """
            MATCH (target:Chunk {id: $chunk_id})<-[:HAS_CHUNK]-(s:Section)
            MATCH (s)<-[:HAS_SECTION]-(d:Document)-[:HAS_SECTION]->(other_s:Section)-[:HAS_CHUNK]->(neighbor:Chunk)
            WHERE target.id <> neighbor.id
            RETURN neighbor
            LIMIT $limit
            """
        return self.execute_query(query, {"chunk_id": chunk_id, "limit": limit})

    def get_section_chunks(self, section_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = """
        MATCH (s:Section {id: $section_id})-[:HAS_CHUNK]->(c:Chunk)
        RETURN c
        """ + (f"LIMIT {limit}" if limit else "")
        return self.execute_query(query, {"section_id": section_id})

    def get_chunks_by_document_id(self, doc_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
        RETURN c, s
        ORDER BY s.level, c.id
        """
        return self.execute_query(query, {"doc_id": doc_id})

    def get_document_sections(self, doc_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
        RETURN s
        ORDER BY s.level
        """
        return self.execute_query(query, {"doc_id": doc_id})

    def expand_chunk_context(
        self, chunk_id: str, window: int = 2
    ) -> List[Dict[str, Any]]:
        query = """
        MATCH (target:Chunk {id: $chunk_id})<-[:HAS_CHUNK]-(s:Section)-[:HAS_CHUNK]->(c:Chunk)
        WITH target, c, s
        ORDER BY c.id
        WITH target, collect(c) as all_chunks
        WITH target, all_chunks, 
             [i IN range(0, size(all_chunks)-1) WHERE all_chunks[i].id = target.id][0] as target_idx
        WITH all_chunks, target_idx, $window as w
        WHERE target_idx IS NOT NULL
        RETURN [c IN all_chunks WHERE 
                apoc.coll.indexOf(all_chunks, c) >= target_idx - w AND 
                apoc.coll.indexOf(all_chunks, c) <= target_idx + w] as context_chunks
        """
        result = self.execute_query(query, {"chunk_id": chunk_id, "window": window})
        if result and "context_chunks" in result[0]:
            return result[0]["context_chunks"]
        return []


# Singleton instance
_neo4j_db: Optional[Neo4jDatabase] = None


def get_neo4j_db() -> Neo4jDatabase:
    global _neo4j_db
    if _neo4j_db is None:
        _neo4j_db = Neo4jDatabase()
    return _neo4j_db


def close_neo4j_db():
    global _neo4j_db
    if _neo4j_db:
        _neo4j_db.close()
        _neo4j_db = None
