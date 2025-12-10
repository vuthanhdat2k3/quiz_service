from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

from app.database.neo4j_db import get_neo4j_db
from app.database.faiss_index import get_faiss_index
from app.core.config import get_settings

router = APIRouter(prefix="/admin", tags=["Admin - Database Management"])
settings = get_settings()


# ============== Pydantic Models ==============

class StatsResponse(BaseModel):
    neo4j: dict
    faiss: dict


class DocumentInfo(BaseModel):
    id: str
    title: str
    source: str
    language: str
    created_at: Optional[str] = None
    sections_count: int = 0
    chunks_count: int = 0
    questions_count: int = 0


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class ChunkInfo(BaseModel):
    id: str
    text: str
    section_id: str
    section_header: str
    token_count: int
    chunk_type: str


class QuestionInfo(BaseModel):
    id: str
    question: str
    question_type: str
    choices: List[str]
    answer: str
    difficulty: str
    chunk_id: str
    created_at: Optional[str] = None


class DeleteResponse(BaseModel):
    success: bool
    message: str
    deleted_count: int = 0


class ClearResponse(BaseModel):
    success: bool
    message: str
    neo4j_cleared: bool = False
    faiss_cleared: bool = False


# ============== Stats Endpoints ==============

@router.get("/stats", response_model=StatsResponse)
async def get_database_stats():
    try:
        neo4j_db = get_neo4j_db()
        faiss_index = get_faiss_index(settings.EMBEDDING_DIMENSION)
        
        # Neo4j stats
        neo4j_stats_query = """
        MATCH (d:Document) WITH count(d) as documents
        MATCH (s:Section) WITH documents, count(s) as sections
        MATCH (c:Chunk) WITH documents, sections, count(c) as chunks
        MATCH (q:Question) WITH documents, sections, chunks, count(q) as questions
        MATCH (concept:Concept) WITH documents, sections, chunks, questions, count(concept) as concepts
        RETURN documents, sections, chunks, questions, concepts
        """
        neo4j_result = neo4j_db.execute_query(neo4j_stats_query)
        
        neo4j_stats = {
            "documents": neo4j_result[0]["documents"] if neo4j_result else 0,
            "sections": neo4j_result[0]["sections"] if neo4j_result else 0,
            "chunks": neo4j_result[0]["chunks"] if neo4j_result else 0,
            "questions": neo4j_result[0]["questions"] if neo4j_result else 0,
            "concepts": neo4j_result[0]["concepts"] if neo4j_result else 0,
        }
        
        # FAISS stats
        faiss_stats = faiss_index.get_stats()
        
        return StatsResponse(neo4j=neo4j_stats, faiss=faiss_stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Document Management ==============

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    language: Optional[str] = None
):
    try:
        neo4j_db = get_neo4j_db()
        
        # Build query with optional language filter
        where_clause = "WHERE d.language = $language" if language else ""
        
        query = f"""
        MATCH (d:Document)
        {where_clause}
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)<-[:BASED_ON]-(q:Question)
        WITH d, count(DISTINCT s) as sections_count, count(DISTINCT c) as chunks_count, count(DISTINCT q) as questions_count
        RETURN d.id as id, d.title as title, d.source as source, d.language as language, 
               toString(d.created_at) as created_at, sections_count, chunks_count, questions_count
        ORDER BY d.created_at DESC
        SKIP $skip LIMIT $limit
        """
        
        params = {"skip": skip, "limit": limit}
        if language:
            params["language"] = language
            
        results = neo4j_db.execute_query(query, params)
        
        # Get total count
        count_query = f"MATCH (d:Document) {where_clause} RETURN count(d) as total"
        count_params = {"language": language} if language else {}
        total_result = neo4j_db.execute_query(count_query, count_params)
        total = total_result[0]["total"] if total_result else 0
        
        documents = [
            DocumentInfo(
                id=r["id"],
                title=r["title"] or "Untitled",
                source=r["source"] or "",
                language=r["language"] or "vi",
                created_at=r["created_at"],
                sections_count=r["sections_count"],
                chunks_count=r["chunks_count"],
                questions_count=r["questions_count"]
            )
            for r in results
        ]
        
        return DocumentListResponse(documents=documents, total=total)
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document_detail(doc_id: str):
    try:
        neo4j_db = get_neo4j_db()
        
        # Get document info
        doc_query = """
        MATCH (d:Document {id: $doc_id})
        RETURN d.id as id, d.title as title, d.source as source, d.language as language,
               toString(d.created_at) as created_at, d.llamaparse_used as llamaparse_used
        """
        doc_result = neo4j_db.execute_query(doc_query, {"doc_id": doc_id})
        
        if not doc_result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = doc_result[0]
        
        # Get sections
        sections_query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
        WITH s, count(c) as chunks_count
        RETURN s.id as id, s.header as header, s.level as level, s.path as path, chunks_count
        ORDER BY s.level, s.id
        """
        sections = neo4j_db.execute_query(sections_query, {"doc_id": doc_id})
        
        # Get questions count
        questions_query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)<-[:BASED_ON]-(q:Question)
        RETURN count(q) as count
        """
        questions_result = neo4j_db.execute_query(questions_query, {"doc_id": doc_id})
        questions_count = questions_result[0]["count"] if questions_result else 0
        
        return {
            "document": doc,
            "sections": sections,
            "questions_count": questions_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str, delete_faiss: bool = True):
    try:
        neo4j_db = get_neo4j_db()
        faiss_index = get_faiss_index(settings.EMBEDDING_DIMENSION)
        
        # Get chunk IDs for FAISS cleanup
        chunks_query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.id as chunk_id
        """
        chunk_results = neo4j_db.execute_query(chunks_query, {"doc_id": doc_id})
        chunk_ids = [r["chunk_id"] for r in chunk_results]
        
        # Delete from Neo4j (cascade delete)
        delete_query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)<-[:BASED_ON]-(q:Question)
        OPTIONAL MATCH (c)-[:MENTIONS]->(concept:Concept)
        DETACH DELETE d, s, c, q
        RETURN count(*) as deleted
        """
        result = neo4j_db.execute_write(delete_query, {"doc_id": doc_id})
        
        # Delete orphan concepts
        cleanup_concepts_query = """
        MATCH (concept:Concept)
        WHERE NOT (concept)<-[:MENTIONS]-()
        DELETE concept
        """
        neo4j_db.execute_write(cleanup_concepts_query)
        
        # Delete from FAISS
        faiss_deleted = 0
        if delete_faiss and chunk_ids:
            # Note: FAISS doesn't support deletion by ID easily
            # We need to rebuild the index or mark entries as deleted
            for chunk_id in chunk_ids:
                if chunk_id in faiss_index.metadata:
                    del faiss_index.metadata[chunk_id]
                    faiss_deleted += 1
            
            # Remove from id_map
            faiss_index.id_map = {
                idx: cid for idx, cid in faiss_index.id_map.items() 
                if cid not in chunk_ids
            }
            faiss_index.save()
        
        deleted_count = len(chunk_ids)
        return DeleteResponse(
            success=True,
            message=f"Document {doc_id} deleted. Removed {deleted_count} chunks from Neo4j, {faiss_deleted} from FAISS metadata.",
            deleted_count=deleted_count
        )
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Chunk Management ==============

@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200)
):
    try:
        neo4j_db = get_neo4j_db()
        
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.id as id, c.text as text, s.id as section_id, s.header as section_header,
               c.token_count as token_count, c.chunk_type as chunk_type
        ORDER BY s.level, c.id
        SKIP $skip LIMIT $limit
        """
        results = neo4j_db.execute_query(query, {"doc_id": doc_id, "skip": skip, "limit": limit})
        
        # Get total count
        count_query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
        RETURN count(c) as total
        """
        count_result = neo4j_db.execute_query(count_query, {"doc_id": doc_id})
        total = count_result[0]["total"] if count_result else 0
        
        chunks = [
            ChunkInfo(
                id=r["id"],
                text=r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"],
                section_id=r["section_id"],
                section_header=r["section_header"] or "Root",
                token_count=r["token_count"] or 0,
                chunk_type=r["chunk_type"] or "text"
            )
            for r in results
        ]
        
        return {"chunks": chunks, "total": total}
    except Exception as e:
        logger.error(f"Error getting chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chunks/{chunk_id}", response_model=DeleteResponse)
async def delete_chunk(chunk_id: str):
    try:
        neo4j_db = get_neo4j_db()
        faiss_index = get_faiss_index(settings.EMBEDDING_DIMENSION)
        
        # Delete chunk and its questions from Neo4j
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        OPTIONAL MATCH (c)<-[:BASED_ON]-(q:Question)
        DETACH DELETE c, q
        RETURN count(*) as deleted
        """
        result = neo4j_db.execute_write(query, {"chunk_id": chunk_id})
        
        # Remove from FAISS metadata
        if chunk_id in faiss_index.metadata:
            del faiss_index.metadata[chunk_id]
            faiss_index.save()
        
        return DeleteResponse(
            success=True,
            message=f"Chunk {chunk_id} deleted",
            deleted_count=1
        )
    except Exception as e:
        logger.error(f"Error deleting chunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Question Management ==============

@router.get("/documents/{doc_id}/questions")
async def get_document_questions(
    doc_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200)
):
    try:
        neo4j_db = get_neo4j_db()
        
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)<-[:BASED_ON]-(q:Question)
        RETURN q.id as id, q.question as question, q.type as question_type, 
               q.choices as choices, q.answer as answer, q.difficulty as difficulty,
               c.id as chunk_id, toString(q.created_at) as created_at
        ORDER BY q.created_at DESC
        SKIP $skip LIMIT $limit
        """
        results = neo4j_db.execute_query(query, {"doc_id": doc_id, "skip": skip, "limit": limit})
        
        # Get total count
        count_query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)<-[:BASED_ON]-(q:Question)
        RETURN count(q) as total
        """
        count_result = neo4j_db.execute_query(count_query, {"doc_id": doc_id})
        total = count_result[0]["total"] if count_result else 0
        
        questions = [
            QuestionInfo(
                id=r["id"],
                question=r["question"],
                question_type=r["question_type"],
                choices=r["choices"] or [],
                answer=r["answer"],
                difficulty=r["difficulty"],
                chunk_id=r["chunk_id"],
                created_at=r["created_at"]
            )
            for r in results
        ]
        
        return {"questions": questions, "total": total}
    except Exception as e:
        logger.error(f"Error getting questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/questions/{question_id}", response_model=DeleteResponse)
async def delete_question(question_id: str):
    try:
        neo4j_db = get_neo4j_db()
        
        query = """
        MATCH (q:Question {id: $question_id})
        DETACH DELETE q
        RETURN count(*) as deleted
        """
        result = neo4j_db.execute_write(query, {"question_id": question_id})
        
        return DeleteResponse(
            success=True,
            message=f"Question {question_id} deleted",
            deleted_count=1
        )
    except Exception as e:
        logger.error(f"Error deleting question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== FAISS Management ==============

@router.get("/faiss/stats")
async def get_faiss_stats():
    try:
        faiss_index = get_faiss_index(settings.EMBEDDING_DIMENSION)
        stats = faiss_index.get_stats()
        
        # Add more details
        stats["dimension"] = faiss_index.dimension
        stats["index_path"] = faiss_index.index_path
        
        # Get documents in FAISS
        doc_ids = set()
        for chunk_id, meta in faiss_index.metadata.items():
            if "document_id" in meta:
                doc_ids.add(meta["document_id"])
        stats["documents_count"] = len(doc_ids)
        
        return stats
    except Exception as e:
        logger.error(f"Error getting FAISS stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/faiss/rebuild")
async def rebuild_faiss_index():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        neo4j_db = get_neo4j_db()
        faiss_index = get_faiss_index(settings.EMBEDDING_DIMENSION)
        
        # Clear current index
        faiss_index.clear()
        
        # Get all chunks from Neo4j
        query = """
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.id as chunk_id, c.text as text, d.id as document_id, s.id as section_id
        """
        chunks = neo4j_db.execute_query(query)
        
        if not chunks:
            return {"success": True, "message": "No chunks to index", "indexed": 0}
        
        # Load embedding model
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Generate embeddings
        texts = [c["text"] for c in chunks]
        embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Add to FAISS
        chunk_ids = [c["chunk_id"] for c in chunks]
        metadata_list = [
            {
                "document_id": c["document_id"],
                "section_id": c["section_id"],
                "text": c["text"][:500]
            }
            for c in chunks
        ]
        
        faiss_index.add_embeddings_batch(chunk_ids, embeddings, metadata_list)
        faiss_index.save()
        
        return {
            "success": True,
            "message": "FAISS index rebuilt successfully",
            "indexed": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error rebuilding FAISS index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/faiss/clear", response_model=ClearResponse)
async def clear_faiss_index():
    try:
        faiss_index = get_faiss_index(settings.EMBEDDING_DIMENSION)
        faiss_index.clear()
        faiss_index.save()
        
        return ClearResponse(
            success=True,
            message="FAISS index cleared",
            faiss_cleared=True
        )
    except Exception as e:
        logger.error(f"Error clearing FAISS index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Clear All Data ==============

@router.delete("/clear-all", response_model=ClearResponse)
async def clear_all_data(confirm: str = Query(..., description="Type 'CONFIRM' to proceed")):
    if confirm != "CONFIRM":
        raise HTTPException(status_code=400, detail="Please provide confirm='CONFIRM' to proceed")
    
    try:
        neo4j_db = get_neo4j_db()
        faiss_index = get_faiss_index(settings.EMBEDDING_DIMENSION)
        
        # Clear Neo4j
        neo4j_db.execute_write("MATCH (n) DETACH DELETE n")
        
        # Clear FAISS
        faiss_index.clear()
        faiss_index.save()
        
        return ClearResponse(
            success=True,
            message="All data cleared from Neo4j and FAISS",
            neo4j_cleared=True,
            faiss_cleared=True
        )
    except Exception as e:
        logger.error(f"Error clearing all data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Search & Query ==============

@router.get("/search/chunks")
async def search_chunks(
    query: str = Query(..., min_length=1),
    document_id: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50)
):
    try:
        from sentence_transformers import SentenceTransformer
        
        faiss_index = get_faiss_index(settings.EMBEDDING_DIMENSION)
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Search FAISS
        results = faiss_index.search(query_embedding, k=limit * 2, deduplicate=True)
        
        # Filter by document if specified
        if document_id:
            results = [
                (chunk_id, score, meta) 
                for chunk_id, score, meta in results 
                if meta.get("document_id") == document_id
            ]
        
        # Limit results
        results = results[:limit]
        
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "text": meta.get("text", "")[:300] + "...",
                    "document_id": meta.get("document_id", ""),
                    "section_id": meta.get("section_id", "")
                }
                for chunk_id, score, meta in results
            ]
        }
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neo4j/query")
async def execute_neo4j_query(query: str, read_only: bool = True):
    try:
        neo4j_db = get_neo4j_db()
        
        # Security check - block dangerous operations if read_only
        if read_only:
            dangerous_keywords = ["DELETE", "REMOVE", "SET", "CREATE", "MERGE", "DROP"]
            query_upper = query.upper()
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Query contains '{keyword}' but read_only=True. Set read_only=False to execute write operations."
                    )
        
        if read_only:
            results = neo4j_db.execute_query(query)
        else:
            results = neo4j_db.execute_write(query)
        
        return {"results": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing Neo4j query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
