import sys
from loguru import logger

from app.database.neo4j_db import get_neo4j_db, close_neo4j_db


def create_constraints(db):
    
    logger.info("Creating uniqueness constraints...")

    constraints = [
        "CREATE CONSTRAINT unique_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT unique_section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT unique_concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT unique_question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
    ]

    for constraint in constraints:
        try:
            db.execute_write(constraint, {})
            logger.info(f"✓ Created: {constraint.split('unique_')[1].split(' ')[0]}")
        except Exception as e:
            logger.warning(f"Constraint may already exist: {e}")


def create_indexes(db):
    
    logger.info("Creating indexes...")

    indexes = [
        # Text search indexes
        "CREATE INDEX chunk_text_index IF NOT EXISTS FOR (c:Chunk) ON (c.text)",
        "CREATE INDEX concept_text_index IF NOT EXISTS FOR (c:Concept) ON (c.text)",
        "CREATE INDEX question_text_index IF NOT EXISTS FOR (q:Question) ON (q.question)",
        
        # Lookup indexes
        "CREATE INDEX document_source_index IF NOT EXISTS FOR (d:Document) ON (d.source)",
        "CREATE INDEX question_status_index IF NOT EXISTS FOR (q:Question) ON (q.status)",
        "CREATE INDEX question_difficulty_index IF NOT EXISTS FOR (q:Question) ON (q.difficulty)",
        
        # Embedding indexes
        "CREATE INDEX chunk_embedding_index IF NOT EXISTS FOR (c:Chunk) ON (c.embedding_id)",
        "CREATE INDEX concept_embedding_index IF NOT EXISTS FOR (c:Concept) ON (c.embedding_id)",
    ]

    for index in indexes:
        try:
            db.execute_write(index, {})
            logger.info(f"✓ Created: {index.split('_index')[0].split()[-1]}")
        except Exception as e:
            logger.warning(f"Index may already exist: {e}")


def create_fulltext_indexes(db):
    
    logger.info("Creating fulltext indexes...")

    # Check if fulltext index exists
    try:
        result = db.execute_query("SHOW INDEXES")
        existing_indexes = [r.get("name", "") for r in result]
        
        if "chunk_fulltext" not in existing_indexes:
            db.execute_write(
                "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
                {},
            )
            logger.info("✓ Created fulltext index on Chunk.text")
        else:
            logger.info("✓ Fulltext index already exists")
    except Exception as e:
        logger.warning(f"Could not create fulltext index: {e}")


def create_vector_index(db):
    
    logger.info("Checking vector index support...")

    try:
        # Try to create vector index (Neo4j 5.11+)
        # Note: This requires Neo4j Enterprise or specific configuration
        query = """
        CREATE VECTOR INDEX chunk_embedding_vector IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding_vector)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
        }}
        """
        db.execute_write(query, {})
        logger.info("✓ Created vector index (Neo4j 5.11+ detected)")
    except Exception as e:
        logger.warning(f"Vector index not supported or already exists: {e}")
        logger.info("→ Will use FAISS for vector search instead")


def verify_schema(db):
    
    logger.info("Verifying schema...")

    # Check constraints
    result = db.execute_query("SHOW CONSTRAINTS")
    constraint_count = len(result)
    logger.info(f"✓ Found {constraint_count} constraints")

    # Check indexes
    result = db.execute_query("SHOW INDEXES")
    index_count = len(result)
    logger.info(f"✓ Found {index_count} indexes")

    return constraint_count > 0 and index_count > 0


def main():
    
    logger.info("=" * 60)
    logger.info("Neo4j Database Migration")
    logger.info("=" * 60)

    try:
        # Connect to database
        db = get_neo4j_db()
        logger.info("✓ Connected to Neo4j")

        # Run migrations
        create_constraints(db)
        create_indexes(db)
        create_fulltext_indexes(db)
        create_vector_index(db)

        # Verify
        if verify_schema(db):
            logger.info("=" * 60)
            logger.success("✅ Migration completed successfully!")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("❌ Schema verification failed")
            return 1

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1

    finally:
        close_neo4j_db()


if __name__ == "__main__":
    sys.exit(main())
