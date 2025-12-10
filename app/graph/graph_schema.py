from loguru import logger
from typing import List

from app.database.neo4j_db import Neo4jDatabase


class GraphSchema:
    def __init__(self, db: Neo4jDatabase):
        self.db = db
    
    def create_constraints(self):
        constraints = [
            # Document constraints
            """
            CREATE CONSTRAINT document_id_unique IF NOT EXISTS
            FOR (d:Document) REQUIRE d.id IS UNIQUE
            """,
            
            # Section constraints
            """
            CREATE CONSTRAINT section_id_unique IF NOT EXISTS
            FOR (s:Section) REQUIRE s.id IS UNIQUE
            """,
            
            # Chunk constraints
            """
            CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """,
            
            # Concept constraints
            """
            CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
            FOR (c:Concept) REQUIRE c.id IS UNIQUE
            """,
            
            # Question constraints
            """
            CREATE CONSTRAINT question_id_unique IF NOT EXISTS
            FOR (q:Question) REQUIRE q.id IS UNIQUE
            """
        ]
        
        logger.info("Creating graph constraints...")
        for constraint_query in constraints:
            try:
                self.db.execute_write(constraint_query)
                logger.info(f"✓ Created constraint")
            except Exception as e:
                # Constraint might already exist
                logger.debug(f"Constraint creation: {str(e)}")
        
        logger.info("✓ All constraints created/verified")
    
    def create_indexes(self):
        indexes = [
            # Document indexes
            """
            CREATE INDEX document_title IF NOT EXISTS
            FOR (d:Document) ON (d.title)
            """,
            
            # Section indexes
            """
            CREATE INDEX section_document_id IF NOT EXISTS
            FOR (s:Section) ON (s.document_id)
            """,
            """
            CREATE INDEX section_level IF NOT EXISTS
            FOR (s:Section) ON (s.level)
            """,
            """
            CREATE INDEX section_order IF NOT EXISTS
            FOR (s:Section) ON (s.order)
            """,
            
            # Chunk indexes
            """
            CREATE INDEX chunk_document_id IF NOT EXISTS
            FOR (c:Chunk) ON (c.document_id)
            """,
            """
            CREATE INDEX chunk_section_id IF NOT EXISTS
            FOR (c:Chunk) ON (c.section_id)
            """,
            """
            CREATE INDEX chunk_order IF NOT EXISTS
            FOR (c:Chunk) ON (c.order)
            """,
            
            # Question indexes
            """
            CREATE INDEX question_status IF NOT EXISTS
            FOR (q:Question) ON (q.status)
            """,
            """
            CREATE INDEX question_difficulty IF NOT EXISTS
            FOR (q:Question) ON (q.difficulty)
            """
        ]
        
        logger.info("Creating graph indexes...")
        for index_query in indexes:
            try:
                self.db.execute_write(index_query)
                logger.info(f"✓ Created index")
            except Exception as e:
                logger.debug(f"Index creation: {str(e)}")
        
        logger.info("✓ All indexes created/verified")
    
    def initialize_schema(self):
        logger.info("Initializing Neo4j graph schema...")
        self.create_constraints()
        self.create_indexes()
        logger.info("✓ Graph schema initialization complete")
    def drop_all_constraints(self):
        logger.warning("Dropping all constraints...")
        
        # Get all constraints
        query = "SHOW CONSTRAINTS"
        result = self.db.execute_query(query)
        
        for record in result:
            constraint_name = record.get('name')
            if constraint_name:
                drop_query = f"DROP CONSTRAINT {constraint_name} IF EXISTS"
                try:
                    self.db.execute_write(drop_query)
                    logger.info(f"✓ Dropped constraint: {constraint_name}")
                except Exception as e:
                    logger.error(f"Failed to drop constraint {constraint_name}: {e}")
    
    def drop_all_indexes(self):
        logger.warning("Dropping all indexes...")
        
        # Get all indexes
        query = "SHOW INDEXES"
        result = self.db.execute_query(query)
        
        for record in result:
            index_name = record.get('name')
            if index_name:
                drop_query = f"DROP INDEX {index_name} IF EXISTS"
                try:
                    self.db.execute_write(drop_query)
                    logger.info(f"✓ Dropped index: {index_name}")
                except Exception as e:
                    logger.error(f"Failed to drop index {index_name}: {e}")
