import os
import tempfile
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
import numpy as np
import aiofiles

# Pre-import SentenceTransformer for faster loading
try:
    from sentence_transformers import SentenceTransformer
    import torch
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None
    torch = None

from app.database.neo4j_db import Neo4jDatabase
from app.database.faiss_index import FAISSIndex
from app.models.quiz import QuizQuestion, QuizOption
from app.parsers import ParserFactory
from app.chunkers.markdown_chunker import MarkdownChunkerV2
from app.llm import get_llm_adapter
from app.core.config import get_settings
from app.graph import DocumentGraphBuilder
from app.services.chunk_selector import ChunkSelector, ChunkSelectionConfig


# Global embedding model cache (singleton)
_embedding_model_cache = {}
_embedding_model_lock = asyncio.Lock() if asyncio else None


def _get_device() -> str:
    """Get the best available device for embeddings."""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def _load_embedding_model(model_name: str, device: str = None):
    """Load and cache embedding model (thread-safe singleton)."""
    global _embedding_model_cache
    
    if model_name in _embedding_model_cache:
        return _embedding_model_cache[model_name]
    
    if not _SENTENCE_TRANSFORMER_AVAILABLE:
        raise ImportError("sentence-transformers package is required")
    
    device = device or _get_device()
    logger.info(f"🚀 Loading embedding model '{model_name}' on device '{device}'...")
    
    model = SentenceTransformer(model_name, device=device)
    
    # Optimize for inference
    if hasattr(model, 'eval'):
        model.eval()
    
    _embedding_model_cache[model_name] = model
    logger.info(f"✅ Embedding model loaded successfully on {device}")
    
    return model


class QuizGenerationService:

    # Thread pool for CPU-bound embedding operations
    _executor = ThreadPoolExecutor(max_workers=2)

    def __init__(self):
        self.settings = get_settings()
        self.neo4j_db = Neo4jDatabase()
        self.faiss_index = FAISSIndex(dimension=self.settings.EMBEDDING_DIMENSION)
        self.llm_adapter = get_llm_adapter()
        self.markdown_chunker = MarkdownChunkerV2()
        self.chunk_selector = ChunkSelector(ChunkSelectionConfig(
            max_total_tokens=3000,
            tokens_per_question=150,
            min_chunks=3,
            max_chunks=30,
            include_adjacent=True,
            semantic_weight=0.7
        ))
        
        # Pre-load embedding model (lazy initialization)
        self._embedding_model = None
        self._device = _get_device()
    
    @property
    def embedding_model(self):
        """Lazy-load and cache embedding model."""
        if self._embedding_model is None:
            self._embedding_model = _load_embedding_model(
                self.settings.EMBEDDING_MODEL,
                self._device
            )
        return self._embedding_model
    
    async def _encode_texts_async(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Encode texts asynchronously using thread pool to avoid blocking event loop."""
        loop = asyncio.get_event_loop()
        
        def _encode():
            return self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True  # Pre-normalize for cosine similarity
            )
        
        return await loop.run_in_executor(self._executor, _encode)

    async def save_uploaded_file(self, file_content: bytes, file_name: str) -> str:
        
        # Create temp directory if not exists
        temp_dir = Path(tempfile.gettempdir()) / "quiz_service_uploads"
        temp_dir.mkdir(exist_ok=True)

        # Generate unique file name
        file_ext = Path(file_name).suffix
        temp_file_path = temp_dir / f"{os.urandom(16).hex()}{file_ext}"

        # Save file
        async with aiofiles.open(temp_file_path, "wb") as f:
            await f.write(file_content)

        logger.info(f"Saved uploaded file to: {temp_file_path}")

        return str(temp_file_path)

    async def generate_from_prompt(
        self,
        prompt: str,
        num_questions: int,
        difficulty: str,
        language: str,
        question_types: List[int],
    ) -> List[QuizQuestion]:
        
        logger.info(f"Generating {num_questions} quiz questions from prompt (batch mode)")

        questions = []
        
        try:
            # Generate all questions in a single API call
            options_cfg = {
                "difficulty": difficulty,
                "language": language,
                "question_types": question_types,
            }
            
            batch_result = await self.llm_adapter.generate_batch_mcq(
                prompt, num_questions, options=options_cfg
            )
            
            if batch_result and batch_result.questions:
                from app.models.quiz import QuestionTypeEnum
                
                # Determine question type distribution
                if 2 in question_types:  # Mix mode
                    actual_types = [0, 1]  # Single and Multiple choice
                else:
                    actual_types = question_types
                
                for idx, result in enumerate(batch_result.questions[:num_questions]):
                    if not result.question or not result.choices:
                        continue
                    
                    # Alternate question types based on answer format
                    if isinstance(result.answer, list) and len(result.answer) > 1:
                        q_type = 1  # Multiple choice
                    else:
                        q_type = 0  # Single choice
                    
                    # Map answer letter(s) to index
                    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                    
                    if isinstance(result.answer, list):
                        correct_indices = [answer_map.get(a.strip().upper(), -1) for a in result.answer]
                        correct_indices = [i for i in correct_indices if i >= 0]
                    else:
                        correct_indices = [answer_map.get(str(result.answer).strip().upper(), 0)]

                    # Build options list - ID always defaults to 0
                    options = []
                    for opt_idx, opt_text in enumerate(result.choices):
                        options.append(
                            QuizOption(
                                id=0,
                                optionText=opt_text,
                                isCorrect=(opt_idx in correct_indices),
                            )
                        )

                    question = QuizQuestion(
                        id=0,
                        questionText=result.question,
                        questionType=QuestionTypeEnum(q_type),
                        point=1.0,
                        options=options,
                    )
                    questions.append(question)
                
                logger.info(f"Successfully generated {len(questions)}/{num_questions} questions in single API call")
                    
        except Exception as e:
            logger.error(f"Error generating questions from prompt: {e}")
            raise
        finally:
            # Always cleanup database and FAISS index after generation (success or failure)
            self.cleanup_data()

        return questions

    async def generate_from_file_advanced(
        self,
        file_path: str,
        num_questions: int,
        difficulty: str,
        language: str,
        question_types: List[int],
        additional_prompt: Optional[str] = None,
    ) -> tuple[List[QuizQuestion], str, str, List[str], List[Dict[str, Any]]]:
        
        logger.info(f"🚀 Starting advanced quiz generation for: {file_path}")
        document_id = str(uuid.uuid4())
        metadata = {"steps": []}

        try:
            # Step 1: Parse document to Markdown using LlamaParse
            logger.info("📄 Step 1: Parsing document to Markdown...")
            parser = ParserFactory.get_parser(file_path)
            if not parser:
                raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

            markdown_content = await parser.parse(file_path)
            
            if not markdown_content or len(markdown_content.strip()) < 50:
                raise ValueError("Insufficient content extracted from document")

            logger.info(f"✓ Parsed: {len(markdown_content)} characters")
            metadata["steps"].append({
                "step": 1,
                "name": "parse",
                "status": "success",
                "content_length": len(markdown_content)
            })

            # Step 2: Chunk by Markdown AST structure
            logger.info("🧩 Step 2: Chunking by Markdown AST (≤180 tokens)...")
            chunks = self.markdown_chunker.parse(markdown_content)
            
            logger.info(f"✓ Created {len(chunks)} chunks")
            metadata["steps"].append({
                "step": 2,
                "name": "chunk",
                "status": "success",
                "num_chunks": len(chunks),
                "avg_tokens": np.mean([c.token_count for c in chunks]) if chunks else 0
            })

            # Step 3: Store in Neo4j graph with proper hierarchy
            logger.info("💾 Step 3: Storing in Neo4j graph with hierarchy...")
            file_name = Path(file_path).name
            
            # Extract document, sections, and chunks using graph builder
            document, section_nodes, chunk_nodes = DocumentGraphBuilder.extract_nodes_from_chunks(
                chunks=chunks,
                document_id=document_id,
                file_name=file_name,
                file_path=file_path,
                language=language,
                llamaparse_used=isinstance(parser, type) and parser.__name__ == "LlamaParseParser"
            )
            
            # Get chunk IDs for later use
            chunk_ids = [chunk.id for chunk in chunk_nodes]
            
            # Build graph using DocumentGraphBuilder
            graph_builder = DocumentGraphBuilder(self.neo4j_db)
            graph_builder.initialize_schema()
            
            graph_stats = graph_builder.build_document_graph(
                document=document,
                sections=section_nodes,
                chunks=chunk_nodes
            )
            
            logger.info(f"✓ Built graph: {graph_stats}")
            metadata["steps"].append({
                "step": 3,
                "name": "neo4j_storage",
                "status": "success",
                "document_id": document_id,
                "sections_created": graph_stats.get("sections_created", 0),
                "chunks_stored": graph_stats.get("chunks_created", 0),
                "relationships_created": graph_stats.get("relationships_created", 0)
            })

            # Step 4: Compute & store embeddings (OPTIMIZED)
            logger.info(f"🔢 Step 4: Computing embeddings for {len(chunks)} chunks on {self._device}...")
            
            chunk_texts = [c.text for c in chunks]
            
            # Use optimized batch size based on device
            batch_size = 128 if self._device == "cuda" else 64
            
            # Async encoding to avoid blocking event loop
            embeddings = await self._encode_texts_async(
                chunk_texts,
                batch_size=batch_size,
                show_progress=len(chunks) > 100
            )
            
            # Pre-build metadata list (vectorized approach)
            metadata_list = [
                {
                    "document_id": document_id,
                    "chunk_index": idx,
                    "token_count": chunk.token_count,
                    "chunk_type": chunk.chunk_type,
                    "section_path": getattr(chunk, 'section_path', []),
                    "text": chunk.text[:500],  # Store more text for BM25 search
                    "heading": chunk.metadata.get("heading", "")
                }
                for idx, chunk in enumerate(chunks)
            ]
            
            self.faiss_index.add_embeddings_batch(chunk_ids, embeddings, metadata_list, already_normalized=True)
            
            logger.info(f"✓ Stored {len(embeddings)} embeddings in FAISS")
            metadata["steps"].append({
                "step": 4,
                "name": "embeddings",
                "status": "success",
                "num_embeddings": len(embeddings),
                "dimension": embeddings.shape[1]
            })

            # Step 5: Candidate selection (uses hybrid search if additional_prompt provided)
            logger.info("🎯 Step 5: Selecting candidate chunks...")
            candidate_chunks = await self._select_candidate_chunks(
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                num_questions=num_questions,
                document_id=document_id,
                additional_prompt=additional_prompt
            )
            
            logger.info(f"✓ Selected {len(candidate_chunks)} candidate chunks")
            metadata["steps"].append({
                "step": 5,
                "name": "candidate_selection",
                "status": "success",
                "num_candidates": len(candidate_chunks)
            })

            # Step 6: Generate questions with LLM (single batch call)
            logger.info("🤖 Step 6: Generating questions with LLM (batch mode)...")
            questions, question_chunk_ids = await self._generate_questions_batch(
                candidate_chunks=candidate_chunks,
                num_questions=num_questions,
                difficulty=difficulty,
                language=language,
                question_types=question_types,
            )
            
            logger.info(f"✓ Generated {len(questions)} questions in single API call")
            metadata["steps"].append({
                "step": 6,
                "name": "question_generation",
                "status": "success",
                "num_questions": len(questions)
            })

            # Step 7: Store questions in Neo4j
            logger.info("💾 Step 7: Storing questions in Neo4j...")
            for idx, question in enumerate(questions):
                question_id = f"{document_id}_q_{idx}"
                src_chunk_id = question_chunk_ids[idx] if idx < len(question_chunk_ids) else chunk_ids[0]

                # Build choices and correct answer letter
                choices = [opt.optionText for opt in question.options]
                correct_idx = next((i for i, o in enumerate(question.options) if o.isCorrect), 0)
                answer_letter = ["A", "B", "C", "D"][correct_idx] if correct_idx < 4 else "A"

                import json as _json
                self.neo4j_db.create_question(
                    question_id=question_id,
                    chunk_id=src_chunk_id,
                    question=question.questionText,
                    question_type=str(question.questionType),
                    choices=choices,
                    answer=answer_letter,
                    explanation="",
                    difficulty=difficulty,
                    confidence=0.7,
                    metadata_json=_json.dumps({"language": language}, ensure_ascii=False)
                )
            
            logger.info(f"✓ Stored {len(questions)} questions in Neo4j")
            metadata["steps"].append({
                "step": 7,
                "name": "question_storage",
                "status": "success",
                "questions_stored": len(questions)
            })

            # Return results
            chunk_texts_list = [c.text for c in chunks]
            return questions, file_name, markdown_content, chunk_texts_list, metadata

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"❌ Error in advanced pipeline: {type(e).__name__}: {str(e)}")
            logger.error(f"Full traceback:\n{error_traceback}")
            metadata["steps"].append({
                "step": "error",
                "status": "failed",
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": error_traceback
            })
            raise
        finally:
            # Always cleanup database and FAISS index after generation (success or failure)
            self.cleanup_data()

    async def _select_candidate_chunks(
        self,
        chunks: List[Any],
        chunk_ids: List[str],
        embeddings: np.ndarray,
        num_questions: int,
        document_id: str = None,
        additional_prompt: str = None
    ) -> List[Dict[str, Any]]:
        
        # Calculate how many chunks we need
        num_chunks = self.chunk_selector.calculate_chunks_needed(num_questions)
        
        # Select based on mode
        if additional_prompt and len(additional_prompt.strip()) > 0:
            # Mode: Search-based selection with hybrid search
            logger.info(f"🔍 Using search-based selection with prompt: '{additional_prompt[:50]}...'")
            
            # Use cached embedding model instead of loading again
            
            candidates = self.chunk_selector.select_by_search(
                query=additional_prompt,
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                faiss_index=self.faiss_index,
                embedding_model=self.embedding_model,
                num_chunks=num_chunks,
                document_id=document_id
            )
        else:
            # Mode: Representative selection for comprehensive coverage
            logger.info("📊 Using representative selection for document coverage")
            
            candidates = self.chunk_selector.select_representative(
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                num_chunks=num_chunks
            )
            
            # Ensure section coverage
            candidates = self.chunk_selector.ensure_section_coverage(
                selected=candidates,
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                min_coverage=0.7
            )
        
        # Log selection summary
        total_tokens = sum(c["token_count"] for c in candidates)
        methods = {}
        for c in candidates:
            method = c.get("selection_method", "unknown")
            methods[method] = methods.get(method, 0) + 1
        
        logger.info(f"✓ Selected {len(candidates)} chunks ({total_tokens} tokens)")
        logger.info(f"  Selection breakdown: {methods}")
        
        return candidates

    async def _generate_questions_batch(
        self,
        candidate_chunks: List[Dict[str, Any]],
        num_questions: int,
        difficulty: str,
        language: str,
        question_types: List[int],
    ) -> tuple[List[QuizQuestion], List[str]]:
        """Generate all questions in a single LLM API call using batch mode."""
        
        from app.models.quiz import QuizOption, QuestionTypeEnum
        
        questions: List[QuizQuestion] = []
        question_chunk_ids: List[str] = []
        
        # Combine candidate chunks into a single content block
        combined_content = "\n\n---\n\n".join([
            f"[Section {idx+1}]\n{chunk['text']}" 
            for idx, chunk in enumerate(candidate_chunks[:num_questions])
        ])
        
        # Build options for batch generation
        options_cfg = {
            "difficulty": difficulty,
            "language": language,
            "question_types": question_types,
        }
        
        try:
            # Single API call to generate all questions
            batch_result = await self.llm_adapter.generate_batch_mcq(
                combined_content, num_questions, options=options_cfg
            )
            
            if batch_result and batch_result.questions:
                for idx, result in enumerate(batch_result.questions[:num_questions]):
                    if not result.question or not result.choices:
                        continue
                    
                    # Determine question type based on answer format
                    if isinstance(result.answer, list) and len(result.answer) > 1:
                        q_type = 1  # Multiple choice
                    else:
                        q_type = 0  # Single choice
                    
                    # Map answer letter(s) to index
                    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                    
                    if isinstance(result.answer, list):
                        correct_indices = [answer_map.get(a.strip().upper(), -1) for a in result.answer]
                        correct_indices = [i for i in correct_indices if i >= 0]
                    else:
                        correct_indices = [answer_map.get(str(result.answer).strip().upper(), 0)]

                    # Build options list - ID always defaults to 0
                    options = []
                    for opt_idx, opt_text in enumerate(result.choices):
                        options.append(
                            QuizOption(
                                id=0,
                                optionText=opt_text,
                                isCorrect=(opt_idx in correct_indices),
                            )
                        )

                    question = QuizQuestion(
                        id=0,
                        questionText=result.question,
                        questionType=QuestionTypeEnum(q_type),
                        point=1.0,
                        options=options,
                    )
                    questions.append(question)
                    
                    # Map to corresponding chunk ID
                    chunk_idx = min(idx, len(candidate_chunks) - 1)
                    question_chunk_ids.append(candidate_chunks[chunk_idx].get("chunk_id", ""))
                
                logger.info(f"Generated {len(questions)} questions in single batch API call")
                    
        except Exception as e:
            logger.error(f"Error in batch question generation: {e}")
            raise
        
        return questions, question_chunk_ids

    def cleanup_data(self):
        """Clear Neo4j database and FAISS index."""
        try:
            logger.info("🧹 Cleaning up database and FAISS index...")
            
            # Clear Neo4j database
            try:
                self.neo4j_db.execute_write("MATCH (n) DETACH DELETE n")
                logger.info("✓ Cleared Neo4j database")
            except Exception as e:
                logger.error(f"Error clearing Neo4j: {e}")
            
            # Clear FAISS index
            try:
                self.faiss_index.clear()
                self.faiss_index.save()
                logger.info("✓ Cleared FAISS index")
            except Exception as e:
                logger.error(f"Error clearing FAISS: {e}")
            
            logger.info("✓ Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def close(self):
        
        self.neo4j_db.close()
