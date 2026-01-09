import os
import tempfile
import uuid
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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
from app.models.quiz import (
    QuizQuestion,
    QuizOption,
    DifficultyEnum,
    QuestionTypeEnum,
    DifficultyDistribution,
    QuestionTypeDistribution,
    PointStrategyEnum,
)
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


@dataclass
class QuestionPlanItem:
    difficulty: str  # easy | medium | hard
    question_type: int  # 0 = single, 1 = multiple


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
            semantic_weight=0.7,
            enable_relevance_detection=True,  # Bật relevance detection
            relevance_strict_mode=False  # Chế độ thông thường (không quá chặt)
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

    @staticmethod
    def _interleave_counts(counts: Dict[str, int], order: List[str]) -> List[str]:
        """Turn count dict into a balanced sequence preserving declared order priority."""
        normalized = {key: max(0, int(counts.get(key, 0))) for key in order}
        sequence: List[str] = []
        while sum(normalized.values()) > 0:
            # Always pick the bucket with the most remaining items first (stable on order)
            for key in sorted(order, key=lambda k: (-normalized[k], order.index(k))):
                if normalized[key] > 0:
                    sequence.append(key)
                    normalized[key] -= 1
        return sequence

    def _build_question_plan(
        self,
        difficulty_counts: Dict[str, int],
        type_counts: Dict[str, int],
    ) -> List[QuestionPlanItem]:
        """Validate inputs and build a per-question blueprint."""

        total_questions = sum(max(0, int(v)) for v in difficulty_counts.values())
        type_total = sum(max(0, int(v)) for v in type_counts.values())

        if total_questions <= 0:
            raise ValueError("Tổng số câu hỏi phải lớn hơn 0")
        if total_questions > self.settings.MAX_NUM_QUESTIONS:
            raise ValueError(
                f"Tổng số câu hỏi không được vượt quá {self.settings.MAX_NUM_QUESTIONS}"
            )
        if type_total != total_questions:
            raise ValueError("Tổng số câu hỏi theo loại phải bằng tổng số câu theo độ khó")

        diff_seq = self._interleave_counts(
            difficulty_counts, [DifficultyEnum.EASY.value, DifficultyEnum.MEDIUM.value, DifficultyEnum.HARD.value]
        )
        type_seq = self._interleave_counts(type_counts, ["single", "multiple"])

        if len(diff_seq) != len(type_seq):
            raise ValueError("Không thể ghép độ khó và loại câu hỏi – kiểm tra lại số lượng nhập vào")

        plan: List[QuestionPlanItem] = []
        for idx in range(len(diff_seq)):
            plan.append(
                QuestionPlanItem(
                    difficulty=diff_seq[idx],
                    question_type=0 if type_seq[idx] == "single" else 1,
                )
            )

        return plan

    @staticmethod
    def _serialize_plan(plan: List[QuestionPlanItem]) -> List[Dict[str, Any]]:
        return [
            {
                "index": idx + 1,
                "difficulty": item.difficulty,
                "type": "single_choice" if item.question_type == 0 else "multiple_choice",
            }
            for idx, item in enumerate(plan)
        ]

    def _assign_points(
        self,
        questions: List[QuizQuestion],
        plan: List[QuestionPlanItem],
        total_points: float,
        strategy: PointStrategyEnum,
    ) -> None:
        """Distribute total_points across questions based on strategy."""
        if not questions:
            return

        total_questions = len(questions)

        if strategy == PointStrategyEnum.EQUAL or total_questions == 0:
            base = total_points / total_questions
            accumulated = 0.0
            for idx, q in enumerate(questions):
                if idx == total_questions - 1:
                    q.point = round(total_points - accumulated, 2)
                else:
                    q.point = round(base, 2)
                    accumulated += q.point
            return

        # Difficulty-weighted: easy > medium > hard, still sum to total_points
        weights = {
            DifficultyEnum.EASY.value: 3,
            DifficultyEnum.MEDIUM.value: 2,
            DifficultyEnum.HARD.value: 1,
        }

        weight_sum = 0
        item_weights: List[int] = []
        for idx in range(total_questions):
            difficulty = plan[idx].difficulty if idx < len(plan) else DifficultyEnum.MEDIUM.value
            w = weights.get(difficulty, 1)
            item_weights.append(w)
            weight_sum += w

        if weight_sum == 0:
            # Fallback to equal distribution if weights collapse
            return self._assign_points(questions, plan, total_points, PointStrategyEnum.EQUAL)

        accumulated = 0.0
        for idx, q in enumerate(questions):
            raw_point = total_points * item_weights[idx] / weight_sum
            if idx == total_questions - 1:
                q.point = round(total_points - accumulated, 2)
            else:
                q.point = round(raw_point, 2)
                accumulated += q.point

    def _build_questions_from_batch_result(
        self,
        batch_result: Any,
        question_plan: List[QuestionPlanItem],
    ) -> List[QuizQuestion]:
        """Map LLM batch result into QuizQuestion objects following the blueprint."""
        questions: List[QuizQuestion] = []

        if not batch_result or not getattr(batch_result, "questions", None):
            return questions

        answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        for idx, result in enumerate(batch_result.questions[: len(question_plan)]):
            plan_item = question_plan[idx]

            raw_answers = result.answer
            answers_normalized: List[str]
            if plan_item.question_type == 1:
                if isinstance(raw_answers, list):
                    answers_normalized = [str(a).strip().upper() for a in raw_answers]
                else:
                    answers_normalized = [str(raw_answers).strip().upper()]
            else:
                if isinstance(raw_answers, list) and raw_answers:
                    answers_normalized = [str(raw_answers[0]).strip().upper()]
                else:
                    answers_normalized = [str(raw_answers).strip().upper()]

            answers_normalized = [a for a in answers_normalized if a]
            if not answers_normalized:
                answers_normalized = ["A"]

            correct_indices = [answer_map.get(letter, -1) for letter in answers_normalized]
            correct_indices = [i for i in correct_indices if i >= 0]
            if not correct_indices:
                correct_indices = [0]

            options: List[QuizOption] = []
            for opt_idx, opt_text in enumerate(result.choices or []):
                options.append(
                    QuizOption(
                        id=0,
                        optionText=opt_text,
                        isCorrect=(opt_idx in correct_indices),
                    )
                )

            if not options:
                # Ensure at least 4 options exist to avoid downstream issues
                options = [
                    QuizOption(id=0, optionText=f"Option {chr(65 + i)}", isCorrect=(i == 0))
                    for i in range(4)
                ]

            questions.append(
                QuizQuestion(
                    id=0,
                    questionText=result.question,
                    questionType=QuestionTypeEnum(plan_item.question_type),
                    difficulty=DifficultyEnum(plan_item.difficulty),
                    point=1.0,
                    options=options,
                )
            )

        return questions
    
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
        difficulty_distribution: DifficultyDistribution,
        question_type_distribution: QuestionTypeDistribution,
        language: str,
        total_points: float,
        point_strategy: PointStrategyEnum,
    ) -> List[QuizQuestion]:
        
        difficulty_counts = difficulty_distribution.dict()
        type_counts = question_type_distribution.dict()
        question_plan = self._build_question_plan(difficulty_counts, type_counts)

        logger.info(
            f"Generating {len(question_plan)} quiz questions from prompt with plan: {self._serialize_plan(question_plan)}"
        )

        try:
            options_cfg = {
                "language": language,
                "question_plan": self._serialize_plan(question_plan),
                "difficulty_counts": difficulty_counts,
                "type_counts": type_counts,
            }
            
            batch_result = await self.llm_adapter.generate_batch_mcq(
                prompt,
                len(question_plan),
                options=options_cfg,
            )

            questions = self._build_questions_from_batch_result(
                batch_result=batch_result,
                question_plan=question_plan,
            )
            self._assign_points(questions, question_plan, total_points, point_strategy)
            logger.info(
                f"Successfully generated {len(questions)}/{len(question_plan)} questions in single API call"
            )
            return questions

        except Exception as e:
            logger.error(f"Error generating questions from prompt: {e}")
            raise
        finally:
            # Always cleanup database and FAISS index after generation (success or failure)
            self.cleanup_data()

    async def generate_from_file_advanced(
        self,
        file_path: str,
        difficulty_distribution: DifficultyDistribution,
        question_type_distribution: QuestionTypeDistribution,
        language: str,
        total_points: float,
        point_strategy: PointStrategyEnum,
        additional_prompt: Optional[str] = None,
    ) -> tuple[List[QuizQuestion], str, str, List[str], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        
        logger.info(f"🚀 Starting advanced quiz generation for: {file_path}")
        document_id = str(uuid.uuid4())
        metadata = {"steps": []}

        question_plan = self._build_question_plan(
            difficulty_counts=difficulty_distribution.dict(),
            type_counts=question_type_distribution.dict(),
        )
        num_questions = len(question_plan)

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
            candidate_chunks, relevance_metadata = await self._select_candidate_chunks(
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                num_questions=num_questions,
                document_id=document_id,
                additional_prompt=additional_prompt
            )
            
            logger.info(f"✓ Selected {len(candidate_chunks)} candidate chunks")
            step5_info = {
                "step": 5,
                "name": "candidate_selection",
                "status": "success",
                "num_candidates": len(candidate_chunks)
            }
            
            # Thêm relevance info nếu có
            if relevance_metadata:
                step5_info.update(relevance_metadata)
            
            metadata["steps"].append(step5_info)

            # Step 6: Generate questions with LLM (single batch call)
            logger.info("🤖 Step 6: Generating questions with LLM (batch mode)...")
            questions, question_chunk_ids = await self._generate_questions_batch(
                candidate_chunks=candidate_chunks,
                question_plan=question_plan,
                language=language,
            )
            self._assign_points(questions, question_plan, total_points, point_strategy)

            logger.info(f"✓ Generated {len(questions)} questions in single API call")
            metadata["steps"].append({
                "step": 6,
                "name": "question_generation",
                "status": "success",
                "num_questions": len(questions),
                "point_strategy": point_strategy.value,
                "total_points": total_points,
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
                    difficulty=question.difficulty.value,
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
            return questions, file_name, markdown_content, chunk_texts_list, metadata, relevance_metadata

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
        additional_prompt: str = None,
        document_overview: str = None
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Select candidate chunks with optional relevance detection.
        
        Returns:
            Tuple of (candidate_chunks, relevance_metadata)
        """
        # Calculate how many chunks we need
        num_chunks = self.chunk_selector.calculate_chunks_needed(num_questions)
        
        relevance_metadata = None
        
        # Select based on mode
        if additional_prompt and len(additional_prompt.strip()) > 0:
            # Mode: Search-based selection with hybrid search + relevance detection
            logger.info(f"🔍 Using search-based selection with prompt: '{additional_prompt[:50]}...'")
            
            # Generate document overview for relevance detection (first 3 chunks)
            if not document_overview and len(chunks) > 0:
                overview_texts = [c.text for c in chunks[:3]]
                document_overview = "\n".join(overview_texts)[:1000]
            
            # Use cached embedding model instead of loading again
            candidates, relevance_result = self.chunk_selector.select_by_search(
                query=additional_prompt,
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                faiss_index=self.faiss_index,
                embedding_model=self.embedding_model,
                num_chunks=num_chunks,
                document_id=document_id,
                document_overview=document_overview
            )
            
            # Build relevance metadata để trả về cho caller
            if relevance_result:
                relevance_metadata = {
                    "query_relevance": {
                        "is_relevant": relevance_result.is_relevant,
                        "relevance_score": relevance_result.relevance_score,
                        "confidence": relevance_result.confidence,
                        "strategy_used": relevance_result.strategy,
                        "warning_message": relevance_result.warning_message,
                        "details": relevance_result.details
                    }
                }
                
                # Log warning nếu có
                if relevance_result.warning_message:
                    logger.warning(f"⚠️ {relevance_result.warning_message}")
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
        
        return candidates, relevance_metadata

    async def _generate_questions_batch(
        self,
        candidate_chunks: List[Dict[str, Any]],
        question_plan: List[QuestionPlanItem],
        language: str,
    ) -> tuple[List[QuizQuestion], List[str]]:
        """Generate all questions in a single LLM API call using batch mode."""

        questions: List[QuizQuestion] = []
        question_chunk_ids: List[str] = []
        num_questions = len(question_plan)

        combined_content = "\n\n---\n\n".join([
            f"[Section {idx+1}]\n{chunk['text']}" 
            for idx, chunk in enumerate(candidate_chunks[:num_questions])
        ])

        # Summaries for prompt clarity
        difficulty_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        type_counts: Dict[str, int] = {"single": 0, "multiple": 0}
        for item in question_plan:
            difficulty_counts[item.difficulty] = difficulty_counts.get(item.difficulty, 0) + 1
            if item.question_type == 0:
                type_counts["single"] += 1
            else:
                type_counts["multiple"] += 1

        options_cfg = {
            "language": language,
            "question_plan": self._serialize_plan(question_plan),
            "difficulty_counts": difficulty_counts,
            "type_counts": type_counts,
        }

        try:
            batch_result = await self.llm_adapter.generate_batch_mcq(
                combined_content,
                num_questions,
                options=options_cfg,
            )

            questions = self._build_questions_from_batch_result(
                batch_result=batch_result,
                question_plan=question_plan,
            )

            for idx in range(len(questions)):
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
