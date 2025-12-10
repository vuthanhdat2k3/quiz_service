import os
import tempfile
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from loguru import logger
import numpy as np
import aiofiles

from app.database.neo4j_db import Neo4jDatabase
from app.database.faiss_index import FAISSIndex
from app.models.quiz import QuizQuestion, QuizOption
from app.parsers import ParserFactory
from app.chunkers.markdown_chunker import MarkdownChunkerV2
from app.llm import get_llm_adapter
from app.core.config import get_settings
from app.graph import DocumentGraphBuilder, DocumentNode, SectionNode, ChunkNode
from app.services.chunk_selector import ChunkSelector, ChunkSelectionConfig


class QuizGenerationService:

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

            # Step 4: Compute & store embeddings
            logger.info("🔢 Step 4: Computing embeddings and storing in FAISS...")
            from sentence_transformers import SentenceTransformer
            
            embedding_model = SentenceTransformer(self.settings.EMBEDDING_MODEL)
            chunk_texts = [c.text for c in chunks]
            embeddings = embedding_model.encode(chunk_texts, convert_to_numpy=True)
            
            # Store in FAISS with chunk IDs and metadata for hybrid search
            metadata_list = []
            for idx, chunk in enumerate(chunks):
                metadata_list.append({
                    "document_id": document_id,
                    "chunk_index": idx,
                    "token_count": chunk.token_count,
                    "chunk_type": chunk.chunk_type,
                    "section_path": chunk.section_path if hasattr(chunk, 'section_path') else [],
                    "text": chunk.text[:500],  # Store more text for BM25 search
                    "heading": chunk.metadata.get("heading", "")
                })
            
            self.faiss_index.add_embeddings_batch(chunk_ids, embeddings, metadata_list)
            
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

            # Step 6: Generate questions with LLM
            logger.info("🤖 Step 6: Generating questions with LLM...")
            questions, question_chunk_ids = await self._generate_questions_from_candidates(
                candidate_chunks=candidate_chunks,
                num_questions=num_questions,
                difficulty=difficulty,
                language=language,
                question_types=question_types,
                additional_prompt=additional_prompt
            )
            
            logger.info(f"✓ Generated {len(questions)} questions")
            metadata["steps"].append({
                "step": 6,
                "name": "question_generation",
                "status": "success",
                "num_questions": len(questions)
            })

            # Step 7: Generate distractors using graph neighbors
            logger.info("🔀 Step 7: Generating distractors from graph...")
            questions = await self._enhance_distractors_from_graph(
                questions=questions,
                question_chunk_ids=question_chunk_ids
            )
            
            logger.info(f"✓ Enhanced distractors for {len(questions)} questions")
            metadata["steps"].append({
                "step": 7,
                "name": "distractor_generation",
                "status": "success"
            })

            # Step 8: Store questions in Neo4j
            logger.info("💾 Step 8: Storing questions in Neo4j...")
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
                "step": 8,
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

    async def _select_candidate_chunks(
        self,
        chunks: List[Any],
        chunk_ids: List[str],
        embeddings: np.ndarray,
        num_questions: int,
        document_id: str = None,
        additional_prompt: str = None
    ) -> List[Dict[str, Any]]:
        
        from sentence_transformers import SentenceTransformer
        
        # Calculate how many chunks we need
        num_chunks = self.chunk_selector.calculate_chunks_needed(num_questions)
        
        # Select based on mode
        if additional_prompt and len(additional_prompt.strip()) > 0:
            # Mode: Search-based selection with hybrid search
            logger.info(f"🔍 Using search-based selection with prompt: '{additional_prompt[:50]}...'")
            
            embedding_model = SentenceTransformer(self.settings.EMBEDDING_MODEL)
            
            candidates = self.chunk_selector.select_by_search(
                query=additional_prompt,
                chunks=chunks,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                faiss_index=self.faiss_index,
                embedding_model=embedding_model,
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

    async def _generate_questions_from_candidates(
        self,
        candidate_chunks: List[Dict[str, Any]],
        num_questions: int,
        difficulty: str,
        language: str,
        question_types: List[int],
        additional_prompt: Optional[str] = None
    ) -> tuple[List[QuizQuestion], List[str]]:
        
        from app.models.quiz import QuizOption, QuestionTypeEnum
        
        questions: List[QuizQuestion] = []
        question_chunk_ids: List[str] = []
        existing_question_texts: List[str] = []  # Track generated questions to avoid duplicates
        questions_per_chunk = max(1, num_questions // len(candidate_chunks))
        
        # Determine question type distribution
        if 2 in question_types:  # Mix mode
            actual_types = [0, 1]  # Single and Multiple choice
        else:
            actual_types = question_types
        
        for idx, chunk in enumerate(candidate_chunks[:num_questions]):
            if len(questions) >= num_questions:
                break
            
            # Alternate question types
            q_type = actual_types[idx % len(actual_types)]
            
            try:
                # Generate MCQ for this chunk
                if q_type in [0, 1]:  # Single or Multiple choice
                    # Call adapter with passage and optional params
                    options_cfg = {
                        "difficulty": difficulty,
                        "language": language,
                        "num_correct": 1 if q_type == 0 else 2,
                        "existing_questions": existing_question_texts,
                        "question_index": idx,
                    }
                    result = await self.llm_adapter.generate_mcq(
                        chunk["text"], options=options_cfg
                    )

                    if result and result.question and result.choices:
                        # Map answer letter(s) to index
                        answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                        
                        # Handle both single answer (string) and multiple answers (list/array)
                        if isinstance(result.answer, list):
                            # Multiple correct answers: ["A", "B"]
                            correct_indices = [answer_map.get(a.strip().upper(), -1) for a in result.answer]
                            correct_indices = [i for i in correct_indices if i >= 0]  # Filter invalid
                        else:
                            # Single correct answer: "A"
                            correct_indices = [answer_map.get(str(result.answer).strip().upper(), 0)]

                        # Build options list - ID always defaults to 0
                        options = []
                        for opt_idx, opt_text in enumerate(result.choices):
                            options.append(
                                QuizOption(
                                    id=0,  # Always default to 0
                                    optionText=opt_text,
                                    isCorrect=(opt_idx in correct_indices),
                                )
                            )

                        question = QuizQuestion(
                            id=0,  # Always default to 0
                            questionText=result.question,
                            questionType=QuestionTypeEnum(q_type),
                            point=1.0,
                            options=options,
                        )
                        questions.append(question)
                        question_chunk_ids.append(chunk.get("chunk_id", ""))
                        
                        # Track generated question to avoid duplicates
                        existing_question_texts.append(result.question)

                        logger.info(
                            f"Generated question {len(questions)}/{num_questions} (type={q_type})"
                        )
                
            except Exception as e:
                logger.error(f"Error generating question from chunk {idx}: {e}")
                continue
        
        return questions, question_chunk_ids

    async def _enhance_distractors_from_graph(
        self,
        questions: List[QuizQuestion],
        question_chunk_ids: List[str]
    ) -> List[QuizQuestion]:
        
        enhanced_questions = []
        
        for idx, question in enumerate(questions):
            try:
                # Get neighbor chunks from Neo4j for the source chunk (limited to 5)
                src_chunk_id = question_chunk_ids[idx] if idx < len(question_chunk_ids) else None
                if not src_chunk_id:
                    enhanced_questions.append(question)
                    continue
                neighbor_chunks = self.neo4j_db.get_neighbor_chunks(src_chunk_id, same_section=True, limit=5)
                
                if not neighbor_chunks or len(neighbor_chunks) < 2:
                    # Not enough neighbors, keep original
                    enhanced_questions.append(question)
                    continue
                
                # Extract text from neighbors (exclude correct answer context)
                neighbor_texts = [chunk.get("content", "") for chunk in neighbor_chunks]
                neighbor_context = "\n".join(neighbor_texts[:3])  # Use top 3
                
                # Get current incorrect options
                incorrect_options = [
                    opt for opt in question.options if not opt.isCorrect
                ]
                
                if len(incorrect_options) < 2:
                    # Not enough to enhance
                    enhanced_questions.append(question)
                    continue
                
                # Use LLM to refine distractors using neighbor context
                try:
                    # Pick the first correct answer (adapter expects single string)
                    correct_candidates = [opt.optionText for opt in question.options if opt.isCorrect]
                    correct_answer = correct_candidates[0] if correct_candidates else ""
                    current_distractors = [opt.optionText for opt in incorrect_options]

                    result = await self.llm_adapter.refine_distractors(
                        passage=neighbor_context,
                        correct_answer=correct_answer,
                        candidates=current_distractors,
                    )

                    if result and result.distractors and len(result.distractors) >= 2:
                        # Replace incorrect options with refined distractors
                        new_options = [opt for opt in question.options if opt.isCorrect]
                        
                        for idx, distractor in enumerate(result.distractors[:3]):
                            new_options.append(QuizOption(
                                id=0,  # Always default to 0
                                optionText=distractor,
                                isCorrect=False
                            ))
                        
                        # Create enhanced question
                        from app.models.quiz import QuizQuestion as QQ
                        enhanced_q = QQ(
                            id=0,  # Always default to 0
                            questionText=question.questionText,
                            questionType=question.questionType,
                            point=question.point,
                            options=new_options
                        )
                        enhanced_questions.append(enhanced_q)
                        logger.info(f"Enhanced distractors for question {question.id}")
                    else:
                        enhanced_questions.append(question)
                        
                except Exception as e:
                    logger.warning(f"Failed to refine distractors: {e}")
                    enhanced_questions.append(question)
                    
            except Exception as e:
                logger.error(f"Error enhancing question {question.id}: {e}")
                enhanced_questions.append(question)
        
        return enhanced_questions

    def close(self):
        
        self.neo4j_db.close()
