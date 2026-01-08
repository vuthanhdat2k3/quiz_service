import json
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from app.llm.base_adapter import (
    LLMAdapter,
    MCQResult,
    BatchMCQResult,
    DistractorResult,
    ShortAnswerResult,
    TrueFalseResult,
)
from app.core.config import get_settings

settings = get_settings()


class GeminiAdapter(LLMAdapter):

    def __init__(
        self, api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs
    ):
        model_name = model_name or settings.GEMINI_MODEL
        super().__init__(model_name=model_name, **kwargs)

        if genai is None:
            raise ImportError("google-generativeai package is required for GeminiAdapter")

        api_key = api_key or settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized GeminiAdapter with model {model_name}")

    async def _call_with_retry(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                    ),
                )
                return response.text
            except Exception as e:
                wait_time = 0.5 * (2**attempt)
                logger.warning(
                    f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise

    # Note: _repair_json and _extract_json are inherited from LLMAdapter base class

    async def generate_mcq(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> MCQResult:
        
        prompt = self._build_mcq_prompt(passage, options)
        response_text = await self._call_with_retry(prompt)

        try:
            data = self._extract_json(response_text)
            return MCQResult(
                question=data["question"],
                choices=data["choices"],
                answer=data["answer"],
                explanation=data["explanation"],
                difficulty=data["difficulty"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse MCQ response: {e}\nResponse: {response_text}")
            raise

    async def generate_batch_mcq(
        self, passage: str, num_questions: int, options: Optional[Dict[str, Any]] = None
    ) -> BatchMCQResult:
        """Generate multiple MCQ questions in a single API call"""
        
        prompt = self._build_batch_mcq_prompt(passage, num_questions, options)
        response_text = await self._call_with_retry(prompt)

        try:
            data = self._extract_json(response_text)
            
            # _extract_json already validates and returns {'questions': [...]}
            # Handle both array and object with "questions" key
            if isinstance(data, dict) and "questions" in data:
                questions_data = data["questions"]
            elif isinstance(data, list):
                questions_data = data
            else:
                raise ValueError(f"Expected array or object with 'questions' key, got {type(data)}")
            
            questions = []
            required_keys = {'question', 'choices', 'answer'}
            
            for idx, q_data in enumerate(questions_data):
                # Skip incomplete questions (already filtered by _extract_json, but double-check)
                if not isinstance(q_data, dict):
                    logger.warning(f"Skipping non-dict question at index {idx}")
                    continue
                    
                missing_keys = required_keys - set(q_data.keys())
                if missing_keys:
                    logger.warning(f"Skipping question {idx} missing keys: {missing_keys}")
                    continue
                
                # Clean up choices if they have letter prefixes like "A) ..."
                choices = q_data.get("choices", [])
                cleaned_choices = []
                for choice in choices:
                    # Remove "A) ", "B) ", etc. prefixes if present
                    if isinstance(choice, str) and len(choice) > 2 and choice[1] in ")]:.-" and choice[0].upper() in "ABCD":
                        cleaned_choices.append(choice[2:].strip())
                    else:
                        cleaned_choices.append(str(choice) if choice else "")
                
                questions.append(MCQResult(
                    question=q_data["question"],
                    choices=cleaned_choices if cleaned_choices else choices,
                    answer=q_data["answer"],
                    explanation=q_data.get("explanation", ""),
                    difficulty=q_data.get("difficulty", "medium"),
                ))
            
            if not questions:
                raise ValueError("No valid questions generated after filtering incomplete responses")
            
            logger.info(f"Generated {len(questions)} questions in single API call")
            return BatchMCQResult(questions=questions)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse batch MCQ response: {e}\\nResponse: {response_text[:500]}")
            raise

    async def refine_distractors(
        self, passage: str, correct_answer: str, candidates: List[str]
    ) -> DistractorResult:
        
        prompt = self._build_distractor_prompt(passage, correct_answer, candidates)
        response_text = await self._call_with_retry(prompt)

        try:
            distractors = self._extract_json(response_text)
            if not isinstance(distractors, list):
                distractors = distractors.get("distractors", distractors)
            if not isinstance(distractors, list) or len(distractors) < 3:
                raise ValueError(f"Expected at least 3 distractors, got {len(distractors)}")
            return DistractorResult(distractors=distractors[:3])
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse distractor response: {e}\nResponse: {response_text}")
            raise

    async def generate_short_answer(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> ShortAnswerResult:
        
        prompt = self._build_short_answer_prompt(passage, options)
        response_text = await self._call_with_retry(prompt)

        try:
            data = self._extract_json(response_text)
            return ShortAnswerResult(
                question=data["question"],
                answer=data["answer"],
                explanation=data["explanation"],
                difficulty=data["difficulty"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse short answer response: {e}")
            raise

    async def generate_true_false(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> TrueFalseResult:
        
        prompt = self._build_true_false_prompt(passage, options)
        response_text = await self._call_with_retry(prompt)

        try:
            data = self._extract_json(response_text)
            return TrueFalseResult(
                statement=data["statement"],
                answer=data["answer"],
                explanation=data["explanation"],
                difficulty=data["difficulty"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse true/false response: {e}")
            raise
