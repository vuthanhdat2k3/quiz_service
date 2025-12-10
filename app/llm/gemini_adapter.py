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

    def _extract_json(self, text: str) -> Dict[str, Any]:
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        return json.loads(text)

    async def generate_mcq(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> MCQResult:
        
        prompt = self._build_mcq_prompt(passage)
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
