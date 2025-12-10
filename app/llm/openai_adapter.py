import json
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from app.llm.base_adapter import (
    LLMAdapter,
    MCQResult,
    DistractorResult,
    ShortAnswerResult,
    TrueFalseResult,
)
from app.core.config import get_settings

settings = get_settings()


class OpenAIAdapter(LLMAdapter):

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-turbo-preview", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

        if AsyncOpenAI is None:
            raise ImportError("openai package is required for OpenAIAdapter")

        api_key = api_key or settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        self.client = AsyncOpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAIAdapter with model {model_name}")

    async def _call_with_retry(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert exam question generator."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                wait_time = 0.5 * (2**attempt)
                logger.warning(f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def generate_mcq(self, passage: str, options: Optional[Dict[str, Any]] = None) -> MCQResult:
        prompt = self._build_mcq_prompt(passage)
        response_text = await self._call_with_retry(prompt)

        try:
            data = json.loads(response_text)
            return MCQResult(
                question=data["question"],
                choices=data["choices"],
                answer=data["answer"],
                explanation=data["explanation"],
                difficulty=data["difficulty"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse MCQ response: {e}")
            raise

    async def refine_distractors(
        self, passage: str, correct_answer: str, candidates: List[str]
    ) -> DistractorResult:
        prompt = self._build_distractor_prompt(passage, correct_answer, candidates)
        response_text = await self._call_with_retry(prompt)

        try:
            distractors = json.loads(response_text)
            if not isinstance(distractors, list) or len(distractors) != 3:
                raise ValueError(f"Expected 3 distractors, got {len(distractors)}")
            return DistractorResult(distractors=distractors)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse distractor response: {e}")
            raise

    async def generate_short_answer(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> ShortAnswerResult:
        prompt = self._build_short_answer_prompt(passage, options)
        response_text = await self._call_with_retry(prompt)

        try:
            data = json.loads(response_text)
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
            data = json.loads(response_text)
            return TrueFalseResult(
                statement=data["statement"],
                answer=data["answer"],
                explanation=data["explanation"],
                difficulty=data["difficulty"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse true/false response: {e}")
            raise
