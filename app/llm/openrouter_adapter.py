import json
import asyncio
import re
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAdapter(LLMAdapter):

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "google/gemini-2.0-flash-001",
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)

        if AsyncOpenAI is None:
            raise ImportError("openai package is required for OpenRouterAdapter")

        api_key = api_key or settings.OPENROUTER_API_KEY
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        logger.info(f"Initialized OpenRouterAdapter with model {model_name}")

    def _extract_json_from_response(self, text: str) -> str:
        text = text.strip()
        
        # Try to find JSON in markdown code block
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # Try to find raw JSON object or array
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            return json_match.group(1).strip()
        
        return text

    async def _call_with_retry(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert exam question generator. Always respond with valid JSON only, no markdown formatting."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    extra_headers={
                        "HTTP-Referer": "https://ptit-ocm.edu.vn",
                        "X-Title": "PTIT Quiz Service",
                    },
                )
                raw_content = response.choices[0].message.content
                return self._extract_json_from_response(raw_content)
            except Exception as e:
                wait_time = 0.5 * (2**attempt)
                logger.warning(f"OpenRouter API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def generate_mcq(self, passage: str, options: Optional[Dict[str, Any]] = None) -> MCQResult:
        prompt = self._build_mcq_prompt(passage, options)
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
            logger.error(f"Failed to parse MCQ response: {e}\nRaw response: {response_text}")
            raise

    async def generate_batch_mcq(
        self, passage: str, num_questions: int, options: Optional[Dict[str, Any]] = None
    ) -> BatchMCQResult:
        """Generate multiple MCQ questions in a single API call"""
        
        prompt = self._build_batch_mcq_prompt(passage, num_questions, options)
        response_text = await self._call_with_retry(prompt)

        try:
            data = json.loads(response_text)
            
            # Handle both array and object with "questions" key
            if isinstance(data, dict) and "questions" in data:
                questions_data = data["questions"]
            elif isinstance(data, list):
                questions_data = data
            else:
                raise ValueError(f"Expected array or object with 'questions' key, got {type(data)}")
            
            questions = []
            for q_data in questions_data:
                # Clean up choices if they have letter prefixes like "A) ..."
                choices = q_data.get("choices", [])
                cleaned_choices = []
                for choice in choices:
                    if len(choice) > 2 and choice[1] in ")]:.-" and choice[0].upper() in "ABCD":
                        cleaned_choices.append(choice[2:].strip())
                    else:
                        cleaned_choices.append(choice)
                
                questions.append(MCQResult(
                    question=q_data["question"],
                    choices=cleaned_choices if cleaned_choices else choices,
                    answer=q_data["answer"],
                    explanation=q_data.get("explanation", ""),
                    difficulty=q_data.get("difficulty", "medium"),
                ))
            
            logger.info(f"Generated {len(questions)} questions in single API call")
            return BatchMCQResult(questions=questions)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse batch MCQ response: {e}\nRaw response: {response_text}")
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
            logger.error(f"Failed to parse distractor response: {e}\nRaw response: {response_text}")
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
            logger.error(f"Failed to parse short answer response: {e}\nRaw response: {response_text}")
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
            logger.error(f"Failed to parse true/false response: {e}\nRaw response: {response_text}")
            raise
