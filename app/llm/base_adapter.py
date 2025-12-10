from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class MCQResult:

    question: str
    choices: List[str]
    answer: Union[str, List[str]]  # Single: "A" or Multiple: ["A", "B"]
    explanation: str
    difficulty: str


@dataclass
class DistractorResult:

    distractors: List[str]


@dataclass
class ShortAnswerResult:

    question: str
    answer: str
    explanation: str
    difficulty: str


@dataclass
class TrueFalseResult:

    statement: str
    answer: bool
    explanation: str
    difficulty: str


class LLMAdapter(ABC):

    def __init__(self, model_name: str, temperature: float = 0.2, max_retries: int = 3):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

    @abstractmethod
    async def generate_mcq(self, passage: str, options: Optional[Dict[str, Any]] = None) -> MCQResult:
        pass

    @abstractmethod
    async def refine_distractors(
        self, passage: str, correct_answer: str, candidates: List[str]
    ) -> DistractorResult:
        pass

    @abstractmethod
    async def generate_short_answer(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> ShortAnswerResult:
        pass

    @abstractmethod
    async def generate_true_false(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> TrueFalseResult:
        pass

    def _build_mcq_prompt(self, passage: str, options: Optional[Dict[str, Any]] = None) -> str:
        num_correct = 1
        language = "en"
        difficulty = "medium"
        
        if options:
            num_correct = options.get("num_correct", 1)
            language = options.get("language", "en")
            difficulty = options.get("difficulty", "medium")
        
        # Language instruction mapping
        lang_instruction = {
            "vi": "Generate the question, choices, and explanation in Vietnamese (Tiếng Việt).",
            "en": "Generate the question, choices, and explanation in English.",
        }.get(language, "Generate the question, choices, and explanation in English.")
        
        if num_correct == 1:
            instruction = "generate ONE multiple-choice question with exactly ONE correct answer"
            answer_format = '"answer":"A|B|C|D"'
        else:
            instruction = f"generate ONE multiple-choice question with exactly {num_correct} correct answers"
            answer_format = '"answer":["A","B"]  // Array of correct answer letters'
        
        return f"""You are an exam question generator for academic content. Given the short Markdown-derived chunk below (<=150 tokens), {instruction} that tests understanding of the chunk.

{lang_instruction}

Output only JSON with exactly these keys: {{"question":"", "choices":["","","",""], {answer_format}, "explanation":"", "difficulty":"easy|medium|hard"}}

Chunk:
"{passage}"

Output JSON:"""

    def _build_distractor_prompt(self, passage: str, correct_answer: str, candidates: List[str]) -> str:
        candidates_json = [f'"{c}"' for c in candidates[:6]]
        return f"""Input JSON:
{{"passage":"{passage}", "correct":"{correct_answer}", "candidates":[{','.join(candidates_json)}]}}

Task: Return exactly three distractors (strings) that are plausible but incorrect given the passage. Do not repeat the correct answer. Ensure distractors are not directly supported by the passage.

Output: JSON array: ["...","...","..."]"""

    def _build_short_answer_prompt(self, passage: str, options: Optional[Dict[str, Any]] = None) -> str:
        language = "en"
        if options:
            language = options.get("language", "en")
        
        lang_instruction = {
            "vi": "Generate the question, answer, and explanation in Vietnamese (Tiếng Việt).",
            "en": "Generate the question, answer, and explanation in English.",
        }.get(language, "Generate the question, answer, and explanation in English.")
        
        return f"""Generate a short answer question from the following passage. The question should require a brief text response (1-3 sentences).

{lang_instruction}

Output only JSON: {{"question":"", "answer":"", "explanation":"", "difficulty":"easy|medium|hard"}}

Passage:
"{passage}"

Output JSON:"""

    def _build_true_false_prompt(self, passage: str, options: Optional[Dict[str, Any]] = None) -> str:
        language = "en"
        if options:
            language = options.get("language", "en")
        
        lang_instruction = {
            "vi": "Generate the statement and explanation in Vietnamese (Tiếng Việt).",
            "en": "Generate the statement and explanation in English.",
        }.get(language, "Generate the statement and explanation in English.")
        
        return f"""Generate a true/false statement based on the following passage. The statement should test understanding of a key fact.

{lang_instruction}

Output only JSON: {{"statement":"", "answer":true|false, "explanation":"", "difficulty":"easy|medium|hard"}}

Passage:
"{passage}"

Output JSON:"""
