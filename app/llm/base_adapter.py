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
class BatchMCQResult:
    """Result containing multiple MCQ questions from a single API call"""
    questions: List[MCQResult]


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
    async def generate_batch_mcq(self, passage: str, num_questions: int, options: Optional[Dict[str, Any]] = None) -> BatchMCQResult:
        """Generate multiple MCQ questions in a single API call"""
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
        existing_questions = []
        question_index = 0
        
        if options:
            num_correct = options.get("num_correct", 1)
            language = options.get("language", "en")
            difficulty = options.get("difficulty", "medium")
            existing_questions = options.get("existing_questions", [])
            question_index = options.get("question_index", 0)
        
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
        
        # Build existing questions context to avoid duplicates
        avoid_section = ""
        if existing_questions:
            avoid_section = f"""
IMPORTANT: Do NOT create questions similar to these existing ones:
{chr(10).join(f'- {q}' for q in existing_questions[-5:])}

Create a UNIQUE and DIFFERENT question that tests a different aspect or concept from the topic.
"""
        
        # Add variety instruction based on question index
        variety_hints = [
            "Focus on definitions and key concepts.",
            "Focus on applications and examples.",
            "Focus on relationships and comparisons.",
            "Focus on causes and effects.",
            "Focus on processes and procedures.",
            "Focus on advantages and disadvantages.",
            "Focus on specific details and facts.",
        ]
        variety_instruction = variety_hints[question_index % len(variety_hints)]
        
        return f"""You are an exam question generator for academic content. Given the content below, {instruction} that tests understanding of the content.

{lang_instruction}

Question #{question_index + 1} focus: {variety_instruction}
{avoid_section}
Output only JSON with exactly these keys: {{"question":"", "choices":["","","",""], {answer_format}, "explanation":"", "difficulty":"easy|medium|hard"}}

Content:
"{passage}"

Output JSON:"""

    def _build_batch_mcq_prompt(self, passage: str, num_questions: int, options: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for generating multiple MCQ questions in one API call"""
        language = "en"
        difficulty = "medium"
        question_types = [0]  # Default to single choice
        
        if options:
            language = options.get("language", "en")
            difficulty = options.get("difficulty", "medium")
            question_types = options.get("question_types", [0])
        
        # Language instruction mapping
        lang_instruction = {
            "vi": "Generate ALL questions, choices, and explanations in Vietnamese (Tiếng Việt).",
            "en": "Generate ALL questions, choices, and explanations in English.",
        }.get(language, "Generate ALL questions, choices, and explanations in English.")
        
        # Determine question type distribution
        if 2 in question_types:  # Mix mode
            type_instruction = "Alternate between SINGLE correct answer and MULTIPLE correct answers (2 correct) questions."
            answer_format = 'For single-choice: "answer":"A" | For multiple-choice: "answer":["A","B"]'
        elif 1 in question_types:  # Multiple choice only
            type_instruction = "Each question must have exactly 2 correct answers."
            answer_format = '"answer":["A","B"]  // Array of 2 correct answer letters'
        else:  # Single choice only
            type_instruction = "Each question must have exactly ONE correct answer."
            answer_format = '"answer":"A"  // Single correct answer letter'
        
        # Variety hints for diverse questions
        variety_aspects = """
Each question MUST test a DIFFERENT aspect:
1. Definitions and key concepts
2. Applications and examples  
3. Relationships and comparisons
4. Causes and effects
5. Processes and procedures
6. Advantages and disadvantages
7. Specific details and facts
"""
        
        return f"""You are an expert exam question generator. Generate exactly {num_questions} UNIQUE multiple-choice questions based on the content below.

CRITICAL REQUIREMENTS:
- Generate exactly {num_questions} different questions
- Each question must test a DIFFERENT concept or aspect
- NO duplicate or similar questions
- {type_instruction}
- Difficulty level: {difficulty}
{variety_aspects}

{lang_instruction}

Output ONLY a JSON array with exactly {num_questions} objects, each with these keys:
[
  {{"question":"...", "choices":["A)...","B)...","C)...","D)..."], {answer_format}, "explanation":"...", "difficulty":"{difficulty}"}},
  ...
]

Content to generate questions from:
\"\"\"
{passage}
\"\"\"

Output JSON array (exactly {num_questions} questions):"""

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
