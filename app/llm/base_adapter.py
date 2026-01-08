from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import re
from loguru import logger

from app.llm.prompts import (
    MCQ_PROMPT_TEMPLATE,
    BATCH_MCQ_PROMPT_TEMPLATE,
    DISTRACTOR_PROMPT_TEMPLATE,
    SHORT_ANSWER_PROMPT_TEMPLATE,
    TRUE_FALSE_PROMPT_TEMPLATE,
    LANGUAGE_INSTRUCTIONS,
    LANGUAGE_INSTRUCTIONS_ALL,
    LANGUAGE_INSTRUCTIONS_SHORT,
    LANGUAGE_INSTRUCTIONS_TF,
    VARIETY_HINTS,
    SINGLE_CHOICE_INSTRUCTION,
    MULTIPLE_CHOICE_INSTRUCTION_TEMPLATE,
    SINGLE_ANSWER_FORMAT,
    MULTIPLE_ANSWER_FORMAT,
    AVOID_DUPLICATION_TEMPLATE,
)


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

    def _repair_json(self, text: str) -> str:
        """
        Attempt to repair common JSON issues from LLM output.
        Handles: unterminated strings, missing brackets, trailing commas, truncated responses.
        
        Strategy: Extract complete JSON objects and build a valid array from them.
        """
        original_text = text
        
        # Step 1: Find the start of JSON (array or object)
        json_start = -1
        is_array = False
        for i, char in enumerate(text):
            if char == '[':
                json_start = i
                is_array = True
                break
            elif char == '{':
                json_start = i
                is_array = False
                break
        
        if json_start == -1:
            logger.warning("No JSON structure found in text")
            return text
        
        text = text[json_start:]
        
        # Step 2: Try to parse as-is first (may already be valid)
        try:
            json.loads(text)
            return text  # Already valid JSON
        except json.JSONDecodeError:
            pass  # Continue with repair
        
        # Step 3: Extract complete JSON objects from the text
        complete_objects = self._extract_complete_objects(text)
        
        if not complete_objects:
            logger.warning("Could not extract any complete JSON objects")
            return text
        
        logger.info(f"Extracted {len(complete_objects)} complete objects from truncated response")
        
        # Step 4: Rebuild as JSON array
        if is_array or len(complete_objects) > 1:
            repaired = "[\n" + ",\n".join(complete_objects) + "\n]"
        else:
            repaired = complete_objects[0]
        
        logger.debug(f"JSON repair: original length={len(original_text)}, repaired length={len(repaired)}")
        
        return repaired
    
    def _extract_complete_objects(self, text: str) -> List[str]:
        """
        Extract complete JSON objects from potentially truncated text.
        Uses character-by-character parsing to track bracket depth and string state.
        """
        complete_objects = []
        
        i = 0
        while i < len(text):
            # Find start of next object
            while i < len(text) and text[i] != '{':
                i += 1
            
            if i >= len(text):
                break
            
            # Parse the object
            obj_start = i
            depth = 0
            in_string = False
            escape_next = False
            obj_complete = False
            
            while i < len(text):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\' and in_string:
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    i += 1
                    continue
                
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            # Found complete object
                            obj_text = text[obj_start:i+1]
                            # Validate it's parseable
                            try:
                                json.loads(obj_text)
                                complete_objects.append(obj_text)
                                obj_complete = True
                            except json.JSONDecodeError:
                                # Try to fix common issues within the object
                                fixed_obj = self._fix_object_issues(obj_text)
                                if fixed_obj:
                                    complete_objects.append(fixed_obj)
                                    obj_complete = True
                            i += 1
                            break
                
                i += 1
            
            # If we didn't complete an object, we've hit the truncation point
            if not obj_complete:
                break
        
        return complete_objects
    
    def _fix_object_issues(self, obj_text: str) -> Optional[str]:
        """
        Try to fix common issues within a JSON object that looks complete structurally
        but has minor issues like trailing commas.
        """
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r',\s*}', '}', obj_text)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            return None

    def _validate_and_filter_questions(self, data: Any) -> Dict[str, Any]:
        """
        Validate parsed JSON and filter out incomplete questions.
        """
        required_keys = {'question', 'choices', 'answer'}
        
        # Handle both array and object with "questions" key
        if isinstance(data, dict) and 'questions' in data:
            questions_data = data['questions']
        elif isinstance(data, list):
            questions_data = data
            data = {'questions': questions_data}
        else:
            return data  # Return as-is if not a questions array
        
        # Filter to keep only complete questions
        valid_questions = []
        for q in questions_data:
            if isinstance(q, dict) and required_keys.issubset(q.keys()):
                valid_questions.append(q)
            else:
                missing = required_keys - set(q.keys()) if isinstance(q, dict) else required_keys
                logger.warning(f"Skipping incomplete question (missing: {missing})")
        
        if len(valid_questions) < len(questions_data):
            logger.warning(f"Filtered {len(questions_data) - len(valid_questions)} incomplete questions")
        
        if not valid_questions:
            raise ValueError("No complete questions found in LLM response after filtering")
        
        data['questions'] = valid_questions
        return data

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response, with auto-repair for common issues."""
        original_text = text
        
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end == -1:  # No closing ```, take rest of text
                text = text[start:].strip()
            else:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end == -1:
                text = text[start:].strip()
            else:
                text = text[start:end].strip()

        # First attempt: try parsing as-is
        try:
            result = json.loads(text)
            return self._validate_and_filter_questions(result)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}. Attempting repair...")
        
        # Second attempt: try to repair the JSON
        repaired_text = self._repair_json(text)
        try:
            result = json.loads(repaired_text)
            logger.info("JSON repair successful!")
            return self._validate_and_filter_questions(result)
        except json.JSONDecodeError as e:
            logger.error(f"JSON repair failed: {e}")
            logger.error(f"Original text (first 500 chars): {original_text[:500]}")
            logger.error(f"Repaired text (first 500 chars): {repaired_text[:500]}")
            raise ValueError(f"Failed to parse LLM response as JSON: {e}. The LLM may have returned truncated output.")

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
        """Build prompt for generating a single MCQ question"""
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
        
        # Get language instruction
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])
        
        # Determine instruction and answer format based on num_correct
        if num_correct == 1:
            instruction = SINGLE_CHOICE_INSTRUCTION
            answer_format = SINGLE_ANSWER_FORMAT
        else:
            instruction = MULTIPLE_CHOICE_INSTRUCTION_TEMPLATE.format(num_correct=num_correct)
            answer_format = MULTIPLE_ANSWER_FORMAT
        
        # Build existing questions context to avoid duplicates
        avoid_section = ""
        if existing_questions:
            questions_list = '\n'.join(f'- {q}' for q in existing_questions[-5:])
            avoid_section = AVOID_DUPLICATION_TEMPLATE.format(existing_questions_list=questions_list)
        
        # Get variety hint based on question index
        variety_hint = VARIETY_HINTS[question_index % len(VARIETY_HINTS)]
        
        # Build final prompt using template
        return MCQ_PROMPT_TEMPLATE.format(
            instruction=instruction,
            lang_instruction=lang_instruction,
            question_number=question_index + 1,
            variety_hint=variety_hint,
            avoid_section=avoid_section,
            answer_format=answer_format,
            passage=passage
        )

    def _build_batch_mcq_prompt(self, passage: str, num_questions: int, options: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for generating multiple MCQ questions in one API call"""
        language = "en"
        question_plan = []
        difficulty_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}

        if options:
            language = options.get("language", "en")
            question_plan = options.get("question_plan", [])
            difficulty_counts = options.get("difficulty_counts", {}) or {}
            type_counts = options.get("type_counts", {}) or {}

        lang_instruction = LANGUAGE_INSTRUCTIONS_ALL.get(language, LANGUAGE_INSTRUCTIONS_ALL["en"])

        if question_plan:
            plan_lines = "\n".join([
                f"- Q{item.get('index', idx + 1)}: difficulty {item.get('difficulty', 'medium')} | type {item.get('type', 'single_choice')}"
                for idx, item in enumerate(question_plan)
            ])
        else:
            plan_lines = (
                "- Distribute questions according to the counts below and keep the order stable.\n"
                "- If no plan is given, alternate single_choice and multiple_choice while respecting difficulty counts."
            )

        difficulty_summary = (
            f"- easy: {difficulty_counts.get('easy', 0)}\n"
            f"- medium: {difficulty_counts.get('medium', 0)}\n"
            f"- hard: {difficulty_counts.get('hard', 0)}\n"
            f"- single_choice: {type_counts.get('single', 0)} | multiple_choice: {type_counts.get('multiple', 0)}"
        )

        return BATCH_MCQ_PROMPT_TEMPLATE.format(
            num_questions=num_questions,
            question_plan_text=plan_lines,
            difficulty_summary=difficulty_summary,
            lang_instruction=lang_instruction,
            passage=passage,
        )

    def _build_distractor_prompt(self, passage: str, correct_answer: str, candidates: List[str]) -> str:
        """Build prompt for refining distractor options"""
        candidates_json = ','.join(f'"{c}"' for c in candidates[:6])
        return DISTRACTOR_PROMPT_TEMPLATE.format(
            passage=passage,
            correct_answer=correct_answer,
            candidates_json=candidates_json
        )

    def _build_short_answer_prompt(self, passage: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for generating a short answer question"""
        language = "en"
        if options:
            language = options.get("language", "en")
        
        lang_instruction = LANGUAGE_INSTRUCTIONS_SHORT.get(language, LANGUAGE_INSTRUCTIONS_SHORT["en"])
        
        return SHORT_ANSWER_PROMPT_TEMPLATE.format(
            lang_instruction=lang_instruction,
            passage=passage
        )

    def _build_true_false_prompt(self, passage: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for generating a true/false question"""
        language = "en"
        if options:
            language = options.get("language", "en")
        
        lang_instruction = LANGUAGE_INSTRUCTIONS_TF.get(language, LANGUAGE_INSTRUCTIONS_TF["en"])
        
        return TRUE_FALSE_PROMPT_TEMPLATE.format(
            lang_instruction=lang_instruction,
            passage=passage
        )
