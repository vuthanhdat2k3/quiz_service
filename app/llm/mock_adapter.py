import json
import hashlib
from typing import List, Dict, Any, Optional
from loguru import logger

from app.llm.base_adapter import (
    LLMAdapter,
    MCQResult,
    BatchMCQResult,
    DistractorResult,
    ShortAnswerResult,
    TrueFalseResult,
)


class MockLLMAdapter(LLMAdapter):

    def __init__(self, **kwargs):
        super().__init__(model_name="mock", **kwargs)
        logger.info("Initialized MockLLMAdapter - no API calls will be made")

    async def generate_mcq(self, passage: str, options: Optional[Dict[str, Any]] = None) -> MCQResult:
        # Generate deterministic content based on passage hash
        passage_hash = hashlib.md5(passage.encode()).hexdigest()[:8]

        question = f"What is the main concept in this passage? (mock-{passage_hash})"
        choices = [
            f"Option A based on passage content",
            f"Option B - alternative interpretation",
            f"Option C - related but incorrect",
            f"Option D - unrelated concept",
        ]
        answer = "A"
        explanation = f"The correct answer is A because the passage discusses this concept directly. (mock generated)"
        difficulty = "medium"

        logger.debug(f"Mock generated MCQ for passage hash {passage_hash}")

        return MCQResult(
            question=question,
            choices=choices,
            answer=answer,
            explanation=explanation,
            difficulty=difficulty,
        )

    async def generate_batch_mcq(
        self, passage: str, num_questions: int, options: Optional[Dict[str, Any]] = None
    ) -> BatchMCQResult:
        """Generate multiple mock MCQ questions"""
        questions = []

        plan = []
        if options:
            plan = options.get("question_plan", [])

        total = len(plan) if plan else num_questions

        for i in range(total):
            passage_hash = hashlib.md5(f"{passage}_{i}".encode()).hexdigest()[:8]

            if plan and i < len(plan):
                difficulty = plan[i].get("difficulty", "medium") if isinstance(plan[i], dict) else str(plan[i])
                q_type_label = plan[i].get("type", "single_choice") if isinstance(plan[i], dict) else "single_choice"
            else:
                difficulty = "medium"
                q_type_label = "single_choice" if i % 2 == 0 else "multiple_choice"

            answer_value = "A" if q_type_label == "single_choice" else ["A", "B"]

            questions.append(MCQResult(
                question=f"Question {i+1}: What concept is discussed? (mock-{passage_hash})",
                choices=[
                    f"Option A for question {i+1}",
                    f"Option B for question {i+1}",
                    f"Option C for question {i+1}",
                    f"Option D for question {i+1}",
                ],
                answer=answer_value,
                explanation=f"Mock explanation for question {i+1}",
                difficulty=difficulty,
            ))

        logger.debug(f"Mock generated {total} batch MCQ questions")
        return BatchMCQResult(questions=questions)

    async def refine_distractors(
        self, passage: str, correct_answer: str, candidates: List[str]
    ) -> DistractorResult:
        # Select first 3 candidates or generate mock ones
        if len(candidates) >= 3:
            distractors = candidates[:3]
        else:
            distractors = candidates + [
                f"Mock distractor {i}" for i in range(1, 4 - len(candidates))
            ]

        logger.debug(f"Mock refined {len(distractors)} distractors")

        return DistractorResult(distractors=distractors[:3])

    async def generate_short_answer(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> ShortAnswerResult:
        passage_hash = hashlib.md5(passage.encode()).hexdigest()[:8]

        question = f"Explain the main concept discussed in the passage. (mock-{passage_hash})"
        answer = "The main concept is about the topic discussed in the passage."
        explanation = "This answer captures the key idea presented."
        difficulty = "medium"

        return ShortAnswerResult(
            question=question, answer=answer, explanation=explanation, difficulty=difficulty
        )

    async def generate_true_false(
        self, passage: str, options: Optional[Dict[str, Any]] = None
    ) -> TrueFalseResult:
        passage_hash = hashlib.md5(passage.encode()).hexdigest()[:8]

        statement = f"The passage discusses a specific concept. (mock-{passage_hash})"
        answer = True
        explanation = "This statement is true based on the passage content."
        difficulty = "easy"

        return TrueFalseResult(
            statement=statement, answer=answer, explanation=explanation, difficulty=difficulty
        )
