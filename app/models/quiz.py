from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionTypeEnum(int, Enum):
    SINGLE_CHOICE = 0
    MULTIPLE_CHOICE = 1
    MIX = 2  # Mix of single and multiple choice questions


class DifficultyEnum(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuizOption(BaseModel):
    id: Optional[int] = 0
    optionText: str = Field(..., min_length=1, description="Option text content")
    isCorrect: bool = Field(..., description="Whether this option is correct")


class QuizQuestion(BaseModel):
    id: Optional[int] = 0
    questionText: str = Field(..., min_length=1, description="Question text")
    questionType: QuestionTypeEnum = Field(..., description="Type of question")
    point: float = Field(default=1.0, ge=0, description="Points for this question")
    options: List[QuizOption] = Field(..., min_items=2, description="Answer options")


class GenerateFromPromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt to generate quiz from")
    num_questions: int = Field(
        default=10, ge=1, le=30, description="Number of questions to generate"
    )
    difficulty: DifficultyEnum = Field(
        default=DifficultyEnum.MEDIUM, description="Difficulty level"
    )
    language: str = Field(default="vi", description="Language for questions (e.g., 'vi', 'en')")
    question_types: List[int] = Field(
        default=[2], description="Question types: 0=Single Choice, 1=Multiple Choice, 2=Mix (default)"
    )


class GenerateFromFileParams(BaseModel):
    num_questions: Optional[int] = Field(
        default=10, ge=1, le=30, description="Number of questions to generate"
    )
    difficulty: Optional[str] = Field(default="medium", description="Difficulty level")
    language: Optional[str] = Field(
        default="vi", description="Language for questions (e.g., 'vi', 'en')"
    )
    question_types: Optional[List[int]] = Field(
        default=[2], description="Question types: 0=Single Choice, 1=Multiple Choice, 2=Mix (default)"
    )
    prompt: Optional[str] = Field(
        default=None, description="Additional prompt for quiz generation"
    )


class QuizMetadata(BaseModel):
    difficulty: str
    language: str
    source: str
    prompt: Optional[str] = None
    file_name: Optional[str] = None


class GenerateFromPromptResponse(BaseModel):
    success: bool = Field(..., description="Whether generation was successful")
    message: str = Field(..., description="Status message")
    questions: List[QuizQuestion] = Field(..., description="Generated questions")
    total_questions: int = Field(..., description="Total number of questions generated")
    metadata: QuizMetadata = Field(..., description="Generation metadata")
    parsed_text: Optional[str] = Field(None, description="Original parsed text")
    chunks: Optional[List[str]] = Field(None, description="Text chunks used for generation")


class GenerateFromFileResponse(BaseModel):
    success: bool = Field(..., description="Whether generation was successful")
    message: str = Field(..., description="Status message")
    questions: List[QuizQuestion] = Field(..., description="Generated questions")
    total_questions: int = Field(..., description="Total number of questions generated")
    metadata: QuizMetadata = Field(..., description="Generation metadata")
    parsed_text: Optional[str] = Field(None, description="Original parsed text from file")
    chunks: Optional[List[str]] = Field(None, description="Text chunks used for generation")


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error: Optional[str] = None
