from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class QuestionTypeEnum(int, Enum):
    SINGLE_CHOICE = 0
    MULTIPLE_CHOICE = 1


class DifficultyEnum(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PointStrategyEnum(str, Enum):
    EQUAL = "equal"
    DIFFICULTY_WEIGHTED = "difficulty_weighted"


class DifficultyDistribution(BaseModel):
    easy: int = Field(default=3, ge=0, description="Số câu hỏi dễ")
    medium: int = Field(default=4, ge=0, description="Số câu hỏi trung bình")
    hard: int = Field(default=3, ge=0, description="Số câu hỏi khó")

    def total(self) -> int:
        return self.easy + self.medium + self.hard


class QuestionTypeDistribution(BaseModel):
    single: int = Field(default=5, ge=0, description="Số câu hỏi single choice")
    multiple: int = Field(default=5, ge=0, description="Số câu hỏi multiple choice")

    def total(self) -> int:
        return self.single + self.multiple


class QuizOption(BaseModel):
    id: Optional[int] = 0
    optionText: str = Field(..., min_length=1, description="Option text content")
    isCorrect: bool = Field(..., description="Whether this option is correct")


class QuizQuestion(BaseModel):
    id: Optional[int] = 0
    questionText: str = Field(..., min_length=1, description="Question text")
    questionType: QuestionTypeEnum = Field(..., description="Type of question")
    difficulty: DifficultyEnum = Field(..., description="Difficulty level of the question")
    point: float = Field(default=1.0, ge=0, description="Points for this question")
    options: List[QuizOption] = Field(..., min_items=2, description="Answer options")


class GenerateFromPromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt to generate quiz from")
    difficulty_distribution: DifficultyDistribution = Field(
        default_factory=DifficultyDistribution,
        description="Số câu theo từng độ khó"
    )
    question_type_distribution: QuestionTypeDistribution = Field(
        default_factory=QuestionTypeDistribution,
        description="Số câu theo loại (single/multiple)"
    )
    language: str = Field(default="vi", description="Language for questions (e.g., 'vi', 'en')")
    total_points: float = Field(default=10.0, gt=0, description="Tổng điểm của bộ câu hỏi")
    point_strategy: PointStrategyEnum = Field(
        default=PointStrategyEnum.EQUAL,
        description="Cách chia điểm: chia đều hoặc ưu tiên câu dễ"
    )


class GenerateFromFileParams(BaseModel):
    difficulty_distribution: Optional[DifficultyDistribution] = Field(
        default=None, description="Số câu theo từng độ khó"
    )
    question_type_distribution: Optional[QuestionTypeDistribution] = Field(
        default=None, description="Số câu theo loại (single/multiple)"
    )
    language: Optional[str] = Field(
        default="vi", description="Language for questions (e.g., 'vi', 'en')"
    )
    prompt: Optional[str] = Field(
        default=None, description="Additional prompt for quiz generation"
    )
    total_points: float = Field(default=10.0, gt=0, description="Tổng điểm của bộ câu hỏi")
    point_strategy: PointStrategyEnum = Field(
        default=PointStrategyEnum.EQUAL,
        description="Cách chia điểm: chia đều hoặc ưu tiên câu dễ"
    )


class QuizMetadata(BaseModel):
    difficulty_distribution: DifficultyDistribution
    question_type_distribution: QuestionTypeDistribution
    language: str
    source: str
    total_points: float
    point_strategy: PointStrategyEnum
    prompt: Optional[str] = None
    file_name: Optional[str] = None


class QueryRelevanceInfo(BaseModel):
    """Thông tin về độ liên quan của query với document."""
    is_relevant: bool = Field(..., description="Query có liên quan đến document không")
    relevance_score: float = Field(..., ge=0, le=1, description="Điểm liên quan (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Độ tin cậy của đánh giá (0-1)")
    strategy_used: str = Field(..., description="Chiến lược được sử dụng: 'search', 'hybrid', hoặc 'representative'")
    warning_message: Optional[str] = Field(None, description="Cảnh báo cho người dùng (nếu query không/ít liên quan)")
    details: Optional[Dict[str, Any]] = Field(None, description="Chi tiết phân tích (scores, thresholds, etc.)")


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
    query_relevance: Optional[QueryRelevanceInfo] = Field(
        None, 
        description="Thông tin về độ liên quan của prompt/query với document. Chứa cảnh báo nếu query không phù hợp."
    )


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error: Optional[str] = None
