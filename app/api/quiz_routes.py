import json
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import ValidationError

from app.core.config import get_settings
from app.models.quiz import (
    ErrorResponse,
    GenerateFromFileResponse,
    GenerateFromPromptRequest,
    GenerateFromPromptResponse,
    DifficultyDistribution,
    QuestionTypeDistribution,
    PointStrategyEnum,
    QuizMetadata,
    QueryRelevanceInfo,
)
from app.services.quiz_service import QuizGenerationService
from app.utils.file_utils import cleanup_temp_file, validate_file_extension

router = APIRouter(prefix="/quiz", tags=["Quiz Generation"])
settings = get_settings()


@router.post(
    "/generate-from-prompt",
    response_model=GenerateFromPromptResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def generate_from_prompt(request: GenerateFromPromptRequest):
    try:
        total_questions = request.difficulty_distribution.total()
        logger.info(
            f"Received request to generate {total_questions} questions from prompt with plan: "
            f"{request.difficulty_distribution.dict()} | {request.question_type_distribution.dict()}"
        )

        if total_questions < 1:
            raise HTTPException(status_code=400, detail="Tổng số câu hỏi phải lớn hơn 0")
        if total_questions > settings.MAX_NUM_QUESTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Tổng số câu hỏi không được vượt quá {settings.MAX_NUM_QUESTIONS}",
            )
        if request.question_type_distribution.total() != total_questions:
            raise HTTPException(
                status_code=400,
                detail="Số câu theo loại (single/multiple) phải bằng tổng số câu theo độ khó",
            )
        if request.total_points <= 0:
            raise HTTPException(status_code=400, detail="total_points must be greater than 0")

        # Initialize service
        service = QuizGenerationService()

        # Generate questions
        questions = await service.generate_from_prompt(
            prompt=request.prompt,
            language=request.language,
            difficulty_distribution=request.difficulty_distribution,
            question_type_distribution=request.question_type_distribution,
            total_points=request.total_points,
            point_strategy=request.point_strategy,
        )

        # Build metadata
        metadata = QuizMetadata(
            difficulty_distribution=request.difficulty_distribution,
            question_type_distribution=request.question_type_distribution,
            language=request.language,
            source="prompt",
            prompt=request.prompt,
            total_points=request.total_points,
            point_strategy=request.point_strategy,
        )

        response = GenerateFromPromptResponse(
            success=True,
            message=f"Successfully generated {len(questions)} questions",
            questions=questions,
            total_questions=len(questions),
            metadata=metadata,
        )

        logger.info(f"Successfully generated {len(questions)} questions from prompt")

        return response

    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Validation/business error: {error_msg}")
        
        # Check if it's a rate limit error
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate Limit Exceeded",
                    "message": error_msg,
                    "suggestion": "Please wait a few moments and try again, or check your API quota."
                }
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating quiz from prompt: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate quiz: {str(e)}"
        )


@router.post(
    "/generate-from-file",
    response_model=GenerateFromFileResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def generate_from_file(
    file: UploadFile = File(...),
    difficulty_distribution: str = Form(default='{"easy":3,"medium":4,"hard":3}'),
    question_type_distribution: str = Form(default='{"single":5,"multiple":5}'),
    language: str = Form(default="vi"),
    total_points: float = Form(default=10.0),
    point_strategy: str = Form(default="equal"),
    prompt: str = Form(default=None),
):
    temp_file_path = None

    try:
        logger.info(
            f"Received file upload: {file.filename} "
            f"(type: {file.content_type})"
        )

        # Validate file extension
        if not validate_file_extension(file.filename, settings.allowed_extensions_list):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions_list)}",
            )

        # Read file content
        file_content = await file.read()

        # Validate file size
        if len(file_content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB",
            )

        # Parse distributions
        try:
            difficulty_payload = json.loads(difficulty_distribution)
            type_payload = json.loads(question_type_distribution)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="difficulty_distribution/question_type_distribution must be valid JSON")

        try:
            difficulty_obj = DifficultyDistribution(**difficulty_payload)
            type_obj = QuestionTypeDistribution(**type_payload)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=f"Invalid distribution payload: {e}")

        total_questions = difficulty_obj.total()
        if total_questions < 1:
            raise HTTPException(status_code=400, detail="Tổng số câu hỏi phải lớn hơn 0")
        if total_questions > settings.MAX_NUM_QUESTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Tổng số câu hỏi không được vượt quá {settings.MAX_NUM_QUESTIONS}",
            )
        if type_obj.total() != total_questions:
            raise HTTPException(
                status_code=400,
                detail="Số câu theo loại (single/multiple) phải bằng tổng số câu theo độ khó",
            )

        try:
            point_strategy_enum = PointStrategyEnum(point_strategy)
        except ValueError:
            raise HTTPException(status_code=400, detail="point_strategy must be 'equal' or 'difficulty_weighted'")

        if total_points <= 0:
            raise HTTPException(status_code=400, detail="total_points must be greater than 0")

        # Initialize service
        service = QuizGenerationService()

        # Save uploaded file temporarily
        temp_file_path = await service.save_uploaded_file(file_content, file.filename)

        # Generate questions from file using full pipeline
        questions, file_name, parsed_text, chunks, pipeline_metadata, relevance_metadata = await service.generate_from_file_advanced(
            file_path=temp_file_path,
            difficulty_distribution=difficulty_obj,
            question_type_distribution=type_obj,
            language=language,
            total_points=total_points,
            point_strategy=point_strategy_enum,
            additional_prompt=prompt,
        )
        logger.info(f"Pipeline metadata: {pipeline_metadata}")
        
        # Build query relevance info if available
        query_relevance_info = None
        if relevance_metadata and "query_relevance" in relevance_metadata:
            qr = relevance_metadata["query_relevance"]
            query_relevance_info = QueryRelevanceInfo(
                is_relevant=qr.get("is_relevant", True),
                relevance_score=qr.get("relevance_score", 1.0),
                confidence=qr.get("confidence", 1.0),
                strategy_used=qr.get("strategy_used", "search"),
                warning_message=qr.get("warning_message"),
                details=qr.get("details")
            )
            
            # Log warning if present
            if qr.get("warning_message"):
                logger.warning(f"Query relevance warning: {qr.get('warning_message')}")

        # Build metadata
        metadata = QuizMetadata(
            difficulty_distribution=difficulty_obj,
            question_type_distribution=type_obj,
            language=language,
            source="file",
            file_name=file_name,
            prompt=prompt,
            total_points=total_points,
            point_strategy=point_strategy_enum,
        )

        response = GenerateFromFileResponse(
            success=True,
            message=f"Successfully generated {len(questions)} questions from file",
            questions=questions,
            total_questions=len(questions),
            metadata=metadata,
            parsed_text=parsed_text,
            chunks=chunks,
            query_relevance=query_relevance_info,
        )

        logger.info(f"Successfully generated {len(questions)} questions from file: {file_name}")

        return response

    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Validation/business error: {error_msg}")
        
        # Check if it's a rate limit error
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate Limit Exceeded",
                    "message": error_msg,
                    "suggestion": "Please wait a few moments and try again, or check your API quota."
                }
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error generating quiz from file: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback:\n{error_traceback}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate quiz: {type(e).__name__}: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_file_path:
            cleanup_temp_file(temp_file_path)


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "quiz-generation-service",
        "version": settings.API_VERSION,
    }
