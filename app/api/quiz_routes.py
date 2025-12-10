from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from app.core.config import get_settings
from app.models.quiz import (
    ErrorResponse,
    GenerateFromFileResponse,
    GenerateFromPromptRequest,
    GenerateFromPromptResponse,
    QuizMetadata,
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
        logger.info(f"Received request to generate {request.num_questions} questions from prompt")

        # Initialize service
        service = QuizGenerationService()

        # Generate questions
        questions = await service.generate_from_prompt(
            prompt=request.prompt,
            num_questions=request.num_questions,
            difficulty=request.difficulty.value,
            language=request.language,
            question_types=request.question_types,
        )

        # Build metadata
        metadata = QuizMetadata(
            difficulty=request.difficulty.value,
            language=request.language,
            source="prompt",
            prompt=request.prompt,
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
    num_questions: int = Form(default=10),
    difficulty: str = Form(default="medium"),
    language: str = Form(default="vi"),
    question_types: str = Form(default="0,1,2"),
    prompt: str = Form(default=None),
):
    temp_file_path = None

    try:
        logger.info(
            f"Received file upload: {file.filename} "
            f"(type: {file.content_type}, num_questions: {num_questions})"
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

        # Parse question types
        try:
            question_type_list = [int(t.strip()) for t in question_types.split(",")]
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid question_types format"
            )

        # Validate parameters
        if num_questions < 1 or num_questions > settings.MAX_NUM_QUESTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"num_questions must be between 1 and {settings.MAX_NUM_QUESTIONS}",
            )

        if difficulty not in ["easy", "medium", "hard"]:
            raise HTTPException(
                status_code=400, detail="difficulty must be easy, medium, or hard"
            )

        # Initialize service
        service = QuizGenerationService()

        # Save uploaded file temporarily
        temp_file_path = await service.save_uploaded_file(file_content, file.filename)

        # Generate questions from file using full pipeline
        questions, file_name, parsed_text, chunks, pipeline_metadata = await service.generate_from_file_advanced(
            file_path=temp_file_path,
            num_questions=num_questions,
            difficulty=difficulty,
            language=language,
            question_types=question_type_list,
            additional_prompt=prompt,
        )
        logger.info(f"Pipeline metadata: {pipeline_metadata}")

        # Build metadata
        metadata = QuizMetadata(
            difficulty=difficulty,
            language=language,
            source="file",
            file_name=file_name,
            prompt=prompt,
        )

        response = GenerateFromFileResponse(
            success=True,
            message=f"Successfully generated {len(questions)} questions from file",
            questions=questions,
            total_questions=len(questions),
            metadata=metadata,
            parsed_text=parsed_text,
            chunks=chunks,
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
