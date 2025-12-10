@echo off
REM Start script for quiz-service on Windows

echo Starting Quiz Generation Service...

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found!
    echo Please copy .env.example to .env and configure your API keys.
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the application
echo Starting FastAPI server...
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
