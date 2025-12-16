@echo off
echo Fixing imports for Windows...
echo.

REM Create __init__.py files
echo Creating __init__.py files...
type nul > backend\__init__.py
type nul > backend\api\__init__.py
type nul > backend\services\__init__.py
type nul > backend\qdrant\__init__.py
type nul > backend\utils\__init__.py

REM Fix main.py imports
echo Fixing main.py...
(
echo from fastapi import FastAPI
echo from fastapi.middleware.cors import CORSMiddleware
echo from pydantic import BaseModel
echo.
echo from services.ingestion_service import ingestion_service
echo from services.rag_service import rag_service
echo.
echo app = FastAPI^(
echo     title="Physical AI Book RAG API",
echo     description="RAG Chatbot for Physical AI ^& Humanoid Robotics Textbook",
echo     version="1.0.0"
echo ^)
echo.
echo REM Rest of your main.py code...
) > backend\api\main_temp.py

REM Copy original content after imports
findstr /v "from.*import" backend\api\main.py >> backend\api\main_temp.py
move /Y backend\api\main_temp.py backend\api\main.py

echo.
echo ✅ Imports fixed! Try running again.
