from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ✅ CORRECT IMPORTS - no ".."
try:
    from services.ingestion_service import ingestion_service
    from services.rag_service import rag_service
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Create mock objects for testing
    class MockService:
        async def ingest_book_content(self):
            return {"message": "Mock service"}
        async def query_rag(self, query):
            return {"answer": "Mock answer"}
        async def query_selected_text(self, query, text):
            return {"answer": "Mock answer"}
    
    ingestion_service = MockService()
    rag_service = MockService()

app = FastAPI(
    title="Physical AI Book RAG API",
    description="RAG Chatbot for Physical AI & Humanoid Robotics Textbook",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class SelectedTextQueryRequest(BaseModel):
    query: str
    selected_text: str

@app.get("/")
async def root():
    return {
        "message": "Physical AI Book RAG API",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest (POST)",
            "query": "/query (POST)",
            "query_selected_text": "/query_selected_text (POST)"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG API"}

@app.post("/ingest")
async def ingest_content():
    """Ingest book content into Qdrant"""
    try:
        result = await ingestion_service.ingest_book_content()
        return {
            "message": "Ingestion completed successfully",
            "data": result
        }
    except Exception as e:
        return {"error": str(e), "message": "Ingestion failed"}

@app.post("/query")
async def query_rag_system(request: QueryRequest):
    """Query RAG system with book content"""
    try:
        response = await rag_service.query_rag(request.query)
        return response
    except Exception as e:
        return {"error": str(e), "message": "Query failed"}

@app.post("/query_selected_text")
async def query_with_selected_text(request: SelectedTextQueryRequest):
    """Query based only on selected text"""
    try:
        response = await rag_service.query_selected_text(
            request.query, 
            request.selected_text
        )
        return response
    except Exception as e:
        return {"error": str(e), "message": "Query failed"}

@app.get("/test")
async def test_endpoint():
    return {"test": "API is working", "path": __file__}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
