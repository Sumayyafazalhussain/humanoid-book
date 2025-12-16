import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Gemini 2.5 Flash Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # Updated for Flash
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")  # Best for Flash
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "book_chunks")
    
    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Gemini Flash Embedding Dimension
    # text-embedding-004: 768 dimensions (default for Flash)
    # text-embedding-005: 768 dimensions (latest)
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 768))
    
    # Generation Settings
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 8192))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.3))
    
    # Application
    APP_ENVIRONMENT: str = os.getenv("APP_ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

settings = Settings()