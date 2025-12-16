# SIMPLE RAG BACKEND CONSTITUTION
## Only RAG System for Physical AI Book

### 🎯 PROJECT OVERVIEW
We are building ONLY the RAG (Retrieval-Augmented Generation) system for the "Physical AI & Humanoid Robotics" textbook. The system must:
1. Answer questions about textbook content using RAG
2. Support queries based on user-selected text only
3. Use Gemini AI for embeddings and chat
4. Use Qdrant Cloud for vector storage
5. NO authentication, NO personalization, NO translation for now

### 🛠️ TECHNOLOGY STACK
- **LLM**: Google Gemini API (for embeddings and chat)
- **Vector Database**: Qdrant Cloud Free Tier
- **Backend Framework**: FastAPI
- **No Database**: Only Qdrant for vectors
- **No Authentication**: Simple API without auth
- **No Frontend Integration**: Only API endpoints

### 📁 PROJECT STRUCTURE
/physical-ai-book/
├── book_source/          # Docusaurus book (already created)
└── backend/              # Simple RAG backend
    ├── api/              # FastAPI app and routes
    ├── services/         # Gemini and RAG services
    ├── qdrant/           # Qdrant connection
    └── utils/            # Config and helpers

### 🤖 RAG REQUIREMENTS
1. **Ingest**: Load book content into Qdrant
2. **Query**: Search and answer questions
3. **Selected Text**: Answer from selected text only
4. **Source Attribution**: Show which chapters were used

### 🚀 SIMPLE DEPLOYMENT
- FastAPI backend only
- Qdrant Cloud Free Tier
- Environment configuration
- Health check endpoint