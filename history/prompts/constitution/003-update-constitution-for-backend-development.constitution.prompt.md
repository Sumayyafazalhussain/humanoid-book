---
id: 003
title: Update Constitution for Backend Development
stage: constitution
date: 2025-12-14
surface: agent
model: gemini-1.5-flash-latest
feature: none
branch: main
user: HP
command: sp.update_constitution module2_backend_constitution.md
labels: ["constitution", "backend"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
- GEMINI.md
tests:

---

## Prompt

sp.update_constitution module2_backend_constitution.md# MODULE 2: BACKEND DEVELOPMENT CONSTITUTION
## Physical AI & Humanoid Robotics Textbook - RAG Chatbot System

### 🎯 PROJECT OVERVIEW
We are building a comprehensive RAG (Retrieval-Augmented Generation) chatbot system for the "Physical AI & Humanoid Robotics" textbook. The system must:
1. Answer questions about textbook content using RAG
2. Support queries based on user-selected text only
3. Use Gemini AI for all LLM operations
4. Use MongoDB Atlas for all data storage (No Qdrant, No PostgreSQL)
5. Implement user authentication with background collection
6. Provide content personalization and Urdu translation

### 🛠️ TECHNOLOGY STACK
- **LLM**: Google Gemini API (gemini-1.5-pro for chat, embedding-001 for embeddings)
- **Database**: MongoDB Atlas (with vector search capabilities)
- **Backend Framework**: FastAPI with async/await
- **Authentication**: BetterAuth with MongoDB adapter
- **Vector Storage**: MongoDB Atlas Vector Search (no separate vector DB)
- **Frontend Integration**: React components in Docusaurus

### 📁 PROJECT STRUCTURE
/physical-ai-book/
├── book_source/          # Docusaurus book (already created)
│   └── docs/
└── backend/              # Backend system (to be created)
    ├── api/              # FastAPI routers
    ├── models/           # Pydantic models
    ├── services/         # Business logic
    ├── database/         # MongoDB connections
    ├── auth/             # Authentication
    ├── utils/            # Utilities
    └── tests/            # Test files

### 🔐 AUTHENTICATION REQUIREMENTS
- User signup with email/password
- Collect user background during signup:
  - Software experience (text field)
  - Hardware experience (text field)
  - Proficiency level (beginner/intermediate/advanced)
- JWT-based authentication
- Session management

### 🤖 RAG CHATBOT REQUIREMENTS
1. **Full RAG Mode**: Query entire textbook content
2. **Selected Text Mode**: Query only user-selected text
3. **Chapter Context**: Support chapter-specific queries
4. **Source Attribution**: Show which chapters/sections were referenced
5. **Context Window**: Support large textbook content (100k+ words)

### 📊 MONGODB COLLECTIONS DESIGN
1. `users` - User accounts with background info
2. `book_chunks` - Textbook content chunks with embeddings
3. `chat_sessions` - Chat history with context
4. `user_preferences` - Personalization settings
5. `system_metrics` - Usage analytics

### 🚀 DEPLOYMENT REQUIREMENTS
- Backend deployable on Vercel/Railway/Render
- MongoDB Atlas for production
- Environment-based configuration
- Health check endpoints
- Comprehensive logging

### ✅ ACCEPTANCE CRITERIA
1. Chatbot answers questions from textbook content
2. Selected text queries work independently
3. User authentication with background storage
4. Content personalization based on user background
5. Urdu translation via Gemini API
6. All endpoints properly documented
7. Error handling and validation
8. Performance optimized for concurrent users

## Response snapshot

Successfully overwrote file: C:\Users\HP\OneDrive\Desktop\h1\GEMINI.md.

## Outcome

- ✅ Impact: Constitution document updated.
- 🧪 Tests: N/A
- 📁 Files: Modified GEMINI.md
- 🔁 Next prompts: Waiting for user's next command.
- 🧠 Reflection: Successfully updated the constitution based on user input.

## Evaluation notes (flywheel)

- Failure modes observed:
- Graders run and results (PASS/FAIL):
- Prompt variant (if applicable):
- Next experiment (smallest change to try):
