<!--
Sync Impact Report:
- Version change: 0.2.0 → 1.0.0
- Modified principles: Complete rewrite.
- Added sections: All sections are new.
- Removed sections: All sections are replaced.
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md
  - ✅ .specify/templates/spec-template.md
  - ✅ .specify/templates/tasks-template.md
- Follow-up TODOs:
  - TODO(RATIFICATION_DATE): Set initial adoption date.
-->
# Constitution: Backend-Focused RAG Chatbot for Physical AI Book

────────────────────────────────────────────────────────────
Project Name: Physical AI Book — Backend RAG Chatbot
Root Directory: /backend
────────────────────────────────────────────────────────────

## Purpose
Develop a backend-only Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book. The backend will handle:

- Document embedding and vector storage in Qdrant using the embedding model "sentence-transformers/all-MiniLM-L6-v2".
- Query processing and retrieval of relevant passages.
- Answer generation using OpenAI Agents SDK with Gemini API key.
- Exposing all APIs consumed by any frontend (ChatKit JS or others).

## Success Criteria
- All book content embedded into Qdrant using "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2".
- FastAPI backend exposes endpoints for query, embedding, and reload.
- OpenAI Agents SDK responds accurately based on retrieved passages.
- Local Python environment initialized with uv init and all dependencies installed.
- Clear folder structure inside backend/ to maintain maintainability.
- System is fully deployable locally, with optional Docker support.

## Out-of-Scope
- Frontend UI implementation (ChatKit JS will connect later as a client).  
- Non-book-related queries.  
- Hosting or deployment of OpenAI models beyond Gemini API.

## Stakeholders
- Students: Query and explore book content via chatbot.  
- Developers: Implement backend, embedding, retrieval, and agent reasoning.  
- Instructors: Validate answer accuracy and ensure book knowledge coverage.

## System Overview
- Backend-only structure. All operations (embedding, retrieval, agent reasoning) live in /backend.
- FastAPI: Exposes REST endpoints.
- Qdrant Cloud Free Tier: Stores vector embeddings.
- OpenAI Agents SDK (Python): Generates answers using Gemini API key.
- Embedding Model: "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2" for semantic vectorization.
- UV: Manages Python environment and dependencies.
- ChatKit Python SDK: Optional for connecting frontend client later.

## Folder & File Structure (backend/)
- backend/
    - main.py → FastAPI server entry point
    - agents_integration.py → OpenAI Agents SDK integration
    - qdrant_client.py → Qdrant connection and collection management
    - embedding.py → Embedding utilities using "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
    - config.py → Environment variables (GEMINI_API_KEY, GEMINI_MODEL, BASE_URL)
    - utils/ → Text chunking, logging, metadata helpers
    - requirements.txt → Python packages (managed with uv init)
    - .env → API keys and configuration
    - docker/ → Optional Dockerfiles for deployment

## Functional Requirements
- FR1: Initialize backend environment using uv init for Python and package management.
- FR2: Embed all book content into Qdrant using "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2".
- FR3: Expose /query endpoint: accepts user query and optional selection context, returns top-k passages + agent answer.
- FR4: Expose /embed endpoint: accepts text input → stores embedding in Qdrant.
- FR5: Expose /reload endpoint: re-embed all documents into Qdrant.
- FR6: Integrate OpenAI Agents SDK for reasoning and generating answers.
- FR7: Return structured JSON responses including answer, retrieved passages, and metadata.

## Non-Functional Requirements
- NFR1: Backend response time <500ms for Qdrant search; OpenAI Agent generation within API SLA.
- NFR2: Secure API access using API keys or tokens.
- NFR3: Python 3.11+, FastAPI 0.100+, and UV-managed environment.
- NFR4: Logging, error handling, and monitoring for reliability.
- NFR5: Docker-ready for cloud or local deployment.

## Risks & Constraints
- RC1: Free Qdrant tier limits storage or queries.
- RC2: Large book size increases embedding time.
- RC3: Latency from Gemini API affects response time.
- RC4: Ensuring embeddings accurately capture technical content semantics.
- RC5: Dependency on Gemini API availability and token limits.
- RC6: Local machine GPU/CPU limits may affect embedding speed.

────────────────────────────────────────────────────────────
Constitution End
────────────────────────────────────────────────────────────

**Version**: 1.0.0 | **Ratified**: TODO(RATIFICATION_DATE): Set initial adoption date. | **Last Amended**: 2025-12-07
