import os
from typing import List, Dict
import time
from .gemini_service import gemini_service
from qdrant.qdrant_client import qdrant_store
from utils.config import settings
from api.gemini_handler import GeminiChatbot

class RAGService:
    def __init__(self):
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
        self.max_results = int(os.getenv("MAX_RESULTS", 5))
        self.gemini_chatbot = GeminiChatbot()
    
    async def query_rag(self, query_text: str) -> Dict:
        """Query RAG system with full book context using Gemini Flash"""
        print(f"🔍 Querying: {query_text[:50]}...")
        start_time = time.time()
        
        try:
            # 1. Generate embedding for the query
            query_embedding = await gemini_service.get_embedding(query_text)
            embedding_time = time.time()
            print(f"✅ Query embedding generated in {embedding_time - start_time:.2f}s")
            
            # 2. Search Qdrant for relevant chunks
            search_results = qdrant_store.search_vectors(
                query_embedding, 
                limit=self.max_results,
                score_threshold=self.similarity_threshold
            )
            search_time = time.time()
            print(f"✅ Vector search completed in {search_time - embedding_time:.2f}s")
            
            if not search_results:
                return {
                    "answer": "I couldn't find relevant information in the textbook.",
                    "sources": [],
                    "chunks_used": 0,
                    "query": query_text,
                    "response_time": time.time() - start_time
                }
            
            # 3. Build context from relevant chunks
            context_chunks = []
            source_attribution = []
            
            for i, hit in enumerate(search_results):
                if "content" in hit.payload:
                    chunk_content = hit.payload["content"]
                    # Extract chapter name from filename (e.g., "docs/01-intro.md" -> "01-intro")
                    filename = hit.payload.get("filename", f"chunk_{i}")
                    chapter_name = os.path.basename(filename).replace(".md", "")
                    
                    # Add chunk with metadata and chapter reference
                    context_chunks.append(f"### [Chapter: {chapter_name}, Score: {hit.score:.3f}]\n{chunk_content}")
                    source_attribution.append(chapter_name)
            
            context = "\n\n---\n\n".join(context_chunks)
            print(f"📄 Built context from {len(context_chunks)} chunks")
            
            # 4. Generate answer using Gemini Flash with the structured prompt
            prompt = f"""
            ## ROLE: Physical AI & Humanoid Robotics Expert Assistant
            ## TASK: Answer questions STRICTLY using ONLY the textbook content provided in the "TEXTBOOK EXCERPTS" section.

            ## TEXTBOOK EXCERPTS:
            {context}

            ## STUDENT QUESTION:
            {query_text}

            ## INSTRUCTIONS (CRITICAL - MUST FOLLOW):
            1.  **ANSWER FROM CONTEXT ONLY**: Use only information directly present in the "TEXTBOOK EXCERPTS" above.
            2.  **IF ANSWER EXISTS**: Provide a comprehensive and detailed explanation, using specific examples or details from the provided context. Structure your answer clearly.
            3.  **IF PARTIAL ANSWER**: If the excerpts contain some relevant information but not a complete answer, clearly state what *is* covered and what *is not* covered in the context.
            4.  **IF NO ANSWER**: If the information required to answer the question is NOT in the "TEXTBOOK EXCERPTS", you MUST respond with the exact phrase: "This topic is not covered in the available textbook chapters. Please refer to the textbook's table of contents or search for related topics."
            5.  **SOURCE ATTRIBUTION**: For every piece of information, cite the chapter name it came from, e.g., "(Chapter: 01-intro)".
            6.  **FORMATTING**:
                *   Start your answer directly without preamble.
                *   Use clear, concise, and technical language appropriate for a textbook assistant.
                *   Employ markdown formatting (e.g., bullet points, bolding) for readability where appropriate.
                *   Explain technical terms if they are central to the answer.
                *   Keep the response between 50 and 300 words.

            7.  **DO NOT**:
                *   Do not say "according to the context" or "based on the text".
                *   Do not mention that you are an AI or language model.
                *   Do not generate information not found in the provided excerpts.
                *   Do not give generic or vague answers.
                *   Do not use external knowledge.

            ## ASSISTANT'S ANSWER:
            """
            
            print("🤖 Generating answer with Gemini Flash...")
            answer = await self.gemini_chatbot.generate_response(prompt)
            generation_time = time.time()
            print(f"✅ Answer generated in {generation_time - search_time:.2f}s")
            
            total_time = time.time() - start_time
            
            return {
                "answer": answer,
                "sources": list(set(source_attribution)),
                "chunks_used": len(context_chunks),
                "similarity_scores": [hit.score for hit in search_results],
                "query": query_text,
                "response_time": total_time,
                "model": settings.GEMINI_MODEL
            }
            
        except Exception as e:
            print(f"❌ RAG query error: {e}")
            return {
                "answer": self.gemini_chatbot.get_fallback_response("generic"),
                "sources": [],
                "error": str(e),
                "query": query_text
            }

    async def query_selected_text(self, query_text: str, selected_text: str) -> Dict:
        """Query based only on selected text using Gemini Flash"""
        print(f"🔍 Querying selected text: {query_text[:50]}...")
        start_time = time.time()
        
        try:
            prompt = f"""
            ## ROLE: Information Extraction Assistant
            ## TASK: Answer the student's question STRICTLY using ONLY the provided "SELECTED TEXT".

            ## SELECTED TEXT:
            {selected_text}

            ## STUDENT QUESTION:
            {query_text}

            ## INSTRUCTIONS (CRITICAL - MUST FOLLOW):
            1.  **ANSWER FROM SELECTED TEXT ONLY**: Use only information directly present in the "SELECTED TEXT" above.
            2.  **IF ANSWER EXISTS**: Provide a concise and direct answer.
            3.  **IF NO ANSWER**: If the information required to answer the question is NOT in the "SELECTED TEXT", you MUST respond with the exact phrase: "The selected text does not contain this information."
            4.  **FORMATTING**:
                *   Start your answer directly without preamble.
                *   Keep the response under 100 words.

            5.  **DO NOT**:
                *   Do not say "according to the selected text" or "based on the text".
                *   Do not mention that you are an AI or language model.
                *   Do not generate information not found in the provided selected text.
                *   Do not use external knowledge.

            ## ASSISTANT'S ANSWER:
            """
            
            answer = await self.gemini_chatbot.generate_response(prompt)
            
            return {
                "answer": answer,
                "source": "User-selected text",
                "query": query_text,
                "selected_text_length": len(selected_text),
                "response_time": time.time() - start_time,
                "model": settings.GEMINI_MODEL
            }
            
        except Exception as e:
            print(f"❌ Selected text query error: {e}")
            return {
                "answer": self.gemini_chatbot.get_fallback_response("generic"),
                "source": "Error",
                "query": query_text,
                "error": str(e)
            }

rag_service = RAGService()