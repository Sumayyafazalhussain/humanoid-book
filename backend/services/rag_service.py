import os
from typing import List, Dict
import time
from .gemini_service import gemini_service
from qdrant.qdrant_client import qdrant_store
from utils.config import settings

class RAGService:
    def __init__(self):
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
        self.max_results = int(os.getenv("MAX_RESULTS", 5))
    
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
                    filename = hit.payload.get("filename", f"chunk_{i}")
                    
                    # Add chunk with metadata
                    context_chunks.append(f"[Source: {filename}, Score: {hit.score:.3f}]\n{chunk_content}")
                    source_attribution.append(filename)
            
            context = "\n\n---\n\n".join(context_chunks)
            print(f"📄 Built context from {len(context_chunks)} chunks")
            
            # 4. Generate answer using Gemini Flash
            prompt = f"""You are a knowledgeable teaching assistant for the "Physical AI & Humanoid Robotics" textbook.

RELEVANT TEXTBOOK EXCERPTS:
{context}

STUDENT QUESTION: {query_text}

INSTRUCTIONS (CRITICAL):
1. Answer STRICTLY based on the textbook excerpts above
2. If information is NOT in the excerpts, say: "This topic is not covered in the available textbook chapters."
3. Keep answers concise and technical
4. Reference specific chapters/sources when possible
5. Use markdown formatting for readability

ASSISTANT'S ANSWER:"""
            
            print("🤖 Generating answer with Gemini Flash...")
            answer = await gemini_service.generate_content(prompt)
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
                "answer": f"Error processing your query: {str(e)}",
                "sources": [],
                "error": str(e),
                "query": query_text
            }

    async def query_selected_text(self, query_text: str, selected_text: str) -> Dict:
        """Query based only on selected text using Gemini Flash"""
        print(f"🔍 Querying selected text: {query_text[:50]}...")
        start_time = time.time()
        
        try:
            prompt = f"""You are a helpful assistant. Answer based STRICTLY on the following selected text:

SELECTED TEXT:
{selected_text}

QUESTION: {query_text}

RULES:
1. Answer ONLY from the selected text above
2. If answer is NOT in selected text, say: "The selected text does not contain this information."
3. Do not use any external knowledge
4. Be concise and accurate

ANSWER:"""
            
            answer = await gemini_service.generate_content(prompt)
            
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
                "answer": f"Error: {str(e)}",
                "source": "Error",
                "query": query_text
            }

rag_service = RAGService()