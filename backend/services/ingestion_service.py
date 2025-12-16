# -*- coding: utf-8 -*-
"""Ingestion Service for Physical AI Book RAG"""

import os
import re
from typing import List, Dict

# Try to import dependencies
try:
    from .gemini_service import gemini_service
    from qdrant.qdrant_client import qdrant_store
    from utils.config import settings
    IMPORT_SUCCESS = True
    print("✅ ingestion_service: All imports successful")
except ImportError as e:
    print(f"⚠️ ingestion_service: Some imports failed: {e}")
    IMPORT_SUCCESS = False
    
    # Create mock objects for testing
    class MockGemini:
        async def get_embedding(self, text):
            return [0.0] * 768
    class MockQdrant:
        def recreate_collection(self):
            print("Mock: recreate_collection")
        def upsert_vectors(self, ids, vectors, payloads):
            print(f"Mock: upsert_vectors with {len(ids)} items")
    class MockSettings:
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200
    
    gemini_service = MockGemini()
    qdrant_store = MockQdrant()
    settings = MockSettings()


class IngestionService:
    """Service to ingest book content into Qdrant vector database"""
    
    def __init__(self, book_source_path: str = "../book_source/docs"):
        self.book_source_path = book_source_path
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        print(f"📁 IngestionService initialized with path: {book_source_path}")
        print(f"   Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
    
    def _read_markdown_file(self, file_path: str) -> str:
        """Read markdown file with UTF-8 encoding"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _get_all_markdown_files(self) -> List[str]:
        """Get all markdown files from book source directory"""
        markdown_files = []
        
        if not os.path.exists(self.book_source_path):
            print(f"❌ Directory not found: {self.book_source_path}")
            return markdown_files
        
        for root, dirs, files in os.walk(self.book_source_path):
            for file in files:
                if file.endswith('.md') and not file.startswith('_'):
                    full_path = os.path.join(root, file)
                    markdown_files.append(full_path)
        
        return markdown_files
    
    def _chunk_text(self, text: str, file_path: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        file_name = os.path.basename(file_path)
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Create metadata for the chunk
            metadata = {
                "file_path": file_path,
                "file_name": file_name,
                "content": chunk_text,
                "start_char": start,
                "end_char": end,
                "chunk_size": len(chunk_text)
            }
            
            chunks.append({
                "content": chunk_text,
                "metadata": metadata
            })
            
            # Stop if we've reached the end
            if end >= len(text):
                break
            
            # Move start pointer with overlap
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    async def ingest_book_content(self) -> Dict:
        """Main ingestion function"""
        print("\n🔄 Starting book content ingestion...")
        print("=" * 50)
        
        # Step 1: Get all markdown files
        markdown_files = self._get_all_markdown_files()
        
        if not markdown_files:
            print(f"❌ No markdown files found in: {self.book_source_path}")
            print(f"   Current directory: {os.getcwd()}")
            return {
                "success": False,
                "error": f"No markdown files found in {self.book_source_path}",
                "chunks_processed": 0,
                "files_processed": 0
            }
        
        print(f"📚 Found {len(markdown_files)} markdown files")
        
        # Step 2: Create or recreate Qdrant collection
        try:
            qdrant_store.recreate_collection()
            print("✅ Created/recreated Qdrant collection")
        except Exception as e:
            print(f"⚠️ Could not recreate collection: {e}")
        
        # Step 3: Process each file and create chunks
        all_chunks = []
        for i, file_path in enumerate(markdown_files):
            print(f"\n[{i+1}/{len(markdown_files)}] Processing: {os.path.basename(file_path)}")
            
            try:
                # Read file content
                content = self._read_markdown_file(file_path)
                
                # Split into chunks
                chunks = self._chunk_text(content, file_path)
                all_chunks.extend(chunks)
                
                print(f"   Created {len(chunks)} chunks")
                
            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {e}")
        
        print(f"\n✂️ Total chunks created: {len(all_chunks)}")
        
        if not all_chunks:
            print("❌ No chunks were created")
            return {
                "success": False,
                "error": "No text chunks were created",
                "chunks_processed": 0,
                "files_processed": len(markdown_files)
            }
        
        # Step 4: Generate embeddings and store in Qdrant
        print("\n🧠 Generating embeddings...")
        
        ids = []
        vectors = []
        payloads = []
        
        for i, chunk_data in enumerate(all_chunks):
            # Show progress every 10 chunks
            if i % 10 == 0:
                print(f"   🔢 Processing chunk {i+1}/{len(all_chunks)}")
            
            try:
                # Generate embedding
                embedding = await gemini_service.get_embedding(chunk_data["content"])
                
                # Prepare data for Qdrant
                ids.append(i)
                vectors.append(embedding)
                payloads.append(chunk_data["metadata"])
                
            except Exception as e:
                print(f"   ⚠️ Failed to process chunk {i}: {e}")
        
        print(f"✅ Generated embeddings for {len(vectors)} chunks")
        
        # Step 5: Store in Qdrant
        print("💾 Storing in Qdrant...")
        
        try:
            qdrant_store.upsert_vectors(ids, vectors, payloads)
            print(f"🎉 Successfully ingested {len(all_chunks)} chunks from {len(markdown_files)} files")
            
            return {
                "success": True,
                "chunks_processed": len(all_chunks),
                "files_processed": len(markdown_files),
                "message": "Ingestion completed successfully"
            }
            
        except Exception as e:
            print(f"❌ Failed to store in Qdrant: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "chunks_created": len(all_chunks),
                "files_processed": len(markdown_files),
                "message": "Failed to store in vector database"
            }


# Create global instance
ingestion_service = IngestionService()

# Test code when run directly
if __name__ == "__main__":
    import asyncio
    print("🧪 Testing IngestionService...")
    service = IngestionService()
    result = asyncio.run(service.ingest_book_content())
    print(f"\nResult: {result}")