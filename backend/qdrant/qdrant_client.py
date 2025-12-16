from qdrant_client import QdrantClient, models
from utils.config import settings

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30.0  # Increased timeout for large operations
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.EMBEDDING_DIMENSION  # ✅ Now 768 for Flash
        
        print(f"✅ Initialized Qdrant Store")
        print(f"   Collection: {self.collection_name}")
        print(f"   Vector Size: {self.vector_size}")

    def create_collection_if_not_exists(self):
        """Create collection only if it doesn't exist"""
        try:
            # Check if collection exists
            existing = self.client.get_collection(collection_name=self.collection_name)
            print(f"📦 Collection already exists: {self.collection_name}")
            return False
        except:
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, 
                    distance=models.Distance.COSINE
                ),
            )
            print(f"✅ Created new collection: {self.collection_name}")
            return True

    def recreate_collection(self):
        """Delete and recreate collection (for fresh ingestion)"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"🗑️  Deleted existing collection: {self.collection_name}")
        except Exception as e:
            print(f"ℹ️  No existing collection to delete: {e}")
            
        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size, 
                distance=models.Distance.COSINE
            ),
        )
        print(f"✅ Created new collection: {self.collection_name}")
        return True

    def upsert_vectors(self, ids, vectors, payloads, batch_size=100):
        """Store vectors in batches"""
        total = len(ids)
        print(f"📤 Uploading {total} vectors in batches of {batch_size}...")
        
        for i in range(0, total, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    payloads=batch_payloads
                )
            )
            
            progress = min(i + batch_size, total)
            print(f"   📊 Progress: {progress}/{total} vectors uploaded")
        
        print(f"✅ Successfully uploaded {total} vectors")

    def search_vectors(self, query_vector, limit=5, score_threshold=0.7):
        """Search similar vectors with score threshold"""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,  # Filter low similarity results
            with_payload=True,
            with_vectors=False
        )
        
        # Filter results above threshold
        filtered_results = [hit for hit in search_result if hit.score >= score_threshold]
        
        print(f"🔍 Found {len(filtered_results)} relevant chunks (threshold: {score_threshold})")
        return filtered_results

    def get_collection_info(self):
        """Get collection statistics"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "status": "active"
            }
        except Exception as e:
            return {"error": str(e), "status": "not_found"}

qdrant_store = QdrantStore()