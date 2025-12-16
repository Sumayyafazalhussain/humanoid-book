import sys
print("Python path:", sys.path)
print("\nTrying imports...")

try:
    from services.ingestion_service import ingestion_service
    print("✅ ingestion_service imported")
except Exception as e:
    print(f"❌ ingestion_service error: {e}")

try:
    from services.rag_service import rag_service
    print("✅ rag_service imported")
except Exception as e:
    print(f"❌ rag_service error: {e}")

try:
    from qdrant.qdrant_client import qdrant_store
    print("✅ qdrant_store imported")
except Exception as e:
    print(f"❌ qdrant_store error: {e}")

try:
    from utils.config import settings
    print("✅ settings imported")
except Exception as e:
    print(f"❌ settings error: {e}")
