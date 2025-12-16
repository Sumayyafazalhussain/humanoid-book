import os
import sys
from dotenv import load_dotenv

# Load .env from backend folder
load_dotenv('backend/.env')

print("Ì¥ß Testing API Connections...")
print("="*50)

# Test Gemini
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # List available models
    models = genai.list_models()
    print("‚úÖ Gemini API Connected!")
    print(f"   Available models: {len(list(models))}")
    
    # Check if Flash model is available
    flash_available = any('flash' in model.name.lower() for model in models)
    if flash_available:
        print("   ‚úÖ Flash model available")
    else:
        print("   ‚ö†Ô∏è  Flash model not found, using default")
        
except Exception as e:
    print(f"‚ùå Gemini Error: {e}")

# Test Qdrant
try:
    from qdrant_client import QdrantClient
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url and qdrant_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        # Try to get collections
        collections = client.get_collections()
        print("‚úÖ Qdrant Connected!")
        print(f"   Collections: {len(collections.collections)}")
    else:
        print("‚ö†Ô∏è  Qdrant credentials missing in .env")
        
except Exception as e:
    print(f"‚ùå Qdrant Error: {e}")

# Test Environment Variables
print("\nÌ≥ã Environment Variables Check:")
required_vars = ["GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"   ‚úÖ {var}: Set ({'*' * min(len(value), 10)}...)")
    else:
        print(f"   ‚ùå {var}: Missing")

print("\nÌæâ Connection Test Complete!")
