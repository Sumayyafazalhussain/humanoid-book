import requests
import json
import time

BASE = "http://localhost:8000"

def test_api():
    print("Ì∑™ Testing RAG API Endpoints")
    print("=" * 50)
    
    # 1. Health
    print("\n1. Health Check:")
    try:
        r = requests.get(f"{BASE}/health")
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.json()}")
    except Exception as e:
        print(f"   ‚ùå Error: {e}")
    
    # 2. Ingest
    print("\n2. Ingest Book Content:")
    try:
        r = requests.post(f"{BASE}/ingest")
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print(f"   ‚úÖ Success: {r.json()}")
        else:
            print(f"   ‚ùå Failed: {r.text}")
    except Exception as e:
        print(f"   ‚ùå Error: {e}")
    
    # Wait if ingestion was successful
    time.sleep(5)
    
    # 3. RAG Query
    print("\n3. RAG Query:")
    try:
        r = requests.post(
            f"{BASE}/query",
            json={"query": "What is robotics?"}
        )
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   ‚úÖ Answer: {data.get('answer', 'No answer')[:100]}...")
            if 'sources' in data:
                print(f"   Ì≥ö Sources: {data['sources']}")
        else:
            print(f"   ‚ùå Failed: {r.text}")
    except Exception as e:
        print(f"   ‚ùå Error: {e}")
    
    print("\n" + "=" * 50)
    print("‚úÖ Test completed! Open http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    test_api()
