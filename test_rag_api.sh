#!/bin/bash

echo "Ì∫Ä Testing RAG API"
echo "=================="

BASE_URL="http://localhost:8000"

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    echo ""
    echo "Ì¥ç Testing $method $endpoint"
    echo "------------------------"
    
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        response=$(curl -s -X POST "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    else
        response=$(curl -s -X GET "$BASE_URL$endpoint")
    fi
    
    echo "Response:"
    echo "$response" | python -m json.tool 2>/dev/null || echo "$response"
}

# 1. Health check
test_endpoint "GET" "/health"

# 2. Root
test_endpoint "GET" "/"

# 3. Ingest (comment out after first run)
echo ""
read -p "‚ùì Run ingestion? This may take 2-3 minutes. (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    test_endpoint "POST" "/ingest" "{}"
    echo "‚è≥ Waiting 10 seconds for ingestion to complete..."
    sleep 10
fi

# 4. Test RAG query
test_endpoint "POST" "/query" '{"query": "What is Physical AI?"}'

# 5. Test selected text
test_endpoint "POST" "/query_selected_text" '{
    "query": "What is embodied intelligence?",
    "selected_text": "Physical AI represents embodied intelligence that operates in physical space."
}'

echo ""
echo "Ìæâ All tests completed!"
