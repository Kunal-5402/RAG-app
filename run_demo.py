"""Demo script to test the RAG pipeline with sample queries."""

import requests
import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_api_endpoint(question, expected_status=None):
    """Test a single question against the API."""
    print(f"\nQ: {question}")
    print("-" * 50)
    
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Answer: {data['answer']}")
            
            if data['citations']:
                print("Citations:")
                for citation in data['citations']:
                    print(f"  - {citation['source']}:{citation['doc_id']}:{citation['chunk_id']}")
            else:
                print("Citations: None")
            
            if expected_status and data['status'] != expected_status:
                print(f"⚠️  Expected status '{expected_status}' but got '{data['status']}'")
            else:
                print("✓ Response received successfully")
                
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running on http://localhost:8000?")
    except requests.exceptions.Timeout:
        print("❌ Request timeout. Server may be processing...")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run demo queries to test the RAG pipeline."""
    
    print("BYD SEAL RAG Pipeline Demo")
    print("=" * 50)
    
    # Check if server is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✓ Server is healthy")
            print(f"  Facts documents: {health_data.get('facts_documents', 'Unknown')}")
            print(f"  External documents: {health_data.get('external_documents', 'Unknown')}")
        else:
            print("⚠️  Server health check failed")
    except:
        print("❌ Cannot connect to server. Please run: python main.py")
        return
    
    # Test cases
    test_cases = [
        # Factual queries that should be answered
        {
            "question": "What is the battery capacity of BYD SEAL?",
            "expected_status": "answered"
        },
        {
            "question": "How fast can the BYD SEAL accelerate?",
            "expected_status": "answered"
        },
        {
            "question": "What color options are available?",
            "expected_status": "answered"
        },
        
        # Pricing queries (should be answered from facts)
        {
            "question": "What is the price of BYD SEAL?",
            "expected_status": "answered"
        },
        {
            "question": "How much does the Premium trim cost?",
            "expected_status": "answered"
        },
        
        # Sensitive queries without facts (should be refused)
        {
            "question": "What's the warranty coverage like?",
            "expected_status": "refused"  # If not in facts
        },
        {
            "question": "Is it available for purchase in my area?",
            "expected_status": "refused"  # Location-specific availability
        },
        
        # General queries that might use external sources
        {
            "question": "What do people think about the ride quality?",
            "expected_status": "answered"  # Might use external if available
        },
        {
            "question": "How does it compare to other EVs?",
            "expected_status": "answered"  # General comparison
        },
        
        # Queries with no relevant information
        {
            "question": "What is the maintenance schedule?",
            "expected_status": "no_data"  # Likely not in either source
        }
    ]
    
    print("\nTesting various query types...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}]")
        test_api_endpoint(
            test_case["question"], 
            test_case.get("expected_status")
        )
        
        # Small delay between requests
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nKey observations to verify:")
    print("• Pricing questions should be answered with facts and citations")
    print("• Sensitive queries without facts should be refused") 
    print("• General queries can use external sources when appropriate")
    print("• All answers should include proper citations")
    print("• No hallucinated information should appear")

if __name__ == "__main__":
    main()
