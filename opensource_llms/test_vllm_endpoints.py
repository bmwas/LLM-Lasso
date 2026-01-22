#!/usr/bin/env python3
"""
Test script for vLLM OpenAI-compatible endpoints.

This script tests both the chat completion and embedding endpoints
to verify the Docker services are running correctly.

Usage:
    # From project root (after sourcing .env or setting VLLM_API_KEY)
    python opensource_llms/test_vllm_endpoints.py
    
    # Or with explicit API key
    VLLM_API_KEY=your_key python opensource_llms/test_vllm_endpoints.py
"""

import os
import sys
import json
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# Configuration (can be overridden via environment variables)
CHAT_BASE_URL = os.environ.get("VLLM_CHAT_BASE_URL", "http://localhost:8000/v1")
EMBED_BASE_URL = os.environ.get("VLLM_EMBED_BASE_URL", "http://localhost:8001/v1")
CHAT_MODEL = os.environ.get("VLLM_CHAT_MODEL", "qwen3-thinking")
EMBED_MODEL = os.environ.get("VLLM_EMBED_MODEL", "qwen3-embed")


def get_api_key():
    """Get API key from environment."""
    api_key = os.environ.get("VLLM_API_KEY", "")
    if not api_key:
        print("WARNING: VLLM_API_KEY not set. Trying without authentication...")
        print("         Run: source .env && python opensource_llms/test_vllm_endpoints.py")
    return api_key


def make_request(url, data=None, api_key=None):
    """Make HTTP request to vLLM endpoint."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if data:
        data = json.dumps(data).encode("utf-8")
    
    request = Request(url, data=data, headers=headers)
    
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8")), response.status
    except HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else str(e)
        return {"error": error_body, "status_code": e.code}, e.code
    except URLError as e:
        return {"error": str(e.reason)}, None


def print_separator(title):
    """Print a formatted section separator."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_json(data, indent=2):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def test_health(base_url, service_name):
    """Test the health endpoint."""
    print(f"\n[Testing {service_name} Health]")
    print(f"  URL: {base_url.replace('/v1', '/health')}")
    
    start = time.time()
    try:
        request = Request(base_url.replace("/v1", "/health"))
        with urlopen(request, timeout=10) as response:
            elapsed = time.time() - start
            print(f"  Status: {response.status} OK")
            print(f"  Response Time: {elapsed*1000:.1f}ms")
            return True
    except Exception as e:
        print(f"  Status: FAILED - {e}")
        return False


def test_models(base_url, api_key, service_name):
    """Test the models listing endpoint."""
    print(f"\n[Testing {service_name} Models Endpoint]")
    url = f"{base_url}/models"
    print(f"  URL: {url}")
    
    start = time.time()
    response, status = make_request(url, api_key=api_key)
    elapsed = time.time() - start
    
    if status == 200:
        print(f"  Status: {status} OK")
        print(f"  Response Time: {elapsed*1000:.1f}ms")
        
        if "data" in response:
            for model in response["data"]:
                print(f"\n  Model Details:")
                print(f"    ID: {model.get('id', 'N/A')}")
                print(f"    Root: {model.get('root', 'N/A')}")
                print(f"    Max Length: {model.get('max_model_len', 'N/A')}")
                print(f"    Created: {model.get('created', 'N/A')}")
        return True
    else:
        print(f"  Status: FAILED ({status})")
        print(f"  Error: {response}")
        return False


def test_chat_completion(api_key):
    """Test the chat completion endpoint."""
    print(f"\n[Testing Chat Completion]")
    url = f"{CHAT_BASE_URL}/chat/completions"
    print(f"  URL: {url}")
    print(f"  Model: {CHAT_MODEL}")
    
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2 + 2? Answer in one word."}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    print(f"\n  Request Payload:")
    print(f"    Messages: {payload['messages']}")
    print(f"    Max Tokens: {payload['max_tokens']}")
    print(f"    Temperature: {payload['temperature']}")
    
    print(f"\n  Sending request...")
    start = time.time()
    response, status = make_request(url, data=payload, api_key=api_key)
    elapsed = time.time() - start
    
    if status == 200:
        print(f"  Status: {status} OK")
        print(f"  Response Time: {elapsed*1000:.1f}ms")
        
        if "choices" in response:
            choice = response["choices"][0]
            message = choice.get("message", {})
            print(f"\n  Response Details:")
            print(f"    Role: {message.get('role', 'N/A')}")
            print(f"    Content: {message.get('content', 'N/A')}")
            print(f"    Finish Reason: {choice.get('finish_reason', 'N/A')}")
        
        if "usage" in response:
            usage = response["usage"]
            print(f"\n  Token Usage:")
            print(f"    Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"    Completion Tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"    Total Tokens: {usage.get('total_tokens', 'N/A')}")
        
        return True
    else:
        print(f"  Status: FAILED ({status})")
        print(f"  Error: {response}")
        return False


def test_embeddings(api_key):
    """Test the embeddings endpoint."""
    print(f"\n[Testing Embeddings]")
    url = f"{EMBED_BASE_URL}/embeddings"
    print(f"  URL: {url}")
    print(f"  Model: {EMBED_MODEL}")
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    payload = {
        "model": EMBED_MODEL,
        "input": test_texts
    }
    
    print(f"\n  Request Payload:")
    print(f"    Input Texts: {len(test_texts)} strings")
    for i, text in enumerate(test_texts):
        print(f"      [{i}]: \"{text[:50]}...\"" if len(text) > 50 else f"      [{i}]: \"{text}\"")
    
    print(f"\n  Sending request...")
    start = time.time()
    response, status = make_request(url, data=payload, api_key=api_key)
    elapsed = time.time() - start
    
    if status == 200:
        print(f"  Status: {status} OK")
        print(f"  Response Time: {elapsed*1000:.1f}ms")
        
        if "data" in response:
            print(f"\n  Response Details:")
            print(f"    Number of Embeddings: {len(response['data'])}")
            
            for i, emb in enumerate(response["data"]):
                embedding = emb.get("embedding", [])
                print(f"\n    Embedding [{i}]:")
                print(f"      Dimensions: {len(embedding)}")
                print(f"      First 5 values: {embedding[:5]}")
                print(f"      Last 5 values: {embedding[-5:]}")
                
                # Calculate magnitude
                magnitude = sum(x**2 for x in embedding) ** 0.5
                print(f"      Magnitude: {magnitude:.4f}")
        
        if "usage" in response:
            usage = response["usage"]
            print(f"\n  Token Usage:")
            print(f"    Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"    Total Tokens: {usage.get('total_tokens', 'N/A')}")
        
        return True
    else:
        print(f"  Status: FAILED ({status})")
        print(f"  Error: {response}")
        return False


def main():
    """Run all tests."""
    print_separator("vLLM ENDPOINT TEST SUITE")
    print(f"\nChat Service URL: {CHAT_BASE_URL}")
    print(f"Embed Service URL: {EMBED_BASE_URL}")
    
    api_key = get_api_key()
    if api_key:
        print(f"API Key: {'*' * 8}...{api_key[-4:] if len(api_key) > 4 else '****'}")
    
    results = {}
    
    # Test Health Endpoints
    print_separator("HEALTH CHECKS")
    results["chat_health"] = test_health(CHAT_BASE_URL, "Chat Service")
    results["embed_health"] = test_health(EMBED_BASE_URL, "Embedding Service")
    
    # Test Models Endpoints
    print_separator("MODELS ENDPOINTS")
    results["chat_models"] = test_models(CHAT_BASE_URL, api_key, "Chat Service")
    results["embed_models"] = test_models(EMBED_BASE_URL, api_key, "Embedding Service")
    
    # Test Chat Completion
    print_separator("CHAT COMPLETION TEST")
    results["chat_completion"] = test_chat_completion(api_key)
    
    # Test Embeddings
    print_separator("EMBEDDINGS TEST")
    results["embeddings"] = test_embeddings(api_key)
    
    # Summary
    print_separator("TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"\n  Total Tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    print(f"\n  Individual Results:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"    [{symbol}] {test_name}: {status}")
    
    print("\n" + "=" * 70)
    
    if failed > 0:
        print("\n  Some tests failed. Check the output above for details.")
        print("  Common issues:")
        print("    - Services not running: docker compose --env-file .env -f opensource_llms/docker-compose.yml up")
        print("    - Wrong API key: source .env && echo $VLLM_API_KEY")
        print("    - Models still loading: check logs with docker compose logs -f")
        sys.exit(1)
    else:
        print("\n  All tests passed! vLLM services are working correctly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
