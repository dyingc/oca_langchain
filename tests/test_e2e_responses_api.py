#!/usr/bin/env python3
"""
End-to-End Tests for OpenAI Response API

This script tests the Response API endpoints against a running server.
Requires the API server to be running on http://127.0.0.1:8450

Usage:
    python tests/test_e2e_responses_api.py
"""

import httpx
import json
import sys

BASE_URL = "http://127.0.0.1:8450/v1/responses"
HEADERS = {
    "Content-Type": "application/json",
}


def test_request(name: str, payload: dict, expected_status: int = 200) -> dict | None:
    """
    Test a request and print the result.

    Args:
        name: Test name
        payload: Request payload
        expected_status: Expected HTTP status code

    Returns:
        Response JSON if successful, None otherwise
    """
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"Payload: {json.dumps(payload, indent=2)[:500]}")

    try:
        response = httpx.post(BASE_URL, json=payload, headers=HEADERS, timeout=60.0)
        print(f"Status: {response.status_code}")

        if response.status_code != expected_status:
            print(f"❌ Expected status {expected_status}, got {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"Response ID: {data.get('id')}")
            print(f"Status: {data.get('status')}")

            # Print output items
            output = data.get("output", [])
            for i, item in enumerate(output):
                item_type = item.get("type", "unknown")
                print(f"  Output[{i}]: {item_type}")
                if item_type == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            text = content.get("text", "")
                            print(f"    Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                elif item_type == "function_call":
                    print(f"    Function: {item.get('name')}")
                    print(f"    Arguments: {item.get('arguments', '')[:50]}...")

            return data
        else:
            print(f"Response: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_streaming_request(name: str, payload: dict) -> bool:
    """
    Test a streaming request.

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Test: {name} (Streaming)")
    print(f"{'='*60}")

    try:
        with httpx.stream("POST", BASE_URL, json=payload, headers=HEADERS, timeout=60.0) as response:
            print(f"Status: {response.status_code}")

            if response.status_code != 200:
                print(f"❌ Expected status 200, got {response.status_code}")
                return False

            print("✅ Stream started...")
            event_count = 0
            text_chunks = []

            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("event: "):
                    event_type = line[7:]
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        event_count += 1

                        # Print important events
                        if event_type == "response.created":
                            print(f"  📩 response.created: {data.get('response', {}).get('id')}")
                        elif event_type == "response.output_text.delta":
                            delta = data.get("delta", "")
                            text_chunks.append(delta)
                            print(f"  📝 delta: {delta}", end="", flush=True)
                        elif event_type == "response.completed":
                            print(f"\n  ✅ response.completed")
                            usage = data.get("response", {}).get("usage", {})
                            print(f"     Usage: input={usage.get('input_tokens')}, output={usage.get('output_tokens')}")
                        elif event_type == "error":
                            print(f"\n  ❌ Error: {data}")

                    except json.JSONDecodeError:
                        pass

            print(f"\n  Total events: {event_count}")
            full_text = "".join(text_chunks)
            print(f"  Full text: {full_text[:200]}{'...' if len(full_text) > 200 else ''}")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_get_response(response_id: str) -> dict | None:
    """Test GET /v1/responses/{id} endpoint."""
    print(f"\n{'='*60}")
    print(f"Test: GET response by ID")
    print(f"{'='*60}")
    print(f"Response ID: {response_id}")

    try:
        response = httpx.get(f"{BASE_URL}/{response_id}", headers=HEADERS, timeout=30.0)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieved response: {data.get('id')}")
            return data
        else:
            print(f"❌ Failed: {response.text[:200]}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_delete_response(response_id: str) -> bool:
    """Test DELETE /v1/responses/{id} endpoint."""
    print(f"\n{'='*60}")
    print(f"Test: DELETE response")
    print(f"{'='*60}")
    print(f"Response ID: {response_id}")

    try:
        response = httpx.delete(f"{BASE_URL}/{response_id}", headers=HEADERS, timeout=30.0)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Deleted: {data.get('id')}, deleted={data.get('deleted')}")
            return True
        else:
            print(f"❌ Failed: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_error_cases():
    """Test various error scenarios."""
    print(f"\n{'='*60}")
    print("Testing Error Cases")
    print(f"{'='*60}")

    # Test 1: Missing model
    print("\n1. Missing model (expect 422):")
    response = httpx.post(BASE_URL, json={}, headers=HEADERS, timeout=30.0)
    print(f"   Status: {response.status_code} {'✅' if response.status_code == 422 else '❌'}")

    # Test 2: Invalid model
    print("\n2. Invalid model (expect 404):")
    response = httpx.post(BASE_URL, json={"model": "invalid-model-xxx"}, headers=HEADERS, timeout=30.0)
    print(f"   Status: {response.status_code} {'✅' if response.status_code == 404 else '❌'}")

    # Test 3: Non-existent response ID
    print("\n3. Non-existent response ID (expect 404):")
    response = httpx.get(f"{BASE_URL}/resp_nonexistent123", headers=HEADERS, timeout=30.0)
    print(f"   Status: {response.status_code} {'✅' if response.status_code == 404 else '❌'}")


def run_all_tests():
    """Run all e2e tests."""
    print("\n" + "="*60)
    print("OpenAI Response API - End-to-End Tests")
    print("="*60)
    print(f"Base URL: {BASE_URL}")

    # Check server is running
    try:
        response = httpx.get("http://127.0.0.1:8450/v1/models", timeout=5.0)
        if response.status_code != 200:
            print("❌ Server not responding correctly")
            sys.exit(1)
        models = response.json().get("data", [])
        if models:
            model_id = models[0].get("id")
            print(f"✅ Server running, using model: {model_id}")
        else:
            print("❌ No models available")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        sys.exit(1)

    results = []

    # Test 1: Simple text request
    result = test_request(
        "1. Simple text request",
        {
            "model": model_id,
            "input": "Say 'Hello, World!' and nothing else."
        }
    )
    results.append(("Simple text request", result is not None))
    response_id_1 = result.get("id") if result else None

    # Test 2: Request with instructions
    result = test_request(
        "2. Request with instructions",
        {
            "model": model_id,
            "input": "What is 2+2?",
            "instructions": "Be very brief. Answer in one word."
        }
    )
    results.append(("Request with instructions", result is not None))

    # Test 3: Multi-turn conversation with message array
    result = test_request(
        "3. Multi-turn conversation",
        {
            "model": model_id,
            "input": [
                {"role": "user", "content": "My name is Alice.", "type": "message"},
                {"role": "assistant", "content": "Nice to meet you, Alice!", "type": "message"},
                {"role": "user", "content": "What is my name?", "type": "message"}
            ]
        }
    )
    results.append(("Multi-turn conversation", result is not None))

    # Test 4: Streaming request
    result = test_streaming_request(
        "4. Streaming request",
        {
            "model": model_id,
            "input": "Count from 1 to 5, one number per line.",
            "stream": True
        }
    )
    results.append(("Streaming request", result))

    # Test 5: GET response by ID
    if response_id_1:
        result = test_get_response(response_id_1)
        results.append(("GET response by ID", result is not None))

    # Test 6: DELETE response
    if response_id_1:
        result = test_delete_response(response_id_1)
        results.append(("DELETE response", result))

        # Verify deletion
        print("\n  Verifying deletion...")
        verify_result = test_get_response(response_id_1)
        results.append(("Verify deletion (should fail)", verify_result is None))

    # Test 7: Request with tools
    result = test_request(
        "7. Request with tools",
        {
            "model": model_id,
            "input": "What is the weather in Tokyo?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            ]
        }
    )
    results.append(("Request with tools", result is not None))

    # Test 8: Stateful conversation with previous_response_id
    result1 = test_request(
        "8a. First message in conversation",
        {
            "model": model_id,
            "input": "Remember that my favorite color is blue.",
            "store": True
        }
    )
    if result1:
        prev_id = result1.get("id")
        result2 = test_request(
            "8b. Second message with previous_response_id",
            {
                "model": model_id,
                "input": "What is my favorite color?",
                "store": True,
                "previous_response_id": prev_id
            }
        )
        results.append(("Stateful conversation", result2 is not None))
    else:
        results.append(("Stateful conversation", False))

    # Test error cases
    test_error_cases()

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    failed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
