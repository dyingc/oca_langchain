"""
End-to-end API tests for message validation functionality.

These tests make actual HTTP requests to the running API server
to verify that the message validation works correctly in production.
"""

import requests
import json
import sys


# Configuration
API_BASE_URL = "http://127.0.0.1:8450"
MODEL_NAME = "oca/gpt-4.1"


def print_test_header(test_name: str):
    """Print a formatted test header."""
    print(f"\n{'=' * 70}")
    print(f"TEST: {test_name}")
    print('=' * 70)


def print_result(passed: bool, message: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {message}")
    return passed


def test_openai_valid_sequence():
    """Test: Valid tool_calls sequence should succeed."""
    print_test_header("OpenAI API - Valid Sequence")

    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=request_data,
            timeout=30
        )

        passed = response.status_code == 200
        return print_result(
            passed,
            f"Status: {response.status_code}, Expected: 200"
        )
    except Exception as e:
        return print_result(False, f"Exception: {e}")


def test_openai_interrupted_tool_calls():
    """Test: Interrupted tool_calls should be auto-fixed and succeed."""
    print_test_header("OpenAI API - Interrupted tool_calls (Auto-Fix)")

    # Simulate a scenario where tool_calls were interrupted by user message
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_interrupted_123",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "test"}'
                        }
                    }
                ]
            },
            {"role": "user", "content": "Wait, stop that search!"},  # INTERRUPT
            {
                "role": "tool",
                "tool_call_id": "call_interrupted_123",
                "content": '{"result": "This should be skipped"}'
            },
            {"role": "user", "content": "Just say hello"},
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=request_data,
            timeout=30
        )

        # Should succeed (200) because the validation removes the incomplete tool_calls
        passed = response.status_code == 200

        if passed:
            print(f"✅ Request succeeded with auto-fix")
            print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")
        else:
            print(f"❌ Request failed: {response.text}")

        return passed
    except Exception as e:
        return print_result(False, f"Exception: {e}")


def test_anthropic_valid_sequence():
    """Test: Valid Anthropic sequence should succeed."""
    print_test_header("Anthropic API - Valid Sequence")

    request_data = {
        "model": MODEL_NAME,
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/messages",
            json=request_data,
            headers={
                "anthropic-version": "2023-06-01"
            },
            timeout=30
        )

        passed = response.status_code == 200
        return print_result(
            passed,
            f"Status: {response.status_code}, Expected: 200"
        )
    except Exception as e:
        return print_result(False, f"Exception: {e}")


def test_anthropic_interrupted_tool_use():
    """Test: Interrupted tool_use should be auto-fixed and succeed."""
    print_test_header("Anthropic API - Interrupted tool_use (Auto-Fix)")

    request_data = {
        "model": MODEL_NAME,
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Search for something"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_interrupted_123",
                        "name": "web_search",
                        "input": {"query": "test"}
                    }
                ]
            },
            {"role": "user", "content": "Wait, stop!"},  # INTERRUPT
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_interrupted_123",
                        "content": '{"result": "skipped"}'
                    }
                ]
            },
            {"role": "user", "content": "Just say hello"},
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/messages",
            json=request_data,
            headers={
                "anthropic-version": "2023-06-01"
            },
            timeout=30
        )

        # Should succeed (200) because the validation removes the incomplete tool_use
        passed = response.status_code == 200

        if passed:
            print(f"✅ Request succeeded with auto-fix")
            print(f"Response keys: {list(response.json().keys())}")
        else:
            print(f"❌ Request failed: {response.text}")

        return passed
    except Exception as e:
        return print_result(False, f"Exception: {e}")


def test_streaming_with_validation():
    """Test: Streaming response with message validation."""
    print_test_header("OpenAI API - Streaming with Validation")

    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Say 'hello world'"},
        ],
        "stream": True,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=request_data,
            stream=True,
            timeout=30
        )

        passed = response.status_code == 200

        if passed:
            # Read some chunks to verify streaming works
            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    chunk_count += 1
                    if chunk_count >= 5:  # Just check first few chunks
                        break

            print(f"✅ Streaming works, received {chunk_count} chunks")
        else:
            print(f"❌ Streaming failed: {response.status_code}")

        return passed
    except Exception as e:
        return print_result(False, f"Exception: {e}")


def check_server_health():
    """Check if the API server is running."""
    print_test_header("Server Health Check")

    try:
        response = requests.get(f"{API_BASE_URL}/v1/models", timeout=5)
        passed = response.status_code == 200

        if passed:
            models = response.json().get("data", [])
            print(f"✅ Server is running")
            print(f"Available models: {[m['id'] for m in models]}")
        else:
            print(f"❌ Server returned status {response.status_code}")

        return passed
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {API_BASE_URL}")
        print(f"Please start the server with: bash run_api.sh")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def run_all_tests():
    """Run all API tests."""
    print("\n" + "=" * 70)
    print("END-TO-END API VALIDATION TEST SUITE")
    print("=" * 70)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    # First check if server is running
    if not check_server_health():
        print("\n❌ Server is not running. Please start the API server first:")
        print("   bash run_api.sh")
        return False

    tests = [
        test_openai_valid_sequence,
        test_openai_interrupted_tool_calls,
        test_anthropic_valid_sequence,
        test_anthropic_interrupted_tool_use,
        test_streaming_with_validation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total: {passed + failed} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    print("\n💡 Tip: Make sure the API server is running before executing this script:")
    print("   bash run_api.sh\n")

    success = run_all_tests()
    sys.exit(0 if success else 1)
