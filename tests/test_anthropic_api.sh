#!/bin/bash

# Anthropic API Compatibility Test Suite
# Tests for /v1/messages endpoint (non-streaming and streaming)

BASE_URL="http://127.0.0.1:8450"
MODEL_NAME="oca/gpt-4.1"  # Adjust to your available model

echo "========================================="
echo "Anthropic API Compatibility Test Suite"
echo "========================================="
echo "Base URL: $BASE_URL"
echo "Model: $MODEL_NAME"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function for test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        ((TESTS_FAILED++))
    fi
}

# Helper function to check JSON field
check_json_field() {
    local json="$1"
    local field="$2"
    echo "$json" | grep -q "\"$field"
    return $?
}

# Test 1: Basic message (non-streaming)
echo -e "${YELLOW}Test 1: Basic message (non-streaming)${NC}"
echo "----------------------------------------------"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/messages" \
    -H "Content-Type: application/json" \
    -H "x-api-key: test" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"max_tokens\": 100,
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"Hello! Please respond with a short greeting.\"
            }
        ]
    }")

echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
check_json_field "$RESPONSE" "type"
print_result $? "Basic message response has 'type' field"
echo ""

# Test 2: Message with system prompt
echo -e "${YELLOW}Test 2: Message with system prompt${NC}"
echo "----------------------------------------------"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/messages" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"max_tokens\": 100,
        \"system\": \"You are a helpful assistant.\",
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"What is 2+2?\"
            }
        ]
    }")

echo "$RESPONSE" | python3 -m json.tool
check_json_field "$RESPONSE" "content"
print_result $? "Message with system prompt has 'content' field"
echo ""

# Test 3: Multi-turn conversation
echo -e "${YELLOW}Test 3: Multi-turn conversation${NC}"
echo "----------------------------------------------"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/messages" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"max_tokens\": 100,
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"My name is Alice\"
            },
            {
                \"role\": \"assistant\",
                \"content\": \"Nice to meet you, Alice!\"
            },
            {
                \"role\": \"user\",
                \"content\": \"What's my name?\"
            }
        ]
    }")

echo "$RESPONSE" | python3 -m json.tool
check_json_field "$RESPONSE" "role"
print_result $? "Multi-turn conversation has 'role' field"
echo ""

# Test 4: Tool calls (non-streaming)
echo -e "${YELLOW}Test 4: Tool calls (non-streaming)${NC}"
echo "----------------------------------------------"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/messages" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"max_tokens\": 200,
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"What's the weather in Tokyo?\"
            }
        ],
        \"tools\": [
            {
                \"name\": \"get_weather\",
                \"description\": \"Get the current weather in a location\",
                \"input_schema\": {
                    \"type\": \"object\",
                    \"properties\": {
                        \"location\": {
                            \"type\": \"string\",
                            \"description\": \"The city and state, e.g. San Francisco, CA\"
                        }
                    },
                    \"required\": [\"location\"]
                }
            }
        ]
    }")

echo "$RESPONSE" | python3 -m json.tool
echo "$RESPONSE" | grep -q "tool_use"
if [ $? -eq 0 ]; then
    print_result 0 "Tool calls response contains 'tool_use'"
else
    # If no tool_use, check if there's a text response (model might not call the tool)
    echo "$RESPONSE" | grep -q '"type": "text"'
    print_result $? "Tool calls - got text response (model might not have called tool)"
fi
echo ""

# Test 5: Streaming response
echo -e "${YELLOW}Test 5: Streaming response${NC}"
echo "----------------------------------------------"
STREAM_OUTPUT=$(curl -N -s -X POST "${BASE_URL}/v1/messages" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"max_tokens\": 50,
        \"stream\": true,
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"Count from 1 to 3\"
            }
        ]
    }" | head -20)

echo "$STREAM_OUTPUT"
echo "$STREAM_OUTPUT" | grep -q "event: message_start"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}: Streaming response has message_start event"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}: Streaming response missing message_start event"
    ((TESTS_FAILED++))
fi
echo ""

# Test 6: Error handling - invalid model
echo -e "${YELLOW}Test 6: Error handling - invalid model${NC}"
echo "----------------------------------------------"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/messages" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"invalid-model-name\",
        \"max_tokens\": 100,
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"Hello\"
            }
        ]
    }")

echo "$RESPONSE" | python3 -m json.tool
echo "$RESPONSE" | grep -q "not_found_error"
print_result $? "Error response for invalid model"
echo ""

# Test 7: Error handling - empty messages
echo -e "${YELLOW}Test 7: Error handling - empty messages${NC}"
echo "----------------------------------------------"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/messages" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"max_tokens\": 100,
        \"messages\": []
    }")

echo "$RESPONSE" | python3 -m json.tool
echo "$RESPONSE" | grep -q "invalid_request_error"
print_result $? "Error response for empty messages"
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo "Total:  $((TESTS_PASSED + TESTS_FAILED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
