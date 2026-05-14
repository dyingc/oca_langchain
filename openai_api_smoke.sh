#!/usr/bin/env bash

# Smoke test for the local OpenAI-compatible API started by run_api.sh.

set -u

BASE_URL="${BASE_URL:-http://127.0.0.1:8450}"
MODEL="${MODEL:-oca/gpt-5.4}"
CHAT_MODEL="${CHAT_MODEL:-$MODEL}"
RESPONSES_MODEL="${RESPONSES_MODEL:-$MODEL}"
MAX_TOKENS="${MAX_TOKENS:-80}"
TIMEOUT="${TIMEOUT:-180}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
WORK_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

print_header() {
    echo "========================================="
    echo "OpenAI-Compatible API Smoke Test"
    echo "========================================="
    echo "Base URL: $BASE_URL"
    echo "Chat model: $CHAT_MODEL"
    echo "Responses model: $RESPONSES_MODEL"
    echo ""
}

record_pass() {
    echo -e "${GREEN}PASS${NC}: $1"
    PASSED=$((PASSED + 1))
}

record_fail() {
    echo -e "${RED}FAIL${NC}: $1"
    FAILED=$((FAILED + 1))
}

post_json() {
    local endpoint="$1"
    local payload_file="$2"
    local response_file="$3"
    local error_file="${response_file}.curl_error"
    local http_status
    local curl_exit

    http_status="$(curl -sS \
        --max-time "$TIMEOUT" \
        -o "$response_file" \
        -w "%{http_code}" \
        -X POST "${BASE_URL}${endpoint}" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer test" \
        --data @"$payload_file" \
        2>"$error_file")"
    curl_exit=$?

    if [ "$curl_exit" -ne 0 ]; then
        echo "curl_exit_${curl_exit}_http_${http_status}"
    else
        echo "$http_status"
    fi
}

print_json_or_raw() {
    local response_file="$1"
    local error_file="${response_file}.curl_error"

    if [ -s "$response_file" ]; then
        python3 -m json.tool "$response_file" 2>/dev/null || cat "$response_file"
    fi

    if [ -s "$error_file" ]; then
        cat "$error_file"
    fi
}

summarize_chat_stream_response() {
    local response_file="$1"
    python3 - "$response_file" <<'PY'
import json
import sys

text_parts = []
tool_call_seen = False

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if data.get("error"):
            print(data["error"])
            sys.exit(1)
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
            if delta.get("tool_calls"):
                tool_call_seen = True

summary = "".join(text_parts).strip()
if not summary and not tool_call_seen:
    print("missing stream content or tool calls")
    sys.exit(1)

print(summary.replace("\n", " ")[:300] or "[tool_calls]")
PY
}

summarize_responses_stream_response() {
    local response_file="$1"
    python3 - "$response_file" <<'PY'
import json
import sys

text_parts = []
function_call_seen = False

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if data.get("error"):
            print(data["error"])
            sys.exit(1)
        event_type = data.get("type")
        if event_type == "response.output_text.delta":
            text_parts.append(data.get("delta", ""))
        elif event_type == "response.function_call_arguments.delta":
            function_call_seen = True
        elif event_type == "response.completed":
            response = data.get("response") or {}
            for item in response.get("output") or []:
                if item.get("type") == "message":
                    for block in item.get("content") or []:
                        if block.get("type") == "output_text":
                            text_parts.append(block.get("text", ""))
                elif item.get("type") == "function_call":
                    function_call_seen = True

summary = "".join(text_parts).strip()
if not summary and not function_call_seen:
    print("missing stream output text or function call")
    sys.exit(1)

print(summary.replace("\n", " ")[:300] or "[function_call]")
PY
}

run_chat_completions_test() {
    local payload_file="$WORK_DIR/chat_payload.json"
    local response_file="$WORK_DIR/chat_response.json"
    local status

    cat > "$payload_file" <<JSON
{
  "model": "$CHAT_MODEL",
  "messages": [
    {
      "role": "user",
      "content": "Reply with exactly: chat completion ok"
    }
  ],
  "max_tokens": $MAX_TOKENS,
  "stream": true
}
JSON

    echo -e "${YELLOW}Test: /v1/chat/completions${NC}"
    status="$(post_json "/v1/chat/completions" "$payload_file" "$response_file")"

    if [ "$status" = "200" ] && summary="$(summarize_chat_stream_response "$response_file")"; then
        record_pass "Chat Completions API returned a usable stream"
        echo "Result: $summary"
    else
        record_fail "Chat Completions API failed with HTTP $status"
        print_json_or_raw "$response_file"
    fi
    echo ""
}

run_responses_test() {
    local payload_file="$WORK_DIR/responses_payload.json"
    local response_file="$WORK_DIR/responses_response.json"
    local status

    cat > "$payload_file" <<JSON
{
  "model": "$RESPONSES_MODEL",
  "input": "Reply with exactly: responses api ok",
  "max_output_tokens": $MAX_TOKENS,
  "stream": true,
  "store": false
}
JSON

    echo -e "${YELLOW}Test: /v1/responses${NC}"
    status="$(post_json "/v1/responses" "$payload_file" "$response_file")"

    if [ "$status" = "200" ] && summary="$(summarize_responses_stream_response "$response_file")"; then
        record_pass "Responses API returned a usable stream"
        echo "Result: $summary"
    else
        record_fail "Responses API failed with HTTP $status"
        print_json_or_raw "$response_file"
    fi
    echo ""
}

print_header
run_chat_completions_test
run_responses_test

echo "Summary: $PASSED passed, $FAILED failed"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
