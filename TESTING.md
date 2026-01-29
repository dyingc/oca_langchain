# Message Validation Tests

This document describes the test suite for message sequence validation and repair functionality.

## Test Categories

### POSITIVE Tests (Should NOT Change Messages)

These tests verify that **valid** message sequences are **preserved unchanged**:

1. **OpenAI - Normal Complete Sequence**: Valid tool_calls with all responses
2. **OpenAI - No tool_calls**: Simple conversation without tools
3. **Anthropic - Normal Complete Sequence**: Valid tool_use with all tool_results

**Expected Behavior**: Messages should remain **exactly the same** - no modifications.

### NEGATIVE Tests (Should Fix Messages)

These tests verify that **invalid** message sequences are **properly repaired**:

1. **OpenAI - User Message Interrupting tool_calls**: User message appears before all tool responses
2. **OpenAI - Partial Tool Responses**: Some tool responses are missing
3. **OpenAI - Multiple Assistant with tool_calls**: Mix of valid and invalid sequences
4. **Anthropic - Interrupted tool_use**: User message interrupts tool_use blocks
5. **Anthropic - Text + tool_use Interrupted**: Text should be preserved, tool_use removed

**Expected Behavior**: Incomplete tool_calls/tool_use should be removed, orphaned responses skipped.

### Integration Tests

These tests verify the end-to-end pipeline:

1. **OpenAI to LangChain Conversion**: Validation + conversion works together
2. **Anthropic to LangChain Conversion**: Validation + conversion works together

## Running the Tests

### Unit Tests (No server required)

```bash
# Run unit tests for validation logic
python test_message_validation.py
```

Expected output:
```
==================================================================
TEST: OpenAI - POSITIVE: Valid Sequence Should NOT Be Changed
==================================================================
  ✅ All 5 messages preserved exactly as-is
✅ PASS: Expected 5 messages unchanged, got 5

==================================================================
TEST: OpenAI - NEGATIVE: User Message Interrupting tool_calls (Should Fix)
==================================================================
  ✅ tool_calls correctly removed from assistant message
  ✅ Orphaned tool messages correctly removed
  ✅ Result: 3 messages (expected 3)
✅ PASS: Expected 3 messages with no tool_calls, got 3 messages
...
```

### API Tests (Server required)

First, start the API server:

```bash
bash run_api.sh
```

Then run the API tests:

```bash
python test_api_validation.py
```

Expected output:
```
==================================================================
END-TO-END API VALIDATION TEST SUITE
==================================================================
API Base URL: http://127.0.0.1:8450
Model: oca/gpt-4.1
==================================================================

==================================================================
Server Health Check
==================================================================
✅ Server is running
Available models: ['oca/gpt-4.1', ...]

==================================================================
TEST: OpenAI API - Valid Sequence
==================================================================
✅ PASS: Status: 200, Expected: 200

==================================================================
TEST: OpenAI API - Interrupted tool_calls (Auto-Fix)
==================================================================
✅ Request succeeded with auto-fix
✅ PASS: ...
```

## Test Results Interpretation

### ✅ PASS
- **POSITIVE tests**: Valid sequences were NOT modified (correct)
- **NEGATIVE tests**: Invalid sequences were properly repaired (correct)

### ❌ FAIL
- **POSITIVE tests**: Valid sequences were modified (bug - should not change!)
- **NEGATIVE tests**: Invalid sequences were not repaired (bug - should fix!)

## Common Issues

### "Server is not running"
Start the API server first:
```bash
bash run_api.sh
```

### Import errors
Make sure you're running from the project root directory:
```bash
cd /path/to/oca_langchain
python test_message_validation.py
```

### Tests passing but messages still being changed
This indicates a **critical bug** - the validation is too aggressive and modifying valid sequences.
Check the validation logic in `api.py` and `converters/anthropic_request_converter.py`.

## Adding New Tests

When adding new test cases:

1. Clearly mark as **POSITIVE** or **NEGATIVE** in the test name
2. For POSITIVE tests: Verify deep equality (content, roles, tool_calls unchanged)
3. For NEGATIVE tests: Verify the fix was applied correctly
4. Add detailed logging to show what was checked

Example:
```python
def test_openai_positive_case():
    """POSITIVE: Valid sequence should not change."""
    print_test_header("OpenAI - POSITIVE: ...")

    messages = [...]  # Valid sequence
    result = validate_and_fix_message_sequence(messages)

    # Deep verify
    all_match = all(
        m1.role == m2.role and
        m1.content == m2.content and
        m1.tool_calls == m2.tool_calls
        for m1, m2 in zip(messages, result)
    )

    return print_result(all_match, "Valid sequence unchanged")
```

## Continuous Integration

These tests should be run:
- Before committing changes to validation logic
- In CI/CD pipeline to catch regressions
- When updating message format converters

## Related Files

- `api.py`: OpenAI format validation
- `converters/anthropic_request_converter.py`: Anthropic format validation
- `test_message_validation.py`: Unit tests
- `test_api_validation.py`: Integration tests
