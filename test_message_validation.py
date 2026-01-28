"""
Test suite for message sequence validation and repair functionality.

Tests the validation logic that detects and fixes incomplete tool_calls sequences
in both OpenAI and Anthropic message formats.
"""

import sys
import os
import json

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import validate_and_fix_message_sequence, ChatMessage
from converters.anthropic_request_converter import (
    validate_and_fix_anthropic_message_sequence,
    anthropic_to_langchain_messages,
)
from models.anthropic_types import AnthropicMessage, AnthropicContentBlock


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


# ============================================================================
# Tests for OpenAI Format Message Validation
# ============================================================================

def test_openai_normal_sequence():
    """Test: Normal, complete tool_calls sequence should be preserved unchanged."""
    print_test_header("OpenAI - POSITIVE: Valid Sequence Should NOT Be Changed")

    messages = [
        ChatMessage(role="user", content="Search the web"),
        ChatMessage(role="assistant", content="", tool_calls=[
            {
                "type": "function",
                "id": "call_123",
                "function": {"name": "search", "arguments": json.dumps({"query": "test"})}
            },
            {
                "type": "function",
                "id": "call_456",
                "function": {"name": "search", "arguments": json.dumps({"query": "test2"})}
            },
        ]),
        ChatMessage(role="tool", tool_call_id="call_123", content=json.dumps({"result": "ok"})),
        ChatMessage(role="tool", tool_call_id="call_456", content=json.dumps({"result": "ok2"})),
        ChatMessage(role="user", content="What did you find?"),
    ]

    result = validate_and_fix_message_sequence(messages)

    # Should preserve ALL messages unchanged
    count_match = len(result) == len(messages)

    # Deep check: verify each message is identical
    all_match = True
    for i, (orig, res) in enumerate(zip(messages, result)):
        if orig.role != res.role:
            print(f"  ❌ Message {i}: role changed from {orig.role} to {res.role}")
            all_match = False
        if orig.content != res.content:
            print(f"  ❌ Message {i}: content changed")
            all_match = False
        if orig.tool_calls != res.tool_calls:
            print(f"  ❌ Message {i}: tool_calls changed")
            all_match = False
        if orig.tool_call_id != res.tool_call_id:
            print(f"  ❌ Message {i}: tool_call_id changed")
            all_match = False

    passed = count_match and all_match

    if passed:
        print(f"  ✅ All {len(messages)} messages preserved exactly as-is")

    return print_result(passed, f"Expected {len(messages)} messages unchanged, got {len(result)}")


def test_openai_interrupted_by_user():
    """Test: User message interrupting tool_calls should trigger cleanup."""
    print_test_header("OpenAI - NEGATIVE: User Message Interrupting tool_calls (Should Fix)")

    messages = [
        ChatMessage(role="user", content="Search the web"),
        ChatMessage(role="assistant", content="", tool_calls=[
            {
                "type": "function",
                "id": "call_123",
                "function": {"name": "search", "arguments": json.dumps({"query": "test"})}
            },
            {
                "type": "function",
                "id": "call_456",
                "function": {"name": "search", "arguments": json.dumps({"query": "test2"})}
            },
        ]),
        ChatMessage(role="user", content="Wait, stop!"),  # INTERRUPT - Invalid!
        ChatMessage(role="tool", tool_call_id="call_123", content=json.dumps({"result": "ok"})),
        ChatMessage(role="tool", tool_call_id="call_456", content=json.dumps({"result": "ok2"})),
    ]

    result = validate_and_fix_message_sequence(messages)

    # SHOULD remove tool_calls from assistant message
    # SHOULD skip orphaned tool messages
    expected_count = 3

    # Verify the fix was applied correctly
    has_no_tool_calls = result[1].tool_calls is None
    count_match = len(result) == expected_count

    # Verify orphaned tool messages were removed
    no_tool_messages = all(m.role != "tool" for m in result)

    passed = count_match and has_no_tool_calls and no_tool_messages

    if passed:
        print(f"  ✅ tool_calls correctly removed from assistant message")
        print(f"  ✅ Orphaned tool messages correctly removed")
        print(f"  ✅ Result: {len(result)} messages (expected {expected_count})")
    else:
        print(f"  ❌ Expected {expected_count} messages, got {len(result)}")
        print(f"  ❌ Assistant has no tool_calls: {has_no_tool_calls}")
        print(f"  ❌ No tool messages in result: {no_tool_messages}")

    return print_result(
        passed,
        f"Expected {expected_count} messages with no tool_calls, got {len(result)} messages"
    )


def test_openai_partial_tool_responses():
    """Test: Missing some tool responses should trigger cleanup."""
    print_test_header("OpenAI - NEGATIVE: Partial Tool Responses (Should Fix)")

    messages = [
        ChatMessage(role="user", content="Search the web"),
        ChatMessage(role="assistant", content="", tool_calls=[
            {
                "type": "function",
                "id": "call_123",
                "function": {"name": "search", "arguments": json.dumps({"query": "test"})}
            },
            {
                "type": "function",
                "id": "call_456",
                "function": {"name": "search", "arguments": json.dumps({"query": "test2"})}
            },
        ]),
        ChatMessage(role="tool", tool_call_id="call_123", content=json.dumps({"result": "ok"})),
        # Missing response for call_456 - Invalid!
        ChatMessage(role="user", content="Continue"),
    ]

    result = validate_and_fix_message_sequence(messages)

    # SHOULD detect incomplete sequence and remove tool_calls
    expected_count = 3  # user, assistant (cleaned), user
    count_match = len(result) == expected_count
    no_tool_calls = result[1].tool_calls is None

    passed = count_match and no_tool_calls

    if passed:
        print(f"  ✅ Incomplete sequence detected and fixed")
        print(f"  ✅ tool_calls removed (expected 2, only had 1 response)")
    else:
        print(f"  ❌ Expected {expected_count} messages, got {len(result)}")

    return print_result(
        passed,
        f"Expected {expected_count} messages, got {len(result)}"
    )


def test_openai_multiple_assistant_with_tools():
    """Test: Multiple assistant messages with tool_calls - mix of valid and invalid."""
    print_test_header("OpenAI - MIXED: Multiple tool_calls (First Valid, Second Invalid)")

    messages = [
        ChatMessage(role="user", content="First search"),
        ChatMessage(role="assistant", content="", tool_calls=[
            {
                "type": "function",
                "id": "call_1",
                "function": {"name": "search", "arguments": json.dumps({"q": "1"})}
            },
        ]),
        ChatMessage(role="tool", tool_call_id="call_1", content=json.dumps({"result": "1"})),
        ChatMessage(role="user", content="Second search"),
        ChatMessage(role="assistant", content="", tool_calls=[
            {
                "type": "function",
                "id": "call_2",
                "function": {"name": "search", "arguments": json.dumps({"q": "2"})}
            },
        ]),
        ChatMessage(role="user", content="Interrupt again!"),  # Invalid for second tool_calls
        ChatMessage(role="tool", tool_call_id="call_2", content=json.dumps({"result": "2"})),
    ]

    result = validate_and_fix_message_sequence(messages)

    # First tool_calls SHOULD be preserved (valid sequence)
    # Second tool_calls SHOULD be removed (invalid - interrupted)
    expected_count = 6
    count_match = len(result) == expected_count
    first_has_tools = result[1].tool_calls is not None
    second_no_tools = result[4].tool_calls is None

    passed = count_match and first_has_tools and second_no_tools

    if passed:
        print(f"  ✅ First assistant's tool_calls preserved (valid)")
        print(f"  ✅ Second assistant's tool_calls removed (invalid)")
    else:
        print(f"  ❌ Count: expected {expected_count}, got {len(result)}")
        print(f"  ❌ First has tools: {first_has_tools}")
        print(f"  ❌ Second has no tools: {second_no_tools}")

    return print_result(
        passed,
        f"First assistant: has tools={first_has_tools}, "
        f"Second assistant: has tools={second_no_tools}"
    )


def test_openai_no_tool_calls():
    """Test: Messages without tool_calls should remain unchanged."""
    print_test_header("OpenAI - POSITIVE: No tool_calls (Should NOT Be Changed)")

    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
        ChatMessage(role="user", content="How are you?"),
        ChatMessage(role="assistant", content="I'm doing well!"),
    ]

    result = validate_and_fix_message_sequence(messages)

    # Should remain completely unchanged
    count_match = len(result) == len(messages)

    # Deep verify
    all_match = True
    for i, (orig, res) in enumerate(zip(messages, result)):
        if orig.role != res.role or orig.content != res.content:
            print(f"  ❌ Message {i} was modified!")
            all_match = False

    passed = count_match and all_match

    if passed:
        print(f"  ✅ All {len(messages)} messages unchanged")

    return print_result(passed, f"Expected {len(messages)} messages unchanged, got {len(result)}")


# ============================================================================
# Tests for Anthropic Format Message Validation
# ============================================================================

def test_anthropic_normal_sequence():
    """Test: Normal Anthropic tool_use sequence should be preserved unchanged."""
    print_test_header("Anthropic - POSITIVE: Valid Sequence Should NOT Be Changed")

    messages = [
        AnthropicMessage(role="user", content="Search the web"),
        AnthropicMessage(role="assistant", content=[
            AnthropicContentBlock(type="tool_use", id="toolu_123", name="search", input={"query": "test"}),
            AnthropicContentBlock(type="tool_use", id="toolu_456", name="search", input={"query": "test2"}),
        ]),
        AnthropicMessage(role="user", content=[
            AnthropicContentBlock(type="tool_result", tool_use_id="toolu_123", content=json.dumps({"result": "ok"})),
            AnthropicContentBlock(type="tool_result", tool_use_id="toolu_456", content=json.dumps({"result": "ok2"})),
        ]),
        AnthropicMessage(role="user", content="What did you find?"),
    ]

    result = validate_and_fix_anthropic_message_sequence(messages)

    # Should remain completely unchanged
    count_match = len(result) == len(messages)

    # Deep verify - check content blocks are preserved
    all_match = True
    for i, (orig, res) in enumerate(zip(messages, result)):
        if orig.role != res.role:
            print(f"  ❌ Message {i}: role changed from {orig.role} to {res.role}")
            all_match = False
        if orig.content != res.content:
            print(f"  ❌ Message {i}: content blocks changed")
            all_match = False

    passed = count_match and all_match

    if passed:
        print(f"  ✅ All {len(messages)} messages preserved exactly")

    return print_result(passed, f"Expected {len(messages)} messages unchanged, got {len(result)}")


def test_anthropic_interrupted_sequence():
    """Test: User message interrupting tool_use blocks should trigger cleanup."""
    print_test_header("Anthropic - NEGATIVE: Interrupted tool_use (Should Fix)")

    messages = [
        AnthropicMessage(role="user", content="Search the web"),
        AnthropicMessage(role="assistant", content=[
            AnthropicContentBlock(type="tool_use", id="toolu_123", name="search", input={"query": "test"}),
        ]),
        AnthropicMessage(role="user", content="Wait, stop!"),  # INTERRUPT - Invalid!
        AnthropicMessage(role="user", content=[
            AnthropicContentBlock(type="tool_result", tool_use_id="toolu_123", content=json.dumps({"result": "ok"})),
        ]),
    ]

    result = validate_and_fix_anthropic_message_sequence(messages)

    # SHOULD remove tool_use from assistant message
    # SHOULD skip orphaned tool_result
    expected_count = 3  # user, assistant (empty or text-only), user ("Wait, stop!")
    count_match = len(result) == expected_count

    # Verify no tool_use blocks remain
    has_no_tool_use = True
    for msg in result:
        if isinstance(msg.content, list):
            for block in msg.content:
                if block.type == "tool_use":
                    has_no_tool_use = False
                    break

    passed = count_match and has_no_tool_use

    if passed:
        print(f"  ✅ tool_use block correctly removed")
        print(f"  ✅ Orphaned tool_result correctly removed")
    else:
        print(f"  ❌ Expected {expected_count} messages, got {len(result)}")

    return print_result(
        passed,
        f"Expected {expected_count} messages, got {len(result)}"
    )


def test_anthropic_text_with_tool_use():
    """Test: Assistant message with both text and tool_use - should preserve text on fix."""
    print_test_header("Anthropic - NEGATIVE: Text + tool_use Interrupted (Should Fix)")

    messages = [
        AnthropicMessage(role="user", content="Search"),
        AnthropicMessage(role="assistant", content=[
            AnthropicContentBlock(type="text", text="Let me search for you"),
            AnthropicContentBlock(type="tool_use", id="toolu_123", name="search", input={"query": "test"}),
        ]),
        AnthropicMessage(role="user", content="Never mind"),  # INTERRUPT
    ]

    result = validate_and_fix_anthropic_message_sequence(messages)

    # SHOULD preserve text content but remove tool_use
    expected_count = 3

    # Check that text was preserved
    has_text_content = (
        isinstance(result[1].content, str) and
        "Let me search" in result[1].content
    )

    # Check that tool_use was removed
    has_no_tool_use = True
    if isinstance(result[1].content, list):
        for block in result[1].content:
            if block.type == "tool_use":
                has_no_tool_use = False
                break

    count_match = len(result) == expected_count
    passed = count_match and has_text_content and has_no_tool_use

    if passed:
        print(f"  ✅ Text content preserved: 'Let me search for you'")
        print(f"  ✅ tool_use block correctly removed")
    else:
        print(f"  ❌ Expected {expected_count} messages, got {len(result)}")
        print(f"  ❌ Text preserved: {has_text_content}")
        print(f"  ❌ No tool_use: {has_no_tool_use}")

    return print_result(
        passed,
        f"Preserved text: {has_text_content}, No tool_use: {has_no_tool_use}"
    )


# ============================================================================
# Integration Tests
# ============================================================================

def test_openai_to_langchain_conversion():
    """Test: Converting validated OpenAI messages to LangChain format."""
    print_test_header("Integration - OpenAI to LangChain Conversion")

    from api import convert_to_langchain_messages

    # Test with interrupted sequence that needs fixing
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi!", tool_calls=[
            {
                "type": "function",
                "id": "call_123",
                "function": {"name": "test", "arguments": "{}"}
            }
        ]),
        ChatMessage(role="user", content="Interrupt!"),  # Invalid
        ChatMessage(role="tool", tool_call_id="call_123", content="{}"),
    ]

    # Validate first (should fix)
    validated = validate_and_fix_message_sequence(messages)
    # Then convert
    lc_messages = convert_to_langchain_messages(validated)

    # Should convert successfully without errors
    passed = len(lc_messages) > 0

    if passed:
        print(f"  ✅ Validation fixed sequence: {len(messages)} → {len(validated)} messages")
        print(f"  ✅ Conversion successful: {len(validated)} → {len(lc_messages)} LangChain messages")

    return print_result(
        passed,
        f"Converted {len(validated)} OpenAI messages to {len(lc_messages)} LangChain messages"
    )


def test_anthropic_to_langchain_conversion():
    """Test: Converting validated Anthropic messages to LangChain format."""
    print_test_header("Integration - Anthropic to LangChain Conversion")

    messages = [
        AnthropicMessage(role="user", content="Hello"),
        AnthropicMessage(role="assistant", content=[
            AnthropicContentBlock(type="tool_use", id="toolu_123", name="test", input={}),
        ]),
        AnthropicMessage(role="user", content="Interrupt!"),  # Invalid
        AnthropicMessage(role="user", content=[
            AnthropicContentBlock(type="tool_result", tool_use_id="toolu_123", content="{}"),
        ]),
    ]

    # Validate first (should fix)
    validated = validate_and_fix_anthropic_message_sequence(messages)
    # Then convert
    lc_messages = anthropic_to_langchain_messages(validated)

    passed = len(lc_messages) > 0

    if passed:
        print(f"  ✅ Validation fixed sequence: {len(messages)} → {len(validated)} messages")
        print(f"  ✅ Conversion successful: {len(validated)} → {len(lc_messages)} LangChain messages")

    return print_result(
        passed,
        f"Converted {len(validated)} Anthropic messages to {len(lc_messages)} LangChain messages"
    )


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("MESSAGE VALIDATION TEST SUITE")
    print("=" * 70)

    tests = [
        # OpenAI Format Tests
        ("OpenAI - POSITIVE", test_openai_normal_sequence),
        ("OpenAI - NEGATIVE", test_openai_interrupted_by_user),
        ("OpenAI - NEGATIVE", test_openai_partial_tool_responses),
        ("OpenAI - MIXED", test_openai_multiple_assistant_with_tools),
        ("OpenAI - POSITIVE", test_openai_no_tool_calls),

        # Anthropic Format Tests
        ("Anthropic - POSITIVE", test_anthropic_normal_sequence),
        ("Anthropic - NEGATIVE", test_anthropic_interrupted_sequence),
        ("Anthropic - NEGATIVE", test_anthropic_text_with_tool_use),

        # Integration Tests
        ("Integration", test_openai_to_langchain_conversion),
        ("Integration", test_anthropic_to_langchain_conversion),
    ]

    passed = 0
    failed = 0
    results = []

    for test_type, test in tests:
        try:
            if test():
                passed += 1
                results.append(("PASS", test_type))
            else:
                failed += 1
                results.append(("FAIL", test_type))
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            results.append(("ERROR", test_type))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total: {passed + failed} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print()

    # Show breakdown by test type
    print("BREAKDOWN:")
    print("-" * 70)
    for status, test_type in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {status:4} | {test_type}")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
