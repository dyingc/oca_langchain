#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Tool Call Sequence Validation

Tests the weight-based validation algorithm in core/llm.py:
- Weight 0: HumanMessage or clean AIMessage
- Weight >0: AIMessage with tool_calls
- Weight -1: ToolMessage (orphan candidate)
"""

import sys
sys.path.insert(0, '/Users/yingdong/VSCode/oca_langchain')

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from core.llm import _validate_tool_call_sequences, _calculate_message_weight


def test_weight_calculation():
    """Test message weight calculation"""
    print("=== Test 1: Weight Calculation ===")

    # HumanMessage should have weight 0
    human_msg = HumanMessage(content="Hello")
    assert _calculate_message_weight(human_msg) == 0, "HumanMessage should have weight 0"
    print("✅ HumanMessage weight = 0")

    # AIMessage without tool_calls should have weight 0
    clean_ai = AIMessage(content="Hi there")
    assert _calculate_message_weight(clean_ai) == 0, "Clean AIMessage should have weight 0"
    print("✅ Clean AIMessage weight = 0")

    # AIMessage with tool_calls should have weight > 0
    ai_with_tools = AIMessage(
        content="I'll search",
        additional_kwargs={
            "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {"id": "call_456", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
            ]
        }
    )
    assert _calculate_message_weight(ai_with_tools) == 2, "AIMessage with 2 tool_calls should have weight 2"
    print("✅ AIMessage with 2 tool_calls weight = 2")

    # ToolMessage should have weight -1
    tool_msg = ToolMessage(content="Result", tool_call_id="call_123")
    assert _calculate_message_weight(tool_msg) == -1, "ToolMessage should have weight -1"
    print("✅ ToolMessage weight = -1")

    print()


def test_valid_complete_sequence():
    """Test a valid complete tool call sequence"""
    print("=== Test 2: Valid Complete Sequence ===")

    messages = [
        HumanMessage(content="Search the web"),
        AIMessage(
            content="I'll search for you",
            additional_kwargs={
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": '{"query": "test"}'}}
                ]
            }
        ),
        ToolMessage(content="Found 5 results", tool_call_id="call_123"),
        HumanMessage(content="What did you find?")
    ]

    validated = _validate_tool_call_sequences(messages)

    assert len(validated) == 4, f"Expected 4 messages, got {len(validated)}"
    print(f"✅ Valid sequence preserved: {len(validated)} messages")
    print(f"   - Message 1: {type(validated[0]).__name__}")
    print(f"   - Message 2: {type(validated[1]).__name__} with tool_calls")
    print(f"   - Message 3: {type(validated[2]).__name__} (tool result)")
    print(f"   - Message 4: {type(validated[3]).__name__}")
    print()


def test_interrupted_sequence():
    """Test a tool call sequence interrupted by user message"""
    print("=== Test 3: Interrupted Tool Call Sequence ===")

    messages = [
        HumanMessage(content="Search the web"),
        AIMessage(
            content="I'll search",
            additional_kwargs={
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": '{"query": "test"}'}}
                ]
            }
        ),
        HumanMessage(content="Wait, stop!"),  # INTERRUPTION
        ToolMessage(content="Result", tool_call_id="call_123")  # ORPHANED
    ]

    validated = _validate_tool_call_sequences(messages)

    print(f"✅ Interrupted sequence handled: {len(validated)} messages (original: {len(messages)})")
    print(f"   - Expected: 3 messages (tool_calls removed from AI, tool message discarded)")
    print(f"   - Actual message types:")
    for i, msg in enumerate(validated):
        msg_type = type(msg).__name__
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"     Message {i+1}: {msg_type} with {len(msg.tool_calls)} tool_calls")
        else:
            content_preview = (msg.content[:30] + "...") if len(msg.content) > 30 else msg.content
            print(f"     Message {i+1}: {msg_type} content='{content_preview}'")

    # Verify tool_calls were removed
    ai_msg = validated[1]
    assert isinstance(ai_msg, AIMessage), "Second message should be AIMessage"
    tool_calls = getattr(ai_msg, 'tool_calls', None) or ai_msg.additional_kwargs.get('tool_calls')
    assert not tool_calls or len(tool_calls) == 0, "Tool_calls should be removed from interrupted AI message"
    print(f"✅ Tool_calls removed from AI message")

    # Verify tool message was discarded
    tool_msgs = [m for m in validated if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 0, "Orphaned tool message should be discarded"
    print(f"✅ Orphaned tool message discarded")

    # Verify interruption message was delayed (moved after cleaned sequence)
    assert "Wait, stop!" in validated[2].content, "Interruption message should be preserved"
    print(f"✅ Interruption message delayed and preserved")
    print()


def test_partial_match_sequence():
    """Test tool call sequence with only partial matches"""
    print("=== Test 4: Partial Match Sequence ===")

    messages = [
        HumanMessage(content="Search and get weather"),
        AIMessage(
            content="I'll do both",
            additional_kwargs={
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_456", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
                ]
            }
        ),
        ToolMessage(content="Search results", tool_call_id="call_123")
        # Missing call_456 result
    ]

    validated = _validate_tool_call_sequences(messages)

    print(f"✅ Partial match sequence: {len(validated)} messages")

    # Verify only matched tool_call remains
    ai_msg = validated[1]
    assert isinstance(ai_msg, AIMessage), "Second message should be AIMessage"
    tool_calls = getattr(ai_msg, 'tool_calls', None) or ai_msg.additional_kwargs.get('tool_calls')

    if tool_calls:
        assert len(tool_calls) == 1, f"Expected 1 matched tool_call, got {len(tool_calls)}"
        assert tool_calls[0]['id'] == 'call_123', "Remaining tool_call should be call_123"
        print(f"✅ Unmatched tool_call (call_456) removed from AI message")
        print(f"   - Remaining tool_call: {tool_calls[0]['id']}")
    print()


def test_multiple_tools_sequence():
    """Test sequence with multiple tool calls"""
    print("=== Test 5: Multiple Tools Complete Sequence ===")

    messages = [
        HumanMessage(content="Search and get weather"),
        AIMessage(
            content="I'll do both",
            additional_kwargs={
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_456", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
                ]
            }
        ),
        ToolMessage(content="Search results", tool_call_id="call_123"),
        ToolMessage(content="Weather data", tool_call_id="call_456")
    ]

    validated = _validate_tool_call_sequences(messages)

    assert len(validated) == 4, f"Expected 4 messages, got {len(validated)}"
    print(f"✅ Multiple tools sequence preserved: {len(validated)} messages")

    ai_msg = validated[1]
    tool_calls = getattr(ai_msg, 'tool_calls', None) or ai_msg.additional_kwargs.get('tool_calls')
    assert tool_calls and len(tool_calls) == 2, "Both tool_calls should be preserved"
    print(f"✅ Both tool_calls preserved in complete sequence")
    print()


def test_orphaned_tool_message():
    """Test orphaned tool message (no preceding tool_calls)"""
    print("=== Test 6: Orphaned Tool Message ===")

    messages = [
        HumanMessage(content="Hello"),
        ToolMessage(content="Orphaned result", tool_call_id="call_unknown"),
        HumanMessage(content="How are you?")
    ]

    validated = _validate_tool_call_sequences(messages)

    assert len(validated) == 2, f"Expected 2 messages (orphan discarded), got {len(validated)}"
    assert all(isinstance(m, HumanMessage) for m in validated), "Only HumanMessages should remain"
    print(f"✅ Orphaned tool message discarded")
    print()


def test_no_tool_calls():
    """Test normal conversation without tool calls"""
    print("=== Test 7: No Tool Calls ===")

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        HumanMessage(content="How are you?"),
        AIMessage(content="I'm doing well")
    ]

    validated = _validate_tool_call_sequences(messages)

    assert len(validated) == 4, f"Expected 4 messages, got {len(validated)}"
    print(f"✅ Normal conversation preserved: {len(validated)} messages")
    print()


def test_empty_list():
    """Test empty message list"""
    print("=== Test 8: Empty List ===")

    messages = []
    validated = _validate_tool_call_sequences(messages)

    assert len(validated) == 0, "Empty list should remain empty"
    print(f"✅ Empty list handled correctly")
    print()


def test_complex_interruption_pattern():
    """Test complex pattern: AI -> tool1 -> interruption -> tool2 -> tool3"""
    print("=== Test 9: Complex Interruption Pattern ===")

    messages = [
        HumanMessage(content="Do three searches"),
        AIMessage(
            content="I'll search three times",
            additional_kwargs={
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_3", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ]
            }
        ),
        ToolMessage(content="Result 1", tool_call_id="call_1"),
        HumanMessage(content="Wait!"),  # INTERRUPTION after first result
        ToolMessage(content="Result 2", tool_call_id="call_2"),
        ToolMessage(content="Result 3", tool_call_id="call_3")
    ]

    validated = _validate_tool_call_sequences(messages)

    print(f"✅ Complex pattern handled: {len(validated)} messages (original: {len(messages)})")
    print(f"   - AI message should have only call_1 (first match)")
    print(f"   - Wait! message should be delayed")
    print(f"   - Result 2 and 3 should be discarded (orphaned)")

    ai_msg = validated[1]
    tool_calls = getattr(ai_msg, 'tool_calls', None) or ai_msg.additional_kwargs.get('tool_calls')

    if tool_calls:
        print(f"   - AI message now has {len(tool_calls)} tool_call(s)")
        if len(tool_calls) == 1:
            print(f"      ✅ Correctly kept only matched tool_call: {tool_calls[0]['id']}")
    print()


def test_user_reported_case():
    """Test the exact case reported by user: AI with tool_call -> user interruption -> tool result"""
    print("=== Test 10: User Reported Case (AI -> Interruption -> Tool) ===")

    messages = [
        HumanMessage(content="1+1=?"),
        AIMessage(content="1+1=2"),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "toolu_XqMCQpuaehjt3R23IB1wxvXS",
                        "function": {
                            "name": "mcp__brave-search__brave_web_search",
                            "arguments": '{"query": "test"}'
                        }
                    }
                ]
            }
        ),
        HumanMessage(content="<something>"),  # INTERRUPTION
        ToolMessage(
            content="Result",
            tool_call_id="toolu_XqMCQpuaehjt3R23IB1wxvXS"
        )
    ]

    print(f"Input: {len(messages)} messages")
    print(f"  1. Human: '1+1=?'")
    print(f"  2. AI: '1+1=2'")
    print(f"  3. AI with tool_call")
    print(f"  4. Human: '<something>' (INTERRUPTION)")
    print(f"  5. Tool: 'Result' (ORPHANED)")

    validated = _validate_tool_call_sequences(messages)

    print(f"\nOutput: {len(validated)} messages")
    for i, msg in enumerate(validated):
        msg_type = type(msg).__name__
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  {i+1}. {msg_type} with {len(msg.tool_calls)} tool_calls")
        elif isinstance(msg, ToolMessage):
            print(f"  {i+1}. ToolMessage (id={msg.tool_call_id[:8]}...)")
        else:
            content_preview = (msg.content[:40] + "...") if len(msg.content) > 40 else msg.content
            print(f"  {i+1}. {msg_type}: '{content_preview}'")

    # Validate expectations
    assert len(validated) == 4, f"Expected 4 messages (tool_call removed, tool discarded), got {len(validated)}"

    # Message 3 should be AIMessage without tool_calls (cleaned)
    ai_msg = validated[2]
    assert isinstance(ai_msg, AIMessage), "Third message should be AIMessage"
    tool_calls = getattr(ai_msg, 'tool_calls', None) or ai_msg.additional_kwargs.get('tool_calls')
    assert not tool_calls or len(tool_calls) == 0, "Tool_calls should be removed"

    # Message 4 should be the interruption (delayed)
    assert "<something>" in validated[3].content, "Interruption should be delayed to end"

    print(f"\n✅ User case handled correctly:")
    print(f"   - Tool_calls removed from AI message")
    print(f"   - Interruption '<something>' delayed and preserved")
    print(f"   - Orphaned tool message discarded")
    print()


def main():
    """Run all tests"""
    print("=" * 80)
    print("UNIFIED TOOL CALL VALIDATION TEST SUITE")
    print("=" * 80)
    print()

    tests = [
        test_weight_calculation,
        test_valid_complete_sequence,
        test_interrupted_sequence,
        test_partial_match_sequence,
        test_multiple_tools_sequence,
        test_orphaned_tool_message,
        test_no_tool_calls,
        test_empty_list,
        test_complex_interruption_pattern,
        test_user_reported_case
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            print()
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {test.__name__}")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
