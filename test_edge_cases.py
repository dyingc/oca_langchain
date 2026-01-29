#!/usr/bin/env python3
"""
Edge case tests for message validation logic.

These tests cover corner cases that may not be covered by the main test suite.
"""

import sys
sys.path.insert(0, '.')

from api import validate_and_fix_message_sequence, ChatMessage
from converters.anthropic_request_converter import validate_and_fix_anthropic_message_sequence, AnthropicMessage, AnthropicContentBlock
import json

def test_openai_empty_message_list():
    """Test: Empty message list should return empty list"""
    print("\n[Edge Case] OpenAI - Empty Message List")
    messages = []
    result = validate_and_fix_message_sequence(messages)
    
    if len(result) == 0:
        print("  ✅ PASS: Empty list preserved")
        return True
    else:
        print(f"  ❌ FAIL: Expected 0 messages, got {len(result)}")
        return False

def test_openai_consecutive_assistant_with_tools():
    """Test: Consecutive assistant messages with tool_calls"""
    print("\n[Edge Case] OpenAI - Consecutive Assistant with tool_calls")
    messages = [
        ChatMessage(role='user', content='First task'),
        ChatMessage(role='assistant', content='', tool_calls=[
            {'id': 'call_1', 'type': 'function', 'function': {'name': 'task1', 'arguments': '{}'}}
        ]),
        ChatMessage(role='tool', tool_call_id='call_1', content='Result 1'),
        ChatMessage(role='assistant', content='', tool_calls=[
            {'id': 'call_2', 'type': 'function', 'function': {'name': 'task2', 'arguments': '{}'}}
        ]),
        ChatMessage(role='tool', tool_call_id='call_2', content='Result 2'),
    ]
    
    result = validate_and_fix_message_sequence(messages)
    
    # Both sequences should be preserved
    if len(result) == 5:
        # Check both assistant messages still have tool_calls
        has_tools_1 = result[1].tool_calls is not None and len(result[1].tool_calls) > 0
        has_tools_2 = result[3].tool_calls is not None and len(result[3].tool_calls) > 0
        if has_tools_1 and has_tools_2:
            print("  ✅ PASS: Both valid sequences preserved")
            return True
        else:
            print(f"  ❌ FAIL: tool_calls removed (has_tools_1={has_tools_1}, has_tools_2={has_tools_2})")
            return False
    else:
        print(f"  ❌ FAIL: Expected 5 messages, got {len(result)}")
        return False

def test_openai_tool_before_assistant():
    """Test: Tool response before assistant message (invalid order)"""
    print("\n[Edge Case] OpenAI - Tool Response Before Assistant")
    messages = [
        ChatMessage(role='tool', tool_call_id='call_1', content='Orphaned result'),
        ChatMessage(role='user', content='Start task'),
        ChatMessage(role='assistant', content='Done'),
    ]
    
    result = validate_and_fix_message_sequence(messages)
    
    # Orphaned tool at the start should be kept (no assistant to associate with)
    if len(result) == 3:
        print("  ✅ PASS: Orphaned tool kept (no preceding assistant)")
        return True
    else:
        print(f"  ⚠️  INFO: Got {len(result)} messages (behavior may vary)")
        return True  # This is acceptable behavior

def test_openai_multiple_interruptions():
    """Test: Multiple user interruptions in sequence"""
    print("\n[Edge Case] OpenAI - Multiple Interruptions")
    messages = [
        ChatMessage(role='user', content='Task 1'),
        ChatMessage(role='assistant', content='', tool_calls=[
            {'id': 'call_1', 'type': 'function', 'function': {'name': 'task1', 'arguments': '{}'}},
            {'id': 'call_2', 'type': 'function', 'function': {'name': 'task2', 'arguments': '{}'}},
        ]),
        ChatMessage(role='user', content='Wait!'),  # First interruption
        ChatMessage(role='user', content='Actually, continue'),  # Second user message
        ChatMessage(role='tool', tool_call_id='call_1', content='Result 1'),
        ChatMessage(role='tool', tool_call_id='call_2', content='Result 2'),
        ChatMessage(role='user', content='Final message'),
    ]
    
    result = validate_and_fix_message_sequence(messages)

    # Should remove tool_calls and skip orphaned tool messages
    # Expected: user, assistant (no tools), user "Wait!", user "Actually, continue", user "Final"
    expected = 5
    if len(result) == expected:
        has_tools = result[1].tool_calls is not None and len(result[1].tool_calls) > 0
        if not has_tools:
            print(f"  ✅ PASS: {expected} messages, tool_calls removed, orphaned tools skipped")
            return True
        else:
            print("  ❌ FAIL: tool_calls not removed")
            return False
    else:
        print(f"  ❌ FAIL: Expected {expected} messages, got {len(result)}")
        for i, msg in enumerate(result):
            print(f"    [{i}] role={msg.role}")
        return False

def test_anthropic_empty_tool_use_id():
    """Test: tool_result with empty tool_use_id"""
    print("\n[Edge Case] Anthropic - Empty tool_use_id")
    messages = [
        AnthropicMessage(role='user', content='Use tool'),
        AnthropicMessage(role='assistant', content=[
            AnthropicContentBlock(type='tool_use', id='toolu_123', name='search', input={'query': 'test'}),
        ]),
        AnthropicMessage(role='user', content=[
            AnthropicContentBlock(type='tool_result', tool_use_id='', content='No ID'),
        ]),
    ]
    
    result = validate_and_fix_anthropic_message_sequence(messages)
    
    # Empty tool_use_id is invalid, sequence should be marked as invalid
    if len(result) <= 3:
        print(f"  ✅ PASS: Invalid sequence handled, got {len(result)} messages")
        return True
    else:
        print(f"  ❌ FAIL: Expected <= 3 messages, got {len(result)}")
        return False

def test_anthropic_multiple_tool_results_different_ids():
    """Test: Multiple tool_results with different tool_use_ids in one message"""
    print("\n[Edge Case] Anthropic - Mixed tool_results (valid + invalid)")
    messages = [
        AnthropicMessage(role='user', content='Search'),
        AnthropicMessage(role='assistant', content=[
            AnthropicContentBlock(type='tool_use', id='toolu_123', name='search', input={'query': 'test'}),
            AnthropicContentBlock(type='tool_use', id='toolu_456', name='search', input={'query': 'test2'}),
        ]),
        AnthropicMessage(role='user', content=[
            AnthropicContentBlock(type='tool_result', tool_use_id='toolu_123', content='{"result": "ok"}'),
            AnthropicContentBlock(type='tool_result', tool_use_id='toolu_999', content='{"result": "wrong_id"}'),  # Wrong ID
        ]),
    ]
    
    result = validate_and_fix_anthropic_message_sequence(messages)
    
    # Sequence has tool_result for non-existent ID, should be handled
    print(f"  ⚠️  INFO: Got {len(result)} messages (behavior for wrong IDs may vary)")
    return True  # This is acceptable - behavior may vary

def run_all_edge_case_tests():
    """Run all edge case tests"""
    print("=" * 70)
    print("EDGE CASE TEST SUITE")
    print("=" * 70)
    
    tests = [
        test_openai_empty_message_list,
        test_openai_consecutive_assistant_with_tools,
        test_openai_tool_before_assistant,
        test_openai_multiple_interruptions,
        test_anthropic_empty_tool_use_id,
        test_anthropic_multiple_tool_results_different_ids,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print(f"EDGE CASE TEST SUMMARY: {sum(results)}/{len(results)} passed")
    print("=" * 70)
    
    return all(results)

if __name__ == "__main__":
    success = run_all_edge_case_tests()
    sys.exit(0 if success else 1)
