"""
Test suite for unified tool call sequence validation.

Tests the weight-based validation algorithm in core/llm.py:
- Weight 0: HumanMessage or clean AIMessage
- Weight >0: AIMessage with tool_calls
- Weight -1: ToolMessage (orphan candidate)
"""

import pytest

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from core.llm import _validate_tool_call_sequences, _calculate_message_weight


class TestWeightCalculation:
    """Tests for message weight calculation."""

    def test_human_message_weight_zero(self):
        assert _calculate_message_weight(HumanMessage(content="Hello")) == 0

    def test_clean_ai_message_weight_zero(self):
        assert _calculate_message_weight(AIMessage(content="Hi there")) == 0

    def test_ai_with_tool_calls_weight(self):
        ai_msg = AIMessage(
            content="I'll search",
            additional_kwargs={
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_456", "type": "function", "function": {"name": "weather", "arguments": "{}"}},
                ]
            },
        )
        assert _calculate_message_weight(ai_msg) == 2

    def test_tool_message_weight_negative_one(self):
        assert _calculate_message_weight(ToolMessage(content="Result", tool_call_id="call_123")) == -1


class TestValidCompleteSequences:
    """Tests for valid, complete tool call sequences."""

    def test_single_tool_call(self):
        """Valid single tool call sequence should be preserved."""
        messages = [
            HumanMessage(content="Search the web"),
            AIMessage(
                content="I'll search for you",
                additional_kwargs={
                    "tool_calls": [
                        {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": '{"query": "test"}'}}
                    ]
                },
            ),
            ToolMessage(content="Found 5 results", tool_call_id="call_123"),
            HumanMessage(content="What did you find?"),
        ]

        validated = _validate_tool_call_sequences(messages)

        assert len(validated) == 4

    def test_multiple_tool_calls(self):
        """Valid multi-tool sequence should be preserved."""
        messages = [
            HumanMessage(content="Search and get weather"),
            AIMessage(
                content="I'll do both",
                additional_kwargs={
                    "tool_calls": [
                        {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                        {"id": "call_456", "type": "function", "function": {"name": "weather", "arguments": "{}"}},
                    ]
                },
            ),
            ToolMessage(content="Search results", tool_call_id="call_123"),
            ToolMessage(content="Weather data", tool_call_id="call_456"),
        ]

        validated = _validate_tool_call_sequences(messages)

        assert len(validated) == 4
        tool_calls = getattr(validated[1], "tool_calls", None) or validated[1].additional_kwargs.get("tool_calls")
        assert tool_calls and len(tool_calls) == 2

    def test_no_tool_calls(self):
        """Normal conversation without tool calls should be unchanged."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm doing well"),
        ]

        validated = _validate_tool_call_sequences(messages)

        assert len(validated) == 4

    def test_empty_list(self):
        """Empty message list should remain empty."""
        assert len(_validate_tool_call_sequences([])) == 0


class TestInterruptedSequences:
    """Tests for tool call sequences interrupted by user messages."""

    def test_simple_interruption(self):
        """User interruption should remove tool_calls and discard orphaned tool messages."""
        messages = [
            HumanMessage(content="Search the web"),
            AIMessage(
                content="I'll search",
                additional_kwargs={
                    "tool_calls": [
                        {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": '{"query": "test"}'}}
                    ]
                },
            ),
            HumanMessage(content="Wait, stop!"),
            ToolMessage(content="Result", tool_call_id="call_123"),
        ]

        validated = _validate_tool_call_sequences(messages)

        # Tool_calls should be removed from AI message
        ai_msg = validated[1]
        tool_calls = getattr(ai_msg, "tool_calls", None) or ai_msg.additional_kwargs.get("tool_calls")
        assert not tool_calls or len(tool_calls) == 0

        # Orphaned tool message should be discarded
        assert not any(isinstance(m, ToolMessage) for m in validated)

        # Interruption message should be preserved
        assert any("Wait, stop!" in m.content for m in validated if isinstance(m, HumanMessage))

    def test_partial_match(self):
        """Only matched tool_calls should be kept when sequence is incomplete."""
        messages = [
            HumanMessage(content="Search and get weather"),
            AIMessage(
                content="I'll do both",
                additional_kwargs={
                    "tool_calls": [
                        {"id": "call_123", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                        {"id": "call_456", "type": "function", "function": {"name": "weather", "arguments": "{}"}},
                    ]
                },
            ),
            ToolMessage(content="Search results", tool_call_id="call_123"),
            # Missing call_456 result
        ]

        validated = _validate_tool_call_sequences(messages)

        ai_msg = validated[1]
        tool_calls = getattr(ai_msg, "tool_calls", None) or ai_msg.additional_kwargs.get("tool_calls")
        if tool_calls:
            assert len(tool_calls) == 1
            assert tool_calls[0]["id"] == "call_123"

    def test_complex_interruption_pattern(self):
        """AI -> tool1 -> interruption -> tool2 -> tool3: only first result kept."""
        messages = [
            HumanMessage(content="Do three searches"),
            AIMessage(
                content="I'll search three times",
                additional_kwargs={
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                        {"id": "call_2", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                        {"id": "call_3", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    ]
                },
            ),
            ToolMessage(content="Result 1", tool_call_id="call_1"),
            HumanMessage(content="Wait!"),
            ToolMessage(content="Result 2", tool_call_id="call_2"),
            ToolMessage(content="Result 3", tool_call_id="call_3"),
        ]

        validated = _validate_tool_call_sequences(messages)

        ai_msg = validated[1]
        tool_calls = getattr(ai_msg, "tool_calls", None) or ai_msg.additional_kwargs.get("tool_calls")
        if tool_calls:
            assert len(tool_calls) == 1


class TestOrphanedMessages:
    """Tests for orphaned tool messages."""

    def test_orphaned_tool_message_discarded(self):
        """Tool message with no preceding tool_calls should be discarded."""
        messages = [
            HumanMessage(content="Hello"),
            ToolMessage(content="Orphaned result", tool_call_id="call_unknown"),
            HumanMessage(content="How are you?"),
        ]

        validated = _validate_tool_call_sequences(messages)

        assert len(validated) == 2
        assert all(isinstance(m, HumanMessage) for m in validated)


class TestUserReportedCase:
    """Test the exact case reported by user: AI -> Interruption -> Tool."""

    def test_user_reported_case(self):
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
                                "arguments": '{"query": "test"}',
                            },
                        }
                    ]
                },
            ),
            HumanMessage(content="<something>"),
            ToolMessage(content="Result", tool_call_id="toolu_XqMCQpuaehjt3R23IB1wxvXS"),
        ]

        validated = _validate_tool_call_sequences(messages)

        assert len(validated) == 4

        # AI message should have tool_calls removed
        ai_msg = validated[2]
        tool_calls = getattr(ai_msg, "tool_calls", None) or ai_msg.additional_kwargs.get("tool_calls")
        assert not tool_calls or len(tool_calls) == 0

        # Interruption should be preserved
        assert "<something>" in validated[3].content
