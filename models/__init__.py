"""
Models package for API type definitions.

This package contains Pydantic models for different API formats:
- OpenAI compatible models (defined in api.py)
- Anthropic compatible models (in anthropic_types.py)
"""

from models.anthropic_types import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicToolDefinition,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicUsage,
    AnthropicStreamMessageStart,
    AnthropicStreamContentBlockStart,
    AnthropicStreamContentBlockDelta,
    AnthropicStreamContentBlockStop,
    AnthropicStreamMessageDelta,
    AnthropicStreamMessageStop,
    AnthropicErrorResponse,
)

__all__ = [
    "AnthropicContentBlock",
    "AnthropicMessage",
    "AnthropicToolDefinition",
    "AnthropicRequest",
    "AnthropicResponse",
    "AnthropicUsage",
    "AnthropicStreamMessageStart",
    "AnthropicStreamContentBlockStart",
    "AnthropicStreamContentBlockDelta",
    "AnthropicStreamContentBlockStop",
    "AnthropicStreamMessageDelta",
    "AnthropicStreamMessageStop",
    "AnthropicErrorResponse",
]
