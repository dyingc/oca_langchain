"""
Anthropic API Pydantic Models for Messages API

This module defines Pydantic models for Anthropic's Messages API,
compatible with the official Anthropic API specification.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


class AnthropicContentBlock(BaseModel):
    """
    Represents a content block in Anthropic's message format.

    Content blocks can be of different types:
    - text: Plain text content
    - tool_use: Tool/function call request
    - tool_result: Tool/function call result
    - image: Image content (future support)
    """
    type: str  # "text" | "tool_use" | "tool_result" | "image"
    text: Optional[str] = None
    id: Optional[str] = None  # For tool_use blocks
    name: Optional[str] = None  # For tool_use blocks
    input: Optional[Dict[str, Any]] = None  # For tool_use blocks
    tool_use_id: Optional[str] = None  # For tool_result blocks
    content: Optional[Union[str, List[Dict]]] = None  # For tool_result blocks
    source: Optional[Dict[str, Any]] = None  # For image blocks (type, media_type, data)


class AnthropicMessage(BaseModel):
    """
    Represents a message in Anthropic's format.

    Messages have a role (user or assistant) and content,
    which can be a string or a list of content blocks.
    """
    role: str  # "user" | "assistant"
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicToolDefinition(BaseModel):
    """
    Represents a tool definition in Anthropic's format.

    Anthropic uses a simplified tool definition format compared to OpenAI.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON Schema for the tool parameters


class AnthropicRequest(BaseModel):
    """
    Represents a request to Anthropic's /v1/messages endpoint.

    Required fields:
    - model: The model identifier
    - max_tokens: Maximum tokens to generate
    - messages: List of conversation messages

    Optional fields:
    - temperature: Sampling temperature (0-1)
    - tools: List of available tools
    - stream: Whether to use streaming response
    - top_k, top_p: Additional sampling parameters
    """
    model: str
    max_tokens: int
    messages: List[AnthropicMessage]
    temperature: Optional[float] = None
    tools: Optional[List[AnthropicToolDefinition]] = None
    stream: Optional[bool] = False
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None


class AnthropicUsage(BaseModel):
    """
    Represents token usage information in Anthropic's response.
    """
    input_tokens: int
    output_tokens: int


class AnthropicResponse(BaseModel):
    """
    Represents a response from Anthropic's /v1/messages endpoint.

    The response contains:
    - id: Unique message identifier
    - type: Always "message"
    - role: Always "assistant"
    - content: List of content blocks (text or tool_use)
    - model: The model that generated the response
    - stop_reason: Why the generation stopped
    - usage: Token usage statistics
    """
    id: str = Field(default_factory=lambda: f"msg_{''.join(__import__('random').choices('abcdefghijklmnopqrstuvwxyz0123456789', k=24))}")
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: str = "end_turn"
    usage: AnthropicUsage


class AnthropicStreamMessageStart(BaseModel):
    """
    Represents the message_start event in streaming.
    """
    type: str = "message_start"
    message: AnthropicResponse


class AnthropicStreamContentBlockStart(BaseModel):
    """
    Represents the content_block_start event in streaming.
    """
    type: str = "content_block_start"
    index: int
    content_block: AnthropicContentBlock


class AnthropicStreamContentBlockDelta(BaseModel):
    """
    Represents the content_block_delta event in streaming.
    """
    type: str = "content_block_delta"
    index: int
    delta: Dict[str, Any]  # Contains "type" and "text" or "partial_json"


class AnthropicStreamContentBlockStop(BaseModel):
    """
    Represents the content_block_stop event in streaming.
    """
    type: str = "content_block_stop"
    index: int


class AnthropicStreamMessageDelta(BaseModel):
    """
    Represents the message_delta event in streaming.
    """
    type: str = "message_delta"
    delta: Dict[str, Any]  # Contains "stop_reason" and "stop_sequence"
    usage: AnthropicUsage


class AnthropicStreamMessageStop(BaseModel):
    """
    Represents the message_stop event in streaming.
    """
    type: str = "message_stop"


class AnthropicErrorResponse(BaseModel):
    """
    Represents an error response in Anthropic's format.

    Error responses follow the structure:
    {
        "type": "error",
        "error": {
            "type": "invalid_request_error" | "authentication_error" | "not_found_error" | "rate_limit_error" | "api_error",
            "message": "Human-readable error message"
        }
    }
    """
    type: str = "error"
    error: Dict[str, str]  # Contains "type" and "message"
