"""
Models package for API type definitions.

This package contains Pydantic models for different API formats:
- OpenAI compatible models (defined in api.py)
- Anthropic compatible models (in anthropic_types.py)
- OpenAI Response API models (in responses_types.py)
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

from models.responses_types import (
    # Request/Response
    ResponseRequest,
    Response,
    ResponseStatus,
    ResponseUsage,
    ResponseDeleted,
    ResponseList,
    # Input items
    InputItem,
    EasyInputMessage,
    InputMessage,
    FunctionCall,
    FunctionCallOutput,
    InputContentText,
    InputContentImage,
    # Output items
    OutputItem,
    OutputMessage,
    OutputFunctionCall,
    OutputReasoning,
    OutputContentText,
    OutputContentRefusal,
    # Tools
    Tool,
    FunctionTool,
    WebSearchTool,
    FileSearchTool,
    ComputerUseTool,
    CustomTool,
    # Stream events
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseErrorEvent,
    # Helpers
    generate_response_id,
    generate_item_id,
)

__all__ = [
    # Anthropic types
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
    # Response API types
    "ResponseRequest",
    "Response",
    "ResponseStatus",
    "ResponseUsage",
    "ResponseDeleted",
    "ResponseList",
    "InputItem",
    "EasyInputMessage",
    "InputMessage",
    "FunctionCall",
    "FunctionCallOutput",
    "InputContentText",
    "InputContentImage",
    "OutputItem",
    "OutputMessage",
    "OutputFunctionCall",
    "OutputReasoning",
    "OutputContentText",
    "OutputContentRefusal",
    "Tool",
    "FunctionTool",
    "WebSearchTool",
    "FileSearchTool",
    "ComputerUseTool",
    "CustomTool",
    # Stream events
    "ResponseCreatedEvent",
    "ResponseInProgressEvent",
    "ResponseOutputItemAddedEvent",
    "ResponseOutputTextDeltaEvent",
    "ResponseOutputTextDoneEvent",
    "ResponseFunctionCallArgumentsDeltaEvent",
    "ResponseFunctionCallArgumentsDoneEvent",
    "ResponseOutputItemDoneEvent",
    "ResponseCompletedEvent",
    "ResponseFailedEvent",
    "ResponseErrorEvent",
    # Helpers
    "generate_response_id",
    "generate_item_id",
]
