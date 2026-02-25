"""
OpenAI Response API Pydantic Models

This module defines Pydantic models for OpenAI's Response API,
compatible with the official OpenAI Response API specification.

Reference: https://platform.openai.com/docs/api-reference/responses
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
import time
import random
import string


# --- ID Generation Helpers ---

def generate_response_id() -> str:
    """Generate a unique response ID like resp_xxx"""
    return f"resp_{''.join(random.choices(string.ascii_lowercase + string.digits, k=24))}"


def generate_item_id(prefix: str = "msg") -> str:
    """Generate a unique item ID with given prefix (msg, fc, rs, etc.)"""
    return f"{prefix}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=24))}"


# --- Input Item Types ---

class InputContentText(BaseModel):
    """Text content in an input message"""
    type: Literal["input_text", "output_text"] = "input_text"  # Accept both input_text and output_text
    text: str


class InputContentImage(BaseModel):
    """Image content in an input message"""
    type: Literal["input_image"] = "input_image"
    image_url: Optional[str] = None
    file_id: Optional[str] = None
    detail: Optional[str] = "auto"


class InputContentRefusal(BaseModel):
    """Refusal content"""
    type: Literal["refusal"] = "refusal"
    refusal: str


InputContent = Union[InputContentText, InputContentImage, InputContentRefusal]


class EasyInputMessage(BaseModel):
    """
    Simple message format with role and string content.
    Used when input is just a string instead of structured items.
    """
    role: Literal["user", "assistant", "system", "developer"]
    content: str
    type: Literal["message"] = "message"


class InputMessage(BaseModel):
    """
    Full input message with structured content.
    """
    id: Optional[str] = None
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: Union[str, List[InputContent]]


class FunctionCall(BaseModel):
    """
    A function call output item from the model.
    """
    id: Optional[str] = None
    type: Literal["function_call"] = "function_call"
    call_id: Optional[str] = None  # ID to reference this call in function_call_output
    name: str
    arguments: str  # JSON string of arguments
    status: Optional[str] = "completed"


class FunctionCallOutput(BaseModel):
    """
    A function call result to provide back to the model.
    """
    id: Optional[str] = None
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str  # References the function_call's call_id
    output: str  # Result of the function call


class ReasoningItem(BaseModel):
    """
    A reasoning item from reasoning models (e.g., o1).
    """
    id: Optional[str] = None
    type: Literal["reasoning"] = "reasoning"
    summary: Optional[List[Dict[str, Any]]] = None


# Input can be a string, EasyInputMessage, or structured InputItem
InputItem = Union[EasyInputMessage, InputMessage, FunctionCall, FunctionCallOutput, ReasoningItem]


# --- Tool Types ---

class FunctionTool(BaseModel):
    """Function tool definition"""
    type: Literal["function"] = "function"
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSON Schema
    strict: Optional[bool] = False


class WebSearchTool(BaseModel):
    """Built-in web search tool"""
    type: Literal["web_search"] = "web_search"
    search_context_size: Optional[str] = "medium"  # low, medium, high


class FileSearchTool(BaseModel):
    """Built-in file search tool"""
    type: Literal["file_search"] = "file_search"
    vector_store_ids: List[str]


class ComputerUseTool(BaseModel):
    """Computer use tool for agentic workflows"""
    type: Literal["computer"] = "computer"
    display_width: Optional[int] = 1024
    display_height: Optional[int] = 768
    environment: Optional[str] = "browser"  # browser, mac, windows, ubuntu


class CustomTool(BaseModel):
    """Generic tool for non-standard tool types"""
    type: str
    name: Optional[str] = None
    description: Optional[str] = None
    # Allow any additional fields
    model_config = ConfigDict(extra="allow")


Tool = Union[FunctionTool, WebSearchTool, FileSearchTool, ComputerUseTool, CustomTool]


# --- Text Format (for structured outputs) ---

class TextFormatJSONSchema(BaseModel):
    """JSON Schema format for structured outputs"""
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["json_schema"] = "json_schema"
    name: str
    schema_: Dict[str, Any] = Field(alias="schema")
    strict: Optional[bool] = False


class TextFormat(BaseModel):
    """Text format configuration for response"""
    format: Optional[Union[Literal["text", "json_object"], TextFormatJSONSchema]] = None


# --- Truncation Options ---

class TruncationOptions(BaseModel):
    """Options for handling truncated responses"""
    type: Optional[str] = "auto"  # auto, last_available
    max_tokens: Optional[int] = None


# --- Request Model ---

class ResponseRequest(BaseModel):
    """
    Request model for POST /v1/responses endpoint.

    Required fields:
    - model: The model identifier

    Optional fields:
    - input: String or array of input items (messages, function outputs, etc.)
    - instructions: System/developer instructions
    - tools: Available tools
    - tool_choice: How to select tools (auto, required, none, or specific)
    - response_format: Output format (text, json_object, or json_schema)
    - temperature: Sampling temperature
    - max_output_tokens: Maximum tokens in output
    - top_p: Nucleus sampling parameter
    - stream: Whether to stream the response
    - store: Whether to store the response for later retrieval
    - previous_response_id: ID of previous response for conversation continuity
    - truncation: How to handle truncated responses
    - metadata: Additional metadata to attach to response
    """
    model: str
    input: Optional[Union[str, List[InputItem]]] = None
    instructions: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    text: Optional[TextFormat] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    store: Optional[bool] = True
    previous_response_id: Optional[str] = None
    truncation: Optional[Union[str, TruncationOptions]] = "auto"
    metadata: Optional[Dict[str, Any]] = None

    # Legacy compatibility
    user: Optional[str] = None


# --- Output Item Types ---

class OutputContentText(BaseModel):
    """Text content in output"""
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: Optional[List[Dict[str, Any]]] = None


class OutputContentRefusal(BaseModel):
    """Refusal content in output"""
    type: Literal["refusal"] = "refusal"
    refusal: str


OutputContent = Union[OutputContentText, OutputContentRefusal]


class OutputMessage(BaseModel):
    """
    Output message from the model.
    """
    id: str = Field(default_factory=lambda: generate_item_id("msg"))
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: str = "completed"
    content: List[OutputContent] = []
    # For tool calls within message (alternative format)
    tool_calls: Optional[List[FunctionCall]] = None


class OutputFunctionCall(BaseModel):
    """
    Output function call item.
    """
    id: str = Field(default_factory=lambda: generate_item_id("fc"))
    type: Literal["function_call"] = "function_call"
    call_id: str = Field(default_factory=lambda: generate_item_id("call"))
    name: str
    arguments: str
    status: str = "completed"


class OutputReasoning(BaseModel):
    """
    Output reasoning item from reasoning models.
    """
    id: str = Field(default_factory=lambda: generate_item_id("rs"))
    type: Literal["reasoning"] = "reasoning"
    summary: List[Dict[str, Any]] = []


OutputItem = Union[OutputMessage, OutputFunctionCall, OutputReasoning]


# --- Usage ---

class ResponseUsage(BaseModel):
    """Token usage information"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Detailed token breakdown (optional)
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None


# --- Response Model ---

class ResponseStatus(str, Enum):
    """Status of a response"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Response(BaseModel):
    """
    Full response from the /v1/responses endpoint.
    """
    id: str = Field(default_factory=generate_response_id)
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    output: List[OutputItem] = []
    status: ResponseStatus = ResponseStatus.COMPLETED
    usage: Optional[ResponseUsage] = None
    # Stateful conversation
    previous_response_id: Optional[str] = None
    # Instructions used
    instructions: Optional[str] = None
    # Tools used
    tools: Optional[List[Tool]] = None
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    # Error info if failed
    error: Optional[Dict[str, Any]] = None


# --- Streaming Event Types ---

class ResponseStreamEvent(BaseModel):
    """Base class for stream events"""
    type: str


class ResponseCreatedEvent(ResponseStreamEvent):
    """Event: response.created"""
    type: Literal["response.created"] = "response.created"
    response: Response


class ResponseInProgressEvent(ResponseStreamEvent):
    """Event: response.in_progress"""
    type: Literal["response.in_progress"] = "response.in_progress"
    response: Response


class ResponseOutputItemAddedEvent(ResponseStreamEvent):
    """Event: response.output_item.added"""
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: OutputItem


class ResponseOutputTextDeltaEvent(ResponseStreamEvent):
    """Event: response.output_text.delta"""
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(ResponseStreamEvent):
    """Event: response.output_text.done"""
    type: Literal["response.output_text.done"] = "response.output_text.done"
    output_index: int
    content_index: int
    text: str


class ResponseFunctionCallArgumentsDeltaEvent(ResponseStreamEvent):
    """Event: response.function_call_arguments.delta"""
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    output_index: int
    call_id: str
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(ResponseStreamEvent):
    """Event: response.function_call_arguments.done"""
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    output_index: int
    call_id: str
    arguments: str


class ResponseOutputItemDoneEvent(ResponseStreamEvent):
    """Event: response.output_item.done"""
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: OutputItem


class ResponseCompletedEvent(ResponseStreamEvent):
    """Event: response.completed"""
    type: Literal["response.completed"] = "response.completed"
    response: Response


class ResponseFailedEvent(ResponseStreamEvent):
    """Event: response.failed"""
    type: Literal["response.failed"] = "response.failed"
    response: Response


class ResponseErrorEvent(ResponseStreamEvent):
    """Event: error"""
    type: Literal["error"] = "error"
    error: Dict[str, Any]


# Union of all stream events
ResponseStreamEventType = Union[
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
]


# --- Response List (for GET /v1/responses) ---

class ResponseList(BaseModel):
    """List of responses"""
    object: Literal["list"] = "list"
    data: List[Response] = []
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None


# --- Delete Response ---

class ResponseDeleted(BaseModel):
    """Response deletion confirmation"""
    id: str
    object: Literal["response.deleted"] = "response.deleted"
    deleted: bool = True
