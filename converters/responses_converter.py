"""
OpenAI Response API Converter

This module provides conversion functions between OpenAI's Response API format
and LangChain's message format, enabling seamless integration with the OCAChatModel.
"""

import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    SystemMessage
)

logger = logging.getLogger(__name__)

from models.responses_types import (
    ResponseRequest,
    Response,
    InputItem,
    EasyInputMessage,
    InputMessage,
    FunctionCall,
    FunctionCallOutput,
    OutputItem,
    OutputMessage,
    OutputFunctionCall,
    OutputContentText,
    OutputContentRefusal,
    ResponseUsage,
    ResponseStatus,
    Tool,
    FunctionTool,
    generate_response_id,
    generate_item_id,
)


def response_input_to_langchain_messages(
    input_data: Optional[Union[str, List[InputItem]]],
    instructions: Optional[str] = None
) -> List[BaseMessage]:
    """
    Convert Response API input format to LangChain messages.

    Args:
        input_data: Response API input (string or list of input items)
        instructions: System/developer instructions

    Returns:
        List of LangChain BaseMessage objects

    Example:
        Response API input:
        "What is the weather?"

        → LangChain:
        [HumanMessage(content="What is the weather?")]

        ---

        Response API input:
        [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "Let me check."},
            {"type": "function_call_output", "call_id": "call_123", "output": "Sunny"}
        ]

        → LangChain:
        [
            HumanMessage(content="What is the weather?"),
            AIMessage(content="Let me check."),
            ToolMessage(content="Sunny", tool_call_id="call_123")
        ]
    """
    lc_messages: List[BaseMessage] = []

    # Add system instructions if provided
    if instructions:
        lc_messages.append(SystemMessage(content=instructions))

    # Handle None or empty input
    if input_data is None:
        return lc_messages

    # Handle string input (simple user message)
    if isinstance(input_data, str):
        lc_messages.append(HumanMessage(content=input_data))
        return lc_messages

    # Handle list of input items
    if not isinstance(input_data, list):
        # Unknown format, try to convert to string
        lc_messages.append(HumanMessage(content=str(input_data)))
        return lc_messages

    for item in input_data:
        # Handle dict items (Pydantic models are dicts when parsed from JSON)
        if isinstance(item, dict):
            item_type = item.get("type", "message")

            if item_type == "message":
                # Message item
                role = item.get("role", "user")
                content = item.get("content", "")

                # Handle string content
                if isinstance(content, str):
                    if role == "user":
                        lc_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    elif role in ("system", "developer"):
                        lc_messages.append(SystemMessage(content=content))
                # Handle structured content
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") in ("input_text", "output_text"):
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "text":
                                text_parts.append(block.get("text", ""))

                    combined_text = "\n".join(text_parts)
                    # Skip empty assistant messages - they will be created by function_call items
                    if role == "assistant" and not combined_text:
                        continue
                    if role == "user":
                        lc_messages.append(HumanMessage(content=combined_text))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=combined_text))
                    elif role in ("system", "developer"):
                        lc_messages.append(SystemMessage(content=combined_text))

            elif item_type == "function_call":
                # Function call from assistant
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                name = item.get("name", "")
                arguments = item.get("arguments", "{}")

                # Handle empty function name - try to infer from arguments
                if not name or not name.strip():
                    # Try to infer the function name from arguments
                    try:
                        args_obj = json.loads(arguments) if isinstance(arguments, str) else arguments
                        if isinstance(args_obj, dict):
                            # Known argument patterns
                            if "cmd" in args_obj:
                                name = "exec_command"
                                logger.info(f"[CONVERTER] Inferred function name 'exec_command' from arguments for call_id={call_id}")
                            elif "session_id" in args_obj and "chars" in args_obj:
                                name = "write_stdin"
                                logger.info(f"[CONVERTER] Inferred function name 'write_stdin' from arguments for call_id={call_id}")
                            elif "plan" in args_obj:
                                name = "update_plan"
                                logger.info(f"[CONVERTER] Inferred function name 'update_plan' from arguments for call_id={call_id}")
                            elif "questions" in args_obj:
                                name = "request_user_input"
                                logger.info(f"[CONVERTER] Inferred function name 'request_user_input' from arguments for call_id={call_id}")
                            elif "path" in args_obj:
                                name = "view_image"
                                logger.info(f"[CONVERTER] Inferred function name 'view_image' from arguments for call_id={call_id}")
                    except (json.JSONDecodeError, TypeError):
                        pass

                    # If still empty, skip this invalid function call
                    if not name or not name.strip():
                        logger.warning(f"[CONVERTER] Skipping function_call with empty name and uninferrable tool: call_id={call_id}, arguments_preview={str(arguments)[:100]}")
                        continue

                # Ensure arguments is a string
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)

                tool_call = {
                    "type": "function",
                    "id": call_id,
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }

                lc_messages.append(
                    AIMessage(content="", additional_kwargs={"tool_calls": [tool_call]})
                )

            elif item_type == "function_call_output":
                # Function call result
                call_id = item.get("call_id", "")
                output = item.get("output", "")

                # Ensure output is a string
                if isinstance(output, dict):
                    output = json.dumps(output)
                elif not isinstance(output, str):
                    output = str(output)

                lc_messages.append(
                    ToolMessage(content=output, tool_call_id=call_id)
                )

        # Handle Pydantic model items
        elif hasattr(item, "type"):
            item_type = item.type

            if item_type == "message":
                role = getattr(item, "role", "user")
                content = getattr(item, "content", "")

                # Handle string content
                if isinstance(content, str):
                    if role == "user":
                        lc_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                    elif role in ("system", "developer"):
                        lc_messages.append(SystemMessage(content=content))
                # Handle structured content (list of content blocks)
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") in ("input_text", "output_text"):
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        elif hasattr(block, "type"):
                            # Pydantic content block
                            if block.type in ("input_text", "output_text"):
                                text_parts.append(getattr(block, "text", ""))
                            elif block.type == "text":
                                text_parts.append(getattr(block, "text", ""))

                    combined_text = "\n".join(text_parts)
                    # Skip empty assistant messages
                    if role == "assistant" and not combined_text:
                        continue
                    if role == "user":
                        lc_messages.append(HumanMessage(content=combined_text))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=combined_text))
                    elif role in ("system", "developer"):
                        lc_messages.append(SystemMessage(content=combined_text))

            elif item_type == "function_call":
                call_id = getattr(item, "call_id", None) or getattr(item, "id", None) or f"call_{uuid.uuid4().hex[:24]}"
                name = getattr(item, "name", "")
                arguments = getattr(item, "arguments", "{}")

                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)

                # Handle empty function name - try to infer from arguments
                if not name or not name.strip():
                    try:
                        args_obj = json.loads(arguments) if isinstance(arguments, str) else arguments
                        if isinstance(args_obj, dict):
                            if "cmd" in args_obj:
                                name = "exec_command"
                            elif "session_id" in args_obj and "chars" in args_obj:
                                name = "write_stdin"
                            elif "plan" in args_obj:
                                name = "update_plan"
                            elif "questions" in args_obj:
                                name = "request_user_input"
                            elif "path" in args_obj and "image" not in str(args_obj).lower():
                                name = "view_image"
                    except (json.JSONDecodeError, TypeError):
                        pass

                    if not name or not name.strip():
                        logger.warning(f"[CONVERTER] Skipping function_call with empty name: call_id={call_id}, arguments_preview={str(arguments)[:100]}")
                        continue

                tool_call = {
                    "type": "function",
                    "id": call_id,
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }

                lc_messages.append(
                    AIMessage(content="", additional_kwargs={"tool_calls": [tool_call]})
                )

            elif item_type == "function_call_output":
                call_id = getattr(item, "call_id", "")
                output = getattr(item, "output", "")

                if isinstance(output, dict):
                    output = json.dumps(output)
                elif not isinstance(output, str):
                    output = str(output)

                lc_messages.append(
                    ToolMessage(content=output, tool_call_id=call_id)
                )

    return lc_messages


def response_tools_to_openai_tools(
    tools: Optional[List[Tool]]
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Response API tools to OpenAI format.

    Args:
        tools: List of Response API Tool objects

    Returns:
        List of OpenAI-formatted tool definitions
    """
    if not tools:
        return None

    def ensure_valid_parameters(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ensure parameters is a valid JSON Schema object according to OpenAI specs.

        OpenAI requires:
        - type: "object"
        - properties: {} (can be empty for no-arg functions)
        - required: [] (can be empty if no required params)
        - additionalProperties: false (required for strict mode)

        Reference: https://platform.openai.com/docs/guides/function-calling
        """
        if not params or not isinstance(params, dict):
            # No parameters provided - create schema for no-arg function
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }

        # Check if it's just an empty object {} or missing required fields
        result = dict(params)  # Make a copy to avoid modifying original

        if result.get("type") != "object":
            result["type"] = "object"
        if "properties" not in result:
            result["properties"] = {}
        if "required" not in result:
            # If no required specified, derive from properties keys
            # or default to empty array (all optional)
            result["required"] = []
        if "additionalProperties" not in result:
            # Set to False for strict mode compatibility
            result["additionalProperties"] = False

        return result

    openai_tools = []
    for i, tool in enumerate(tools):
        # Handle dict format
        if isinstance(tool, dict):
            tool_type = tool.get("type", "function")

            if tool_type == "function":
                # Standard function type - validate parameters
                params = ensure_valid_parameters(tool.get("parameters"))
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": params
                    }
                }
                openai_tools.append(openai_tool)

            elif tool_type not in ("web_search", "file_search", "computer"):
                # Custom tool type - convert to function format
                # This handles non-standard tool types from clients like Codex
                if tool.get("name"):
                    # Get parameters, handling various field names
                    params = tool.get("parameters") or tool.get("input_schema")
                    params = ensure_valid_parameters(params)

                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": params
                        }
                    }
                    openai_tools.append(openai_tool)
            # Built-in tools (web_search, file_search, computer) are skipped
            # as they need special backend support

        # Handle Pydantic model format
        elif hasattr(tool, "type"):
            if tool.type == "function":
                # Validate parameters for standard function type
                params = ensure_valid_parameters(getattr(tool, "parameters", None))
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": params
                    }
                }
                openai_tools.append(openai_tool)
            elif tool.type not in ("web_search", "file_search", "computer"):
                # Custom tool type - convert to function format
                name = getattr(tool, "name", None)
                if name:
                    # Get parameters from either parameters or input_schema
                    params = getattr(tool, "parameters", None) or getattr(tool, "input_schema", None)
                    params = ensure_valid_parameters(params)

                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": getattr(tool, "description", ""),
                            "parameters": params
                        }
                    }
                    openai_tools.append(openai_tool)

    return openai_tools if openai_tools else None


def response_request_to_langchain(request: ResponseRequest) -> Dict[str, Any]:
    """
    Convert a ResponseRequest to LangChain-compatible parameters.

    Args:
        request: ResponseRequest object

    Returns:
        Dictionary with LangChain-compatible parameters
    """
    logger.info(f"[CONVERTER] response_request_to_langchain: input count={len(request.input) if request.input else 0}, instructions={'provided' if request.instructions else 'none'}")

    lc_messages = response_input_to_langchain_messages(
        request.input,
        request.instructions
    )

    logger.info(f"[CONVERTER] After response_input_to_langchain_messages: {len(lc_messages)} messages")
    for i, msg in enumerate(lc_messages):
        logger.info(f"[CONVERTER]   {i}: {type(msg).__name__}, content_preview={str(msg.content)[:50]}...")
    openai_tools = response_tools_to_openai_tools(request.tools)

    result = {
        "messages": lc_messages,
        "tools": openai_tools,
        "tool_choice": request.tool_choice,
        "temperature": request.temperature,
        "max_tokens": request.max_output_tokens,
    }

    return result


def langchain_to_response_output(
    lc_message: AIMessage,
    model: str,
    response_id: Optional[str] = None,
    previous_response_id: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0
) -> Response:
    """
    Convert a LangChain AIMessage to a Response API Response.

    Args:
        lc_message: LangChain AIMessage object
        model: Model identifier
        response_id: Optional response ID (generated if not provided)
        previous_response_id: Optional previous response ID for stateful conversations
        input_tokens: Input token count
        output_tokens: Output token count

    Returns:
        Response object
    """
    output_items: List[OutputItem] = []

    # Create output message with text content
    text_content = lc_message.content or ""
    content_blocks = []

    if text_content:
        content_blocks.append(
            OutputContentText(type="output_text", text=text_content)
        )

    # Check for tool_calls
    tool_calls = lc_message.additional_kwargs.get("tool_calls")
    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}")
                name = function.get("name", "")
                arguments = function.get("arguments", "{}")

                # Create function call output item
                output_items.append(
                    OutputFunctionCall(
                        id=generate_item_id("fc"),
                        type="function_call",
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        status="completed"
                    )
                )

    # Create output message if there's text content
    if content_blocks:
        output_message = OutputMessage(
            id=generate_item_id("msg"),
            type="message",
            role="assistant",
            status="completed",
            content=content_blocks
        )
        output_items.insert(0, output_message)

    # If no output items at all, create an empty message
    if not output_items:
        output_items.append(
            OutputMessage(
                id=generate_item_id("msg"),
                type="message",
                role="assistant",
                status="completed",
                content=[]
            )
        )

    # Create usage
    usage = ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens
    )

    # Create response
    response = Response(
        id=response_id or generate_response_id(),
        model=model,
        output=output_items,
        status=ResponseStatus.COMPLETED,
        usage=usage,
        previous_response_id=previous_response_id
    )

    return response


def create_response_error(
    message: str,
    error_type: str = "server_error",
    model: Optional[str] = None,
    response_id: Optional[str] = None
) -> Response:
    """
    Create a Response API error response.

    Args:
        message: Error message
        error_type: Error type
        model: Optional model identifier
        response_id: Optional response ID

    Returns:
        Response object with error status
    """
    return Response(
        id=response_id or generate_response_id(),
        model=model or "unknown",
        output=[],
        status=ResponseStatus.FAILED,
        error={
            "type": error_type,
            "message": message
        }
    )


# --- Streaming Helpers ---

def create_response_created_event(
    response_id: str,
    model: str,
    previous_response_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a response.created event."""
    return {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "model": model,
            "status": "in_progress",
            "output": [],
            "previous_response_id": previous_response_id
        }
    }


def create_output_item_added_event(
    output_index: int,
    item_type: str,
    item_id: str
) -> Dict[str, Any]:
    """Create a response.output_item.added event."""
    item = {
        "id": item_id,
        "type": item_type
    }

    if item_type == "message":
        item["role"] = "assistant"
        item["content"] = []
        item["status"] = "in_progress"
    elif item_type == "function_call":
        item["call_id"] = f"call_{uuid.uuid4().hex[:24]}"
        item["name"] = ""
        item["arguments"] = ""
        item["status"] = "in_progress"

    return {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": item
    }


def create_output_text_delta_event(
    output_index: int,
    content_index: int,
    delta: str
) -> Dict[str, Any]:
    """Create a response.output_text.delta event."""
    return {
        "type": "response.output_text.delta",
        "output_index": output_index,
        "content_index": content_index,
        "delta": delta
    }


def create_function_call_arguments_delta_event(
    output_index: int,
    call_id: str,
    delta: str
) -> Dict[str, Any]:
    """Create a response.function_call_arguments.delta event."""
    return {
        "type": "response.function_call_arguments.delta",
        "output_index": output_index,
        "call_id": call_id,
        "delta": delta
    }


def create_output_item_done_event(
    output_index: int,
    item: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a response.output_item.done event."""
    return {
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": item
    }


def create_response_completed_event(
    response_id: str,
    model: str,
    output: List[Dict[str, Any]],
    usage: Dict[str, int],
    previous_response_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a response.completed event."""
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "model": model,
            "status": "completed",
            "output": output,
            "usage": usage,
            "previous_response_id": previous_response_id
        }
    }


def format_stream_event(event: Dict[str, Any]) -> str:
    """Format a stream event as SSE data."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
