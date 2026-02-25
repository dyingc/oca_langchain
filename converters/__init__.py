"""
Converters package for API format transformations.

This package contains converters for transforming between different API formats:
- Anthropic ↔ LangChain
- OpenAI Response API ↔ LangChain
"""

from converters.anthropic_request_converter import (
    anthropic_to_langchain,
    anthropic_to_langchain_messages,
    langchain_to_anthropic_response,
    create_anthropic_error_response,
)

from converters.responses_converter import (
    response_input_to_langchain_messages,
    response_request_to_langchain,
    langchain_to_response_output,
    create_response_error,
    # Streaming helpers
    create_response_created_event,
    create_output_item_added_event,
    create_output_text_delta_event,
    create_function_call_arguments_delta_event,
    create_output_item_done_event,
    create_response_completed_event,
    format_stream_event,
)

__all__ = [
    # Anthropic converters
    "anthropic_to_langchain",
    "anthropic_to_langchain_messages",
    "langchain_to_anthropic_response",
    "create_anthropic_error_response",
    # Response API converters
    "response_input_to_langchain_messages",
    "response_request_to_langchain",
    "langchain_to_response_output",
    "create_response_error",
    # Streaming helpers
    "create_response_created_event",
    "create_output_item_added_event",
    "create_output_text_delta_event",
    "create_function_call_arguments_delta_event",
    "create_output_item_done_event",
    "create_response_completed_event",
    "format_stream_event",
]
