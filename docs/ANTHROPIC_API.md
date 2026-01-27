# Anthropic Messages API Compatibility

This document describes the Anthropic Claude Messages API compatibility layer added to the OCA LangChain service.

## Overview

The service now provides dual API compatibility:
- **OpenAI Compatible**: `/v1/chat/completions` (original endpoint)
- **Anthropic Compatible**: `/v1/messages` (new endpoint)

Both endpoints share the same backend infrastructure (OCAChatModel, OAuth2TokenManager), ensuring consistent behavior and authentication.

## Architecture

```
┌─────────────────────┐
│   Client Apps       │
│ (OpenAI SDK /       │
│  Anthropic SDK)     │
└──────────┬──────────┘
           │
           ├──────────────────┐
           │                  │
           ▼                  ▼
    ┌─────────────┐    ┌──────────────┐
    │ /v1/chat/   │    │ /v1/messages │
    │ completions │    │              │
    └──────┬──────┘    └──────┬───────┘
           │                  │
           └────────┬─────────┘
                    │
                    ▼
           ┌────────────────┐
           │  Converters    │
           │ (Format Trans.)│
           └────────┬───────┘
                    │
                    ▼
           ┌────────────────┐
           │  OCAChatModel  │
           │  (LangChain)   │
           └────────┬───────┘
                    │
                    ▼
           ┌────────────────┐
           │ OAuth2 Manager │
           └────────┬───────┘
                    │
                    ▼
           ┌────────────────┐
           │  Backend LLM   │
           │  (LiteLLM)     │
           └────────────────┘
```

## Quick Start

### 1. Start the API Server

```bash
# Using the startup script
bash run_api.sh

# Or manually
uvicorn api:app --host 127.0.0.1 --port 8450
```

The server will start on `http://127.0.0.1:8450` and expose both endpoints.

### 2. Test with Curl

#### Basic Message (Non-Streaming)

```bash
curl -X POST http://127.0.0.1:8450/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "oca/gpt-4.1",
    "max_tokens": 100,
    "messages": [
      {
        "role": "user",
        "content": "Hello! Please respond with a short greeting."
      }
    ]
  }'
```

#### Streaming Message

```bash
curl -N -X POST http://127.0.0.1:8450/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "oca/gpt-4.1",
    "max_tokens": 100,
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Count from 1 to 5"
      }
    ]
  }'
```

#### Tool Calling

```bash
curl -X POST http://127.0.0.1:8450/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "oca/gpt-4.1",
    "max_tokens": 200,
    "messages": [
      {
        "role": "user",
        "content": "What'\''s the weather in Tokyo?"
      }
    ],
    "tools": [
      {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state"
            }
          },
          "required": ["location"]
        }
      }
    ]
  }'
```

### 3. Test with Anthropic Python SDK

First, install the SDK:

```bash
pip install anthropic
```

Then use it with our service:

```python
import anthropic

# Initialize client with our base URL
client = anthropic.Anthropic(
    base_url="http://127.0.0.1:8450",
    api_key="test"  # Optional, for validation only
)

# Basic message
message = client.messages.create(
    model="oca/gpt-4.1",
    max_tokens=100,
    messages=[
        {
            "role": "user",
            "content": "Hello! Please respond with a short greeting."
        }
    ]
)

print(message.content[0].text)
```

See `examples/anthropic_examples.py` for more examples.

## API Reference

### POST /v1/messages

Create a message completion.

#### Headers

- `Content-Type: application/json` (required)
- `x-api-key: <key>` (optional, for validation)
- `anthropic-version: 2023-06-01` (recommended)

#### Request Body

```typescript
{
  model: string,              // Required: Model identifier
  max_tokens: number,         // Required: Max tokens to generate
  messages: [                 // Required: Conversation messages
    {
      role: "user" | "assistant",
      content: string | Array<ContentBlock>
    }
  ],
  temperature?: number,       // Optional: 0-1, default: model-specific
  tools?: [                   // Optional: Tool definitions
    {
      name: string,
      description: string,
      input_schema: object    // JSON Schema
    }
  ],
  stream?: boolean,           // Optional: Enable streaming, default: false
  top_k?: number,             // Optional: Sampling parameter
  top_p?: number,             // Optional: Sampling parameter
  stop_sequences?: string[]   // Optional: Stop sequences
}
```

#### Response (Non-Streaming)

```typescript
{
  id: string,                 // Unique message ID
  type: "message",
  role: "assistant",
  content: [                  // Content blocks
    {
      type: "text",
      text: string
    }
    | {
      type: "tool_use",
      id: string,
      name: string,
      input: object
    }
  ],
  model: string,
  stop_reason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use",
  usage: {
    input_tokens: number,
    output_tokens: number
  }
}
```

#### Response (Streaming)

Server-Sent Events (SSE) format with the following event types:

1. **message_start**: Initial message metadata
2. **content_block_start**: Start of a content block (text or tool_use)
3. **content_block_delta**: Incremental content (text delta or partial JSON)
4. **content_block_stop**: End of current content block
5. **message_delta**: Final message delta (stop_reason, usage)
6. **message_stop**: Stream completion

Example streaming events:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{...}}

event: message_stop
data: {"type":"message_stop"}
```

## Format Conversion

### Message Format

#### Anthropic → LangChain

**Anthropic:**
```json
{
  "role": "user",
  "content": "Hello"
}
```

**LangChain:**
```python
HumanMessage(content="Hello")
```

### Tool Calls Format

#### Anthropic → OpenAI (in LangChain)

**Anthropic:**
```json
{
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_123",
      "name": "get_weather",
      "input": {"location": "Tokyo"}
    }
  ]
}
```

**OpenAI (LangChain additional_kwargs):**
```python
AIMessage(
    content="",
    additional_kwargs={
        "tool_calls": [{
            "type": "function",
            "id": "toolu_123",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Tokyo"}'
            }
        }]
    }
)
```

### Tool Definitions Format

#### Anthropic → OpenAI

**Anthropic:**
```json
{
  "name": "get_weather",
  "description": "Get weather",
  "input_schema": {
    "type": "object",
    "properties": {...}
  }
}
```

**OpenAI:**
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {    // Renamed from input_schema
      "type": "object",
      "properties": {...}
    }
  }
}
```

## Error Handling

All errors follow the Anthropic error format:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error" | "authentication_error" | "not_found_error" | "rate_limit_error" | "api_error",
    "message": "Human-readable error message"
  }
}
```

### Common Errors

| Status | Error Type | Description |
|--------|-----------|-------------|
| 400 | `invalid_request_error` | Missing required fields (e.g., `max_tokens`), invalid message role |
| 401 | `authentication_error` | Invalid API key (if validation enabled) |
| 404 | `not_found_error` | Model not found |
| 429 | `rate_limit_error` | Rate limit exceeded (not implemented yet) |
| 500 | `api_error` | Internal server error |

## Testing

### Automated Tests

Run the comprehensive test suite:

```bash
bash tests/test_anthropic_api.sh
```

This will test:
- Basic messages
- Multi-turn conversations
- Tool calls
- Streaming responses
- Error handling
- Edge cases

### Manual Testing with Python Examples

```bash
python examples/anthropic_examples.py
```

This will run through all usage examples using the Anthropic Python SDK.

## Limitations and Differences from Official Anthropic API

### Current Limitations (PoC)

1. **Token Counting**: Usage statistics (`input_tokens`, `output_tokens`) are estimated or set to 0. The backend needs to provide accurate token counts.

2. **Streaming Tool Calls**: Tool calls in streaming mode are not yet fully implemented. Only text streaming is supported in Sprint 1.

3. **System Messages**: The `system` parameter is not yet supported in the converter.

4. **Images**: Multipart content with images is not yet supported.

5. **Citations**: Beta features like citations are not implemented.

6. **Thinking**: Claude's "thinking" feature is not available.

### Backend-Specific Behavior

Since we're using a LiteLLM backend (endpoint: `/20250519/app/litellm/chat/completions`), some behaviors may differ from the official Anthropic API:

- **Model Capabilities**: The available models and their capabilities depend on the LiteLLM configuration.

- **Tool Calling**: Tool calling behavior depends on the underlying model's support.

- **Sampling Parameters**: Not all sampling parameters may be supported by the backend.

## Configuration

### Environment Variables

No new environment variables are required for Anthropic API compatibility. The service uses existing configuration:

- `LLM_API_URL`: Backend API endpoint (e.g., LiteLLM endpoint)
- `LLM_MODEL_NAME`: Default model name
- `OAUTH_*`: OAuth2 authentication (same for both endpoints)

### Optional: API Key Validation

To enable API key validation for the Anthropic endpoint, you can add:

```python
# In anthropic_api.py, create_message function
if x_api_key and x_api_key != os.getenv("ANTHROPIC_API_KEY"):
    raise HTTPException(
        status_code=401,
        detail=create_anthropic_error_response(
            "authentication_error",
            "Invalid API key"
        ).dict()
    )
```

Then set in `.env`:

```
ANTHROPIC_API_KEY=your-secret-key
```

## Monitoring and Logging

All Anthropic API requests are logged with the prefix `[ANTHROPIC]`:

```
[ANTHROPIC REQUEST] model=oca/gpt-4.1, max_tokens=100, stream=false, messages=1, tools=0
[ANTHROPIC] Starting non-streaming invoke for message msg_abc123
[ANTHROPIC RESPONSE] message_id=msg_abc123, stop_reason=end_turn, content_blocks=1
```

Streaming responses log event types:

```
[ANTHROPIC] Starting streaming response for message msg_xyz789
[ANTHROPIC STREAM] message_start → content_block_start → content_block_delta → ...
```

## Performance Considerations

### Format Conversion Overhead

The Anthropic endpoint adds one conversion layer:
- **Request**: Anthropic format → LangChain format → Backend format
- **Response**: Backend format → LangChain format → Anthropic format

This overhead is minimal (~1-2ms) and should not significantly impact response times.

### Streaming Performance

Streaming uses the same async infrastructure as the OpenAI endpoint, ensuring similar performance characteristics.

## Migration Guide

### From OpenAI SDK to Anthropic SDK

**Before (OpenAI):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8450", api_key="test")

response = client.chat.completions.create(
    model="oca/gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)
```

**After (Anthropic):**
```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8450", api_key="test")

message = client.messages.create(
    model="oca/gpt-4.1",
    max_tokens=100,  # Required in Anthropic API
    messages=[{"role": "user", "content": "Hello"}]
)

print(message.content[0].text)
```

### Key Differences

1. **Required `max_tokens`**: Anthropic API requires `max_tokens` parameter
2. **Response format**: `message.content[0].text` instead of `choices[0].message.content`
3. **Tool format**: Different structure (see Format Conversion section)
4. **Streaming format**: SSE with 6 event types instead of OpenAI's single `data` events

## Troubleshooting

### Issue: "Model not found" error

**Solution**: Check available models:
```bash
curl http://127.0.0.1:8450/v1/models
```

Use the correct model ID from the response.

### Issue: Streaming not working

**Solution**: Ensure you're using `curl -N` (disable buffering) or the Anthropic SDK's streaming support.

### Issue: Tool calls not triggered

**Solution**:
1. Verify tool definitions use valid JSON Schema
2. Check that the user prompt requires tool usage
3. Review backend logs for conversion errors

### Issue: Import errors when starting the server

**Solution**: Ensure all new files are in place:
```bash
ls models/          # Should have anthropic_types.py
ls converters/      # Should have anthropic_request_converter.py
ls anthropic_api.py # Should exist
```

## Future Enhancements

Beyond the PoC (Sprint 1-4), planned enhancements include:

1. **Accurate Token Counting**: Integrate with backend token counting
2. **Complete Streaming Tool Calls**: Implement partial JSON delta for tools
3. **System Messages**: Full support for system parameter
4. **Image Support**: Multimodal content with images
5. **Unified Metrics**: Combined monitoring for both endpoints
6. **Rate Limiting**: Per-endpoint or unified rate limiting

## Support

For issues or questions:
1. Check logs in `logs/llm_api.log`
2. Run the test suite: `bash tests/test_anthropic_api.sh`
3. Review examples in `examples/anthropic_examples.py`
4. Consult this documentation

## References

- [Anthropic Messages API Documentation](https://platform.claude.com/docs/en/api/messages)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [LangChain Documentation](https://python.langchain.com/)
