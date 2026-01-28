# Task Plan: Fix Streaming tool_calls → tool_use Conversion

## Problem Summary

Claude Code 连接到 `/v1/messages` 端点，该端点调用后端 LiteLLM（OpenAI 格式）。后端返回的 OpenAI 格式 `tool_calls` 没有被转换为 Anthropic 格式的 `tool_use` 事件，导致 Claude Code 无法执行工具调用。

## Architecture

```
Claude Code (Anthropic SDK)
    ↓ Anthropic Messages API format
http://127.0.0.1:8450/v1/messages (OCA API)
    ↓ OpenAI format (converted)
Backend LiteLLM (OpenAI style)
    ↓ OpenAI streaming tool_calls
OCA API (needs to convert!)
    ↓ Anthropic streaming tool_use (NOT IMPLEMENTED!)
Claude Code
```

## Root Cause

`anthropic_api.py` 的 `anthropic_stream_generator()` 函数（第 169-241 行）只处理文本 content：

```python
async for chunk in chat_model.astream(...):
    content_delta = getattr(chunk, "content", None)  # ✓ 处理文本
    if content_delta:
        yield generate_content_block_delta(...)
    # ✗ 完全忽略了 tool_calls!
```

## Required Changes

### 1. Understand OpenAI Streaming tool_calls Format

从后端返回的格式（用户提供）：
```
data: {"choices":[{"delta":{"tool_calls":[{"id":"call_xxx","type":"function","function":{"name":"Bash","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"command"}}]}}]}
...
data: {"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}
```

### 2. Required Anthropic Streaming tool_use Format

Claude Code 期望的格式：
```
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_xxx","name":"Bash","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"command"}}
...

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},...}
```

### 3. Implementation Tasks

- [x] 3.1 Read current `anthropic_stream_generator()` implementation
- [x] 3.2 Modify to track tool_calls from `chunk.message.additional_kwargs["tool_calls"]`
- [x] 3.3 Generate `content_block_start` with `type: "tool_use"` when first tool_call arrives
- [x] 3.4 Generate `input_json_delta` events for argument fragments
- [x] 3.5 Generate `content_block_stop` when tool block completes
- [x] 3.6 Set `stop_reason: "tool_use"` in message_delta when finish_reason is "tool_calls"
- [x] 3.7 Handle multiple tool_calls (each gets its own content block index)

### 4. Testing

- [x] 4.1 Test with curl to verify SSE format ✅
- [ ] 4.2 Test with Anthropic SDK
- [ ] 4.3 Test with Claude Code

## Current Status
✅ Implementation complete and curl test passed! Ready for Claude Code testing.
