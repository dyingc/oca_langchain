# 研究发现和关键信息

## 项目背景

现有项目是一个基于 LangChain 的自定义 LLM 集成服务,提供 OpenAI 兼容的 API。现在需要添加 Anthropic Claude Messages API 兼容性。

## 代码结构分析

### 现有核心组件

1. **api.py** (364 行)
   - FastAPI 应用,启动在 127.0.0.1:8450
   - 提供端点:
     - `GET /v1/models` - 模型列表
     - `POST /v1/chat/completions` - 聊天补全 (支持流式和非流式)
     - `GET /v1/model/info` - LiteLLM 兼容的模型信息
     - `POST /v1/spend/calculate` - 占位端点
   - 使用 Pydantic 模型定义请求/响应
   - 流式响应使用 `StreamingResponse` 和 SSE 格式

2. **core/llm.py** (541 行)
   - `OCAChatModel` 继承 `BaseChatModel`
   - 实现 LangChain 标准方法:
     - `_stream()` - 同步流式
     - `_astream()` - 异步流式
     - `_generate()` - 非流式
   - Tool Calls 格式双向转换 (旧版 ↔ 新版 OpenAI 格式)
   - 共享 `OCAOauth2TokenManager` 实例

3. **core/oauth2_token_manager.py**
   - OAuth2 token 管理
   - 自动刷新 token
   - 网络重试逻辑 (直连/代理切换)

### 关键设计模式

- **依赖注入**: `OCAChatModel` 接收 `token_manager` 实例
- **工厂方法**: `OCAChatModel.from_env()` 便捷初始化
- **生命周期管理**: FastAPI `lifespan` 上下文管理器

## OpenAI vs Anthropic API 差异

### 1. 端点和认证

| 方面 | OpenAI | Anthropic |
|------|--------|-----------|
| 端点 | `/v1/chat/completions` | `/v1/messages` |
| 认证 | `Authorization: Bearer <token>` | `x-api-key: <key>` |
| 版本头 | 无 | `anthropic-version: 2023-06-01` |

### 2. 请求格式

**OpenAI**:
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "stream": false
}
```

**Anthropic**:
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "stream": false
}
```

**关键差异**:
- Anthropic **必须**有 `max_tokens`
- Anthropic `model` 名称格式不同
- 消息结构类似,但 Anthropic 的 `content` 支持多部分 (text + image)

### 3. Tool Calls 格式

**OpenAI**:
```json
{
  "tool_calls": [{
    "type": "function",
    "id": "call_abc123",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"Tokyo\"}"
    }
  }]
}
```

**Anthropic**:
```json
{
  "content": [{
    "type": "tool_use",
    "id": "toolu_abc123",
    "name": "get_weather",
    "input": {"city": "Tokyo"}
  }]
}
```

**转换要点**:
- Anthropic 将 tool_calls 作为 content 的一部分
- OpenAI 使用 `arguments` (字符串),Anthropic 使用 `input` (对象)
- ID 格式不同: `call_` vs `toolu_`

### 4. 流式响应格式

**OpenAI SSE**:
```
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: [DONE]
```

**Anthropic SSE**:
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_123",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
```

**关键差异**:
- Anthropic 使用 `event:` + `data:` 两行
- Anthropic 定义了 6 种事件类型
- Anthropic 有明确的生命周期边界
- Anthropic 在最后发送 usage 统计

### 5. 工具定义格式

**OpenAI**:
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"]
    }
  }
}
```

**Anthropic**:
```json
{
  "name": "get_weather",
  "description": "Get weather",
  "input_schema": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  }
}
```

**转换要点**:
- Anthropic 没有 `type: "function"` 包装
- Anthropic 使用 `input_schema` 而不是 `parameters`

## 技术挑战

### 1. 流式响应转换

**挑战**: OpenAI 和 Anthropic 的流式格式完全不同
- OpenAI: 单一 `data:` 行,简单的 delta 累积
- Anthropic: 6 种事件类型,需要生成边界事件

**解决方案**:
- 创建 `AnthropicStreamEvent` 类生成标准事件
- 在 `anthropic_stream_generator()` 中转换 LangChain 流到 Anthropic 事件
- 累积 content 和 tool_calls,在适当时机发送事件

### 2. Tool Calls 流式传输

**挑战**:
- OpenAI: 在 `delta.tool_calls[i].function.arguments` 中累积 JSON 字符串
- Anthropic: 在 `content_block_delta.delta.partial_json` 中累积

**解决方案**:
- 追踪每个 tool_call 的索引
- 累积 `arguments` 字符串
- 在 `content_block_stop` 时输出完整的 JSON

### 3. 多部分内容处理

**挑战**: Anthropic 的 `content` 可以是字符串或数组
- 纯文本: `"content": "Hello"`
- 多部分: `"content": [{"type": "text", "text": "Hello"}, {"type": "image", "source": {...}}]`

**解决方案**:
- 请求转换: 支持两种格式,统一转换为 LangChain 格式
- 响应转换: 将 LangChain content 转换为 Anthropic 单元素数组

### 4. 错误格式差异

**OpenAI**:
```json
{
  "error": {
    "message": "Invalid request",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

**Anthropic**:
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid request"
  }
}
```

**解决方案**:
- 创建统一的错误处理中间件
- 根据 API 类型返回不同的错误格式

## 可复用组件

### 可以直接复用的代码

1. **`OCAChatModel`** - 核心模型逻辑
   - 所有 LangChain 集成方法
   - OAuth2 认证处理
   - 网络重试逻辑

2. **`OCAOauth2TokenManager`** - Token 管理
   - 自动刷新
   - 持久化
   - 网络模式切换

3. **FastAPI 生命周期管理** - `lifespan` 上下文管理器
   - 组件初始化
   - 清理逻辑

### 需要新建的组件

1. **Anthropic Pydantic 模型** - 请求/响应定义
2. **格式转换器** - OpenAI ↔ Anthropic 格式转换
3. **流事件生成器** - Anthropic SSE 事件生成
4. **错误处理** - Anthropic 错误格式

## 性能考虑

### 开销分析

1. **格式转换开销**
   - 请求: JSON 解析 → Pydantic 验证 → 转换 → 构建 LangChain 消息
   - 响应: LangChain 消息 → 转换 → JSON 序列化
   - **预估开销**: < 5ms (JSON 转换)

2. **流式响应开销**
   - 每个事件需要额外格式化
   - **预估开销**: < 1ms/事件

3. **总体影响**
   - 非流式: < 1% 额外延迟
   - 流式: < 2% 额外延迟

### 优化策略

1. **缓存常用转换** - 如工具定义
2. **使用 `orjson`** - 更快的 JSON 库
3. **减少中间对象** - 直接转换,避免多层包装

## 测试策略

### 单元测试

- 测试格式转换器 (请求/响应)
- 测试流事件生成器
- 测试错误处理

### 集成测试

- 使用 curl 测试端点
- 使用 Anthropic Python SDK 测试
- 多轮对话测试
- 工具调用测试

### 性能测试

- 基准测试 (延迟对比)
- 并发测试 (吞吐量)
- 内存使用测试

## 参考文档

- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages)
- [Streaming with Claude](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [Tool use (Claude)](https://platform.claude.com/docs/en/agents-and-tools/tool-use)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/docs/)

---

## 422 错误调查 (2026-01-26)

### 问题描述

从日志中发现 `/v1/messages` 端点出现了 **422 Unprocessable Content** 错误:

```
INFO:     127.0.0.1:60872 - "POST /v1/messages HTTP/1.1" 200 OK
INFO:     127.0.0.1:60873 - "POST /v1/messages HTTP/1.1" 422 Unprocessable Content
```

### 调查结论 ✅ (2026-01-26 已解决)

**根本原因: 422 错误是由 Pydantic 请求验证失败引起的**

经过详细测试验证:

1. **流式响应格式正确**: 使用 curl 和 Anthropic SDK 测试,SSE 事件格式完全正确
2. **JSON 转义正确**: 包含 JSON 字符串的 content 被正确转义
3. **422 是请求验证错误**: 当请求缺少必填字段时会返回 422

#### 会触发 422 的请求格式:
- 缺少 `max_tokens` 字段
- 消息中缺少 `content` 字段

#### 不会触发 422 的请求格式:
- 带 JSON 字符串的 content ✓
- 带换行符的 content ✓
- 带 emoji 的 content ✓

### 验证测试

```bash
# Test 1: 正常流式请求 - 成功
curl -X POST http://127.0.0.1:8450/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model": "oca/gpt-4.1", "max_tokens": 50, "messages": [{"role": "user", "content": "Say hello"}], "stream": true}'
# Result: 200 OK, SSE events properly formatted

# Test 2: JSON content 流式请求 - 成功
curl -X POST http://127.0.0.1:8450/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "oca/gpt-4.1", "max_tokens": 100, "messages": [{"role": "user", "content": "Respond with JSON"}], "stream": true}'
# Result: 200 OK, JSON properly escaped in SSE delta

# Test 3: Anthropic SDK 测试 - 成功
from anthropic import Anthropic
client = Anthropic(api_key="test", base_url="http://127.0.0.1:8450")
with client.messages.stream(model="oca/gpt-4.1", max_tokens=50, messages=[...]) as stream:
    for text in stream.text_stream:
        print(text)
# Result: SUCCESS, no 422 error
```

### 发现的新问题 ⚠️

在调查过程中发现了另一个问题: **Tool Result 消息转换错误**

当发送包含 `tool_result` 的消息时:
- Anthropic 格式: `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}`
- 被错误转换为: `{"role": "user", "content": "..."}`
- 应该转换为 OpenAI 格式: `{"role": "tool", "tool_call_id": "...", "content": "..."}`

这导致后端返回 400 Bad Request。

**相关文件**: `converters/anthropic_request_converter.py`

### 结论

| 问题 | 状态 | 原因 |
|------|------|------|
| 422 错误 | ✅ 已解决 | 客户端请求缺少必填字段 |
| Tool Result 转换 | ⚠️ 待修复 | 转换器未正确处理 tool_result |

### 相关文件

- `anthropic_api.py:169-240` - 流式生成器实现
- `anthropic_api.py:133-145` - SSE 事件生成函数
- `models/anthropic_types.py:127-134` - `AnthropicStreamContentBlockDelta` 模型
- `converters/anthropic_request_converter.py` - 消息格式转换器

---

## 流式 Tool Calls 转换问题 (2026-01-27)

### 问题描述

Claude Code 连接到 `/v1/messages` 端点时，后端 LiteLLM 返回的 OpenAI 格式流式 `tool_calls` 没有被转换为 Anthropic 格式的 `tool_use` 事件，导致 Claude Code 无法执行工具调用。

### 架构

```
Claude Code (Anthropic SDK)
    ↓ Anthropic Messages API format
http://127.0.0.1:8450/v1/messages (OCA API)
    ↓ OpenAI format
Backend LiteLLM (OpenAI style)
    ↓ OpenAI streaming tool_calls
OCA API (转换层)
    ↓ Anthropic streaming tool_use
Claude Code
```

### 根本原因

`anthropic_api.py` 的 `anthropic_stream_generator()` 函数只处理文本 content，完全忽略了 `tool_calls`：

```python
async for chunk in chat_model.astream(...):
    content_delta = getattr(chunk, "content", None)  # ✓ 处理文本
    if content_delta:
        yield generate_content_block_delta(...)
    # ✗ 完全忽略了 chunk.message.additional_kwargs["tool_calls"]!
```

### 格式差异

**OpenAI 流式 tool_calls 格式**（后端返回）：
```
data: {"choices":[{"delta":{"tool_calls":[{"id":"call_xxx","type":"function","function":{"name":"Bash","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\""}}]}}]}
...
data: {"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}
```

**Anthropic 流式 tool_use 格式**（Claude Code 期望）：
```
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_xxx","name":"Bash","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\""}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},...}
```

### 解决方案 ✅

修改 `anthropic_stream_generator()` 函数：

1. **跟踪 tool_calls 状态**：使用 `tool_states` 字典跟踪每个工具调用的 id、name、arguments buffer 和 block_index
2. **生成 content_block_start**：当第一个 tool_call chunk 到达时，生成 `type: "tool_use"` 的 content_block_start 事件
3. **流式 arguments**：将 OpenAI 的 `arguments` 片段作为 `input_json_delta` 事件发送
4. **正确关闭块**：为每个 tool block 生成 `content_block_stop` 事件
5. **设置 stop_reason**：当有 tool_calls 时，设置 `stop_reason: "tool_use"`
6. **ID 转换**：将 OpenAI 格式的 `call_xxx` 转换为 Anthropic 格式的 `toolu_xxx`

### 修改的文件

- `anthropic_api.py:169-295` - 重写 `anthropic_stream_generator()` 函数

### 状态

✅ 代码已修改并测试通过

---

## 消息序列验证 Bug 修复 (2026-01-28)

### 当前状态
- 已实现消息序列验证功能
- 10个测试中7个通过，3个失败
- 需要修复剩余的bug

### Bug 1 (CRITICAL): Anthropic 验证修改有效序列
**位置**: `converters/anthropic_request_converter.py`
**函数**: `validate_and_fix_anthropic_message_sequence()`
**现象**: 有效的 assistant → tool_use → user → tool_result 序列被错误修改
**影响**: 有效序列应该保持不变，但现在被破坏了

### Bug 2 (MEDIUM): OpenAI 孤儿 tool 消息未跳过
**位置**: `api.py`
**函数**: `validate_and_fix_message_sequence()`
**现象**: 当 tool_calls 被移除后，对应的孤儿 tool 消息应该被跳过，但没有
**影响**: 导致消息数量不符合预期

### 下一步
1. 调试并修复 Bug 1
2. 调试并修复 Bug 2
3. 重新运行所有测试
4. 验证 API 集成测试
