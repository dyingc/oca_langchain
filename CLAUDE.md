# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 用户描述

所有对话、交流必须使用中文。但是生成的代码，包括注释和文档，必须使用英文。

## 开发偏好

- 功能开发优先使用 **git worktree** 进行隔离工作区开发

## 项目概述

这是一个基于 LangChain 的自定义 LLM 集成项目，提供完整的 OAuth2 认证流程和多协议兼容的 FastAPI 服务。主要目的是让内网模型作为"自托管的 OpenAI/Anthropic"使用。

核心功能:
- 自动 OAuth2 Token 刷新和持久化
- OpenAI 兼容端点 (`/v1/chat/completions`, `/v1/responses`)
- Anthropic 兼容端点 (`/v1/messages`)
- 支持流式和非流式响应
- 智能网络重试和代理切换机制
- 完整的 LangChain BaseChatModel 实现

## 常用命令

### 环境设置

```bash
source .venv/bin/activate
uv sync
cp .env.template .env   # then fill in credentials
```

### 运行服务

```bash
# Start API server (port 8450)
bash run_api.sh
# or: uvicorn api:app --host 127.0.0.1 --port 8450

# Start Streamlit UI
bash run_ui.sh

# CLI test
python core/llm.py
```

### 测试

```bash
# Run pytest suite (tests/ directory)
pytest tests/

# Run a single test file
pytest tests/test_responses_api.py

# Run E2E tests (requires running API server)
pytest tests/test_e2e_api_validation.py -m e2e
```

## 核心架构

### 分层设计

1. **认证层** (`core/oauth2_token_manager.py`)
   - `OCAOauth2TokenManager` 管理完整 OAuth2 生命周期
   - 智能网络模式: 直连/代理自动切换，带重试逻辑
   - 支持 CA 证书合并 (`MULTI_CA_BUNDLE`) 和 SSL 验证禁用
   - Token 持久化到 `.env` 文件

2. **LLM 层** (`core/llm.py`)
   - `OCAChatModel` 继承 LangChain `BaseChatModel`
   - 通过依赖注入接收 `OCAOauth2TokenManager`
   - 实现 `_stream`, `_generate`, `_astream` 等 LangChain 标准方法
   - 同步 (requests) 和异步 (httpx) 两种 HTTP 后端

3. **API 层** (`api.py`) — FastAPI 主入口，挂载以下路由模块:
   - `GET /v1/models` — 可用模型列表
   - `GET /v1/model/info` — 模型详情
   - `POST /v1/spend/calculate` — 费用估算
   - `POST /v1/chat/completions` — OpenAI Chat 兼容 (支持 streaming)
   - `POST /v1/messages` — Anthropic Messages API (via `anthropic_api.py`) **[不稳定，不支持复杂 tool calls]**
   - `POST /v1/responses` — OpenAI Responses API (via `responses_api.py`)
   - `GET/DELETE /v1/responses/{id}` — 响应管理

4. **转换层** (`converters/`)
   - `responses_converter.py` — ResponseRequest ↔ LangChain 消息格式
   - `anthropic_request_converter.py` — Anthropic ↔ LangChain 消息格式

5. **类型层** (`models/`)
   - `responses_types.py` — OpenAI Responses API Pydantic 模型
   - `anthropic_types.py` — Anthropic API Pydantic 模型

6. **UI 层** (`app.py`, `ui/`) — Streamlit 聊天界面，支持会话管理和流式输出

### Responses API Passthrough

`responses_passthrough.py` 提供直通模式: 当后端原生支持 Responses API 时，请求直接转发无需 LangChain 转换。适用于 Codex CLI 等工具。

### Tool Calls 格式转换

`_convert_message_to_dict()` 处理两种格式的自动互转:
- **新版 OpenAI 格式**: `{"type": "function", "id": "...", "function": {"name": "...", "arguments": "..."}}`
- **旧版 LangChain 格式**: `{"name": "...", "args": {...}, "id": "...", "type": "tool_call"}`

## 配置说明

### 关键环境变量 (.env)

**OAuth2 认证**:
- `OAUTH_HOST`, `OAUTH_CLIENT_ID`, `OAUTH_REFRESH_TOKEN`
- `OAUTH_ACCESS_TOKEN`, `OAUTH_ACCESS_TOKEN_EXPIRES_AT` (自动管理)

**LLM 配置**:
- `LLM_API_URL` — 模型 API 端点
- `LLM_MODEL_NAME` — 模型名称
- `LLM_RESPONSES_MODEL_NAME` — Responses API 专用模型名 (如 `oca/gpt-5.4-pro`)
- `LLM_MODELS_API_URL` — 模型列表 API
- `LLM_TEMPERATURE`, `LLM_REQUEST_TIMEOUT`

**网络配置**:
- `FORCE_PROXY` — 强制使用代理 ("true"/"false")
- `HTTP_PROXY_URL` — 代理 URL
- `CONNECTION_TIMEOUT` — 连接超时 (默认 2s)
- `MULTI_CA_BUNDLE` — 额外 CA 证书 (逗号分隔)
- `DISABLE_SSL_VERIFY` — 禁用 SSL 验证 (调试用)

**日志配置**:
- `LOG_LEVEL` — DEBUG 或 INFO
- `LOG_FILE_PATH` — 默认 `logs/llm_api.log`

### 网络重试策略

- 快速操作 (OAuth 刷新、模型列表): 使用 `CONNECTION_TIMEOUT`，支持直连/代理切换
- LLM 推理: 使用 `LLM_REQUEST_TIMEOUT`，不做重试以保持输出稳定性
- `.env` 中的 `FORCE_PROXY` 修改无需重启即可生效

### 代理和 CA 配置 (BURP 调试)

```
FORCE_PROXY="true"
HTTP_PROXY_URL="http://127.0.0.1:8080"
MULTI_CA_BUNDLE="./burp_ca.pem"
```

## 本地 Responses API 调用约定

- `oca/gpt-5.4-pro` 应使用 `/v1/responses`，不用 `/v1/chat/completions`
- 请求体用 `input` 而非 `messages`；`input` 为 message object 数组
- `content` 为 content item 数组，文本用 `{"type":"input_text","text":"..."}`
- `max_output_tokens` 至少为 `16`
- `oca/gpt-5.4-pro` 需显式带 `reasoning: {"effort": "high"}`
- 保留推理密文: `include: ["reasoning.encrypted_content"]`

推荐最小 payload:

```json
{
  "model": "oca/gpt-5.4-pro",
  "include": ["reasoning.encrypted_content"],
  "input": [{"role": "user", "content": [{"type": "input_text", "text": "reply with exactly: ok"}]}],
  "reasoning": {"effort": "high"},
  "max_output_tokens": 16
}
```

## Python 版本

Python >= 3.13，使用 `uv` 管理依赖。
