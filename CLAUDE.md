# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 用户描述

所有对话、交流必须使用中文。但是生成的代码，包括注释和文档，必须使用英文。

## 项目概述

这是一个基于 LangChain 的自定义 LLM 集成项目,提供了完整的 OAuth2 认证流程和 OpenAI 兼容的 FastAPI 服务。主要目的是让本地或内网的模型能够作为"自托管的 OpenAI"使用。

核心功能包括:
- 自动 OAuth2 Token 刷新和持久化
- OpenAI 兼容的 API 端点 (/v1/models, /v1/chat/completions)
- 支持流式和非流式响应
- 智能网络重试和代理切换机制
- 完整的 LangChain BaseChatModel 实现

## 常用命令

### 环境设置

```bash
# 激活虚拟环境
source .venv/bin/activate

# 同步依赖 (使用 uv)
uv sync

# 创建 .env 配置文件
cp .env.template .env
# 然后编辑 .env 填入真实的凭证
```

### 运行服务

```bash
# 启动 OpenAI 兼容 API 服务器 (端口 8450)
bash run_api.sh
# 或直接使用 uvicorn
uvicorn api:app --host 127.0.0.1 --port 8450

# 启动 Streamlit 聊天 UI
bash run_ui.sh
# 或直接使用 streamlit
streamlit run app.py

# 测试命令行工具
python core/llm.py
```

## 核心架构

### 分层设计

项目采用清晰的分层架构:

1. **认证层** (`core/oauth2_token_manager.py`)
   - `OCAOauth2TokenManager` 类管理完整的 OAuth2 生命周期
   - 独立于 LangChain,专注于 token 管理
   - 智能网络模式:直连/代理自动切换,带重试逻辑
   - 支持 CA 证书合并 (MULTI_CA_BUNDLE) 和 SSL 验证禁用
   - Token 持久化到 .env 文件

2. **LLM 层** (`core/llm.py`)
   - `OCAChatModel` 继承 LangChain 的 `BaseChatModel`
   - 不处理认证,通过依赖注入接收 `OCAOauth2TokenManager` 实例
   - 实现 `_stream`, `_generate`, `_astream` 等 LangChain 标准方法
   - 支持同步 (requests) 和异步 (httpx) 两种 HTTP 后端
   - Tool Calls 支持:兼容旧版和新版 OpenAI 格式

3. **API 层** (`api.py`)
   - FastAPI 服务,提供 OpenAI 兼容端点
   - GET `/v1/models` - 获取可用模型列表
   - POST `/v1/chat/completions` - 聊天补全 (支持 stream=true)
   - SSE 流式响应解析

4. **UI 层** (`app.py`, `ui/`)
   - Streamlit 聊天界面
   - 会话管理和编辑功能
   - 支持实时流式输出

### 关键设计模式

**依赖注入**: `OCAChatModel` 不自己管理认证,而是接收 `token_manager` 实例,实现关注点分离。

**工厂方法**: `OCAChatModel.from_env()` 提供便捷的环境变量加载方式。

**网络重试策略**:
- 快速操作 (OAuth 刷新、模型列表): 使用 CONNECTION_TIMEOUT (默认 2s),支持直连/代理切换
- LLM 推理: 使用 LLM_REQUEST_TIMEOUT (默认 120s),不做重试以保持输出稳定性

### Tool Calls 格式转换

项目在 `_convert_message_to_dict()` 函数中处理 Tool Calls 的两种格式:
- **新版 OpenAI 格式**: `{"type": "function", "id": "...", "function": {"name": "...", "arguments": "..."}}`
- **旧版 LangChain 格式**: `{"name": "...", "args": {...}, "id": "...", "type": "tool_call"}`

自动双向转换确保与不同版本的 LangChain 和 OpenAI SDK 兼容。

## 配置说明

### 关键环境变量 (.env)

**OAuth2 认证**:
- `OAUTH_HOST` - OAuth 服务器地址
- `OAUTH_CLIENT_ID` - 客户端 ID
- `OAUTH_REFRESH_TOKEN` - 刷新 token
- `OAUTH_ACCESS_TOKEN` - 访问 token (自动管理)
- `OAUTH_ACCESS_TOKEN_EXPIRES_AT` - 过期时间 (自动管理)

**LLM 配置**:
- `LLM_API_URL` - 模型 API 端点
- `LLM_MODEL_NAME` - 模型名称
- `LLM_MODELS_API_URL` - 模型列表 API
- `LLM_TEMPERATURE` - 采样温度
- `LLM_REQUEST_TIMEOUT` - LLM 请求超时 (秒,建议 60+)

**网络配置**:
- `FORCE_PROXY` - 强制使用代理 ("true"/"false")
- `HTTP_PROXY_URL` - 代理 URL
- `CONNECTION_TIMEOUT` - 连接超时 (秒,默认 2)
- `MULTI_CA_BUNDLE` - 额外 CA 证书 (逗号分隔)
- `DISABLE_SSL_VERIFY` - 禁用 SSL 验证 ("true"/"false",调试用)

**日志配置**:
- `LOG_LEVEL` - DEBUG 或 INFO
- `LOG_FILE_PATH` - 日志文件路径

### 代理和 CA 配置

典型 BURP 调试配置:
1. `FORCE_PROXY="true"`
2. `HTTP_PROXY_URL="http://127.0.0.1:8080"`
3. `MULTI_CA_BUNDLE="./burp_ca.pem"` (自动合并系统 CA) 或 `REQUESTS_CA_BUNDLE=/path/to/ca_with_burp.pem`

## 重要注意事项

### 网络模式切换

应用会在运行时重新加载 .env,修改 `FORCE_PROXY` 或 `HTTP_PROXY_URL` 无需重启即可生效。

网络重试逻辑:
- 当 `FORCE_PROXY` 不为 "true" 时,优先直连
- 直连失败时自动尝试代理
- 成功后可能"粘"在最后成功的模式

### 日志记录

所有 HTTP 请求和响应都会记录到日志文件 (默认 `logs/llm_api.log`),包含:
- 请求 URL、方法、头部 (Authorization 已脱敏)
- 响应状态码、错误信息
- Tool Calls 和异常堆栈

### 测试和调试

目前项目没有正式的单元测试。主要测试方式:
1. 直接运行 `python core/llm.py` 进行命令行测试
2. 启动 API 服务后使用 curl 测试
3. 通过 Streamlit UI 手动测试

### Python 版本

项目要求 Python >= 3.13,使用 `uv` 进行依赖管理。
