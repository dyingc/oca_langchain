# LangChain 自定义 LLM 与 OAuth2 认证

本项目实现了一个功能完备的自定义 LangChain `LLM` 类，专门用于与需要 OAuth2 认证（特别是通过刷新令牌 `Refresh Token` 流程）的语言模型 API 进行交互。

它解决了在 LangChain 生态中集成需要动态令牌管理的私有或受保护 LLM 服务的核心问题。

## ✨ 主要特性

- **完整的 OAuth2 刷新令牌流程**: 自动使用 `Refresh Token` 来获取临时的 `Access Token`，并支持令牌轮换（Token Rotation），将新的 `Refresh Token` 持久化回 `.env` 文件。
- **令牌持久化与缓存**: 成功获取的 `Access Token` 会被写入 `.env` 文件，在重启脚本后，只要令牌未过期即可直接复用，避免了不必要的刷新请求。
- **无缝 LangChain 集成**: 严格遵循 LangChain 的 `LLM` 基类规范，可以像使用任何官方 LLM 一样调用 `invoke`, `stream`, `astream` 等方法。
- **支持流式响应 (Streaming)**: 完全实现了对 Server-Sent Events (SSE) 的同步和异步解析，可以实时获取模型输出。
- **配置驱动**: 所有敏感信息和模型参数（如 API 地址、模型名称、温度等）都通过 `.env` 文件进行管理，使代码保持干净和灵活。
- **异步支持**: 同时提供了同步 (`requests`) 和异步 (`httpx`) 的实现，可以轻松集成到现代的异步 Python 应用中。
- **智能网络重试与超时**: 针对不同的 API 请求（如获取模型列表、LLM 推理），采用不同的网络重试策略和超时设置，优化了在代理环境下的性能和稳定性。

## 📂 文件结构

```
.
├── app.py                    # Streamlit 聊天机器人 UI
├── oca_llm.py                # 核心文件：OCAChatModel 类
├── oca_oauth2_token_manager.py   # 认证模块：处理 OAuth2 令牌
├── .env                      # 配置文件 (需手动创建)
├── README.md                 # 本说明文件
├── pyproject.toml            # 项目配置文件 (用于 uv)
└── uv.lock                   # 锁定的依赖版本
```

## 🚀 安装与配置

**1. 环境准备**

本项目使用 `uv` 进行包管理。建议在虚拟环境中操作。

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Linux/macOS)
source .venv/bin/activate

# 如果您没有 uv，请先安装
pip install uv
```

**2. 同步依赖**

使用 `uv.lock` 文件来精确同步所有依赖，确保环境一致性。

```bash
uv sync
```

**3. 创建并配置 `.env` 文件**

根据以下模板创建 `.env` 文件，并填入您的真实凭证和配置：

```dotenv
# --- OAuth2 认证配置 ---
OAUTH_HOST="your-oauth-host.com"
OAUTH_CLIENT_ID="your_client_id"
OAUTH_REFRESH_TOKEN="your_initial_refresh_token"

# --- LLM API 配置 ---
# 语言模型 API 的完整端点 URL
LLM_API_URL="https://your-llm-api-endpoint/v1/chat/completions"

# 要使用的模型名称
LLM_MODEL_NAME="your-model-name"

# 获取可用模型列表的 API 端点
LLM_MODELS_API_URL=""

# 默认的系统提示 (System Prompt)
LLM_SYSTEM_PROMPT="You are a helpful assistant."

# 默认的采样温度 (0.0 到 2.0 之间)
LLM_TEMPERATURE="0.7"

# LLM 请求的超时时间 (单位: 秒，支持浮点数，建议 60 秒及以上。可选)
LLM_REQUEST_TIMEOUT="120"

# --- 网络配置 ---
# 如果应用无法直接访问 OAuth 或 LLM API，请在此处指定 HTTP 代理服务器的地址
# 例如: http://user:password@proxy.example.com:8080
HTTP_PROXY_URL=""

# 网络连接超时时间 (单位: 秒，支持浮点数，建议 2 秒及以上。此超时主要用于快速的 API 请求，如获取模型列表和刷新令牌。)
CONNECTION_TIMEOUT="2"

# --- 以下字段由程序自动管理 ---
OAUTH_ACCESS_TOKEN=
OAUTH_ACCESS_TOKEN_EXPIRES_AT=
```

## 🛠️ 如何运行

### 命令行测试

配置好 `.env` 文件后，直接运行 `oca_llm.py` 即可启动原始的命令行测试程序：

```bash
python oca_llm.py
```

脚本会依次演示三种调用方式：
1.  同步流式调用 (`llm.stream`)
2.  同步非流式调用 (`llm.invoke`)
3.  异步流式调用 (`llm.astream`)

### 启动交互式聊天机器人 UI

我们提供了一个基于 Streamlit 的交互式 Web UI。

**1. 安装 Streamlit**

```bash
pip install streamlit
```

**2. 启动应用**

```bash
streamlit run app.py
```

这会启动一个本地 Web 服务器，并在您的浏览器中打开一个新的标签页，显示聊天机器人界面。您可以在侧边栏调整系统提示、温度和选择不同的模型。

## 🤖 代码概览

### `oca_oauth2_token_manager.py`

- **`OCAOauth2TokenManager` 类**:
  - 这是项目的认证核心。它独立于 LangChain，专门负责管理令牌的整个生命周期。
  - 在初始化时，它会尝试从 `.env` 文件加载一个未过期的 `Access Token`。
  - `get_access_token()` 是其主要公共方法。当被调用时，它会检查内存中的令牌是否有效。如果无效或过期，则会自动触发 `_refresh_tokens()` 方法。
  - `_refresh_tokens()` 方法负责执行与 OAuth2 服务器的通信，用 `Refresh Token` 换取新的 `Access Token`，并处理返回的新 `Refresh Token`，最后将这些信息持久化到 `.env` 文件。
  - **网络连接管理**: 引入了智能网络重试机制，仅在获取模型列表和刷新令牌等快速操作时尝试切换直连/代理模式。LLM 推理请求则使用独立的、更长的超时时间，且不进行网络模式切换重试。

### `oca_llm.py`

- **`OCAChatModel` 类**:
  - 继承自 LangChain 的 `BaseChatModel` 基类。
  - 它不直接处理认证逻辑，而是在初始化时接收一个 `OCAOauth2TokenManager` 实例。
  - 在执行 API 调用（如 `_stream`, `_astream`）前，它会通过调用 `token_manager.get_access_token()` 来获取一个有效的令牌。
  - 它实现了 LangChain 的标准方法，如 `_stream` 用于处理流式响应，`_generate` 用于处理非流式响应，以及对应的异步版本。
  - `@classmethod from_env` 提供了一种便捷的方式来从环境变量实例化该类。
  - **模型列表动态获取**: 在初始化时会尝试从配置的 `LLM_MODELS_API_URL` 获取可用模型列表，并支持在 UI 中手动刷新。
  - **独立 LLM 请求超时**: LLM 推理请求现在使用 `LLM_REQUEST_TIMEOUT` 配置的超时时间，以适应长时间的生成任务。

