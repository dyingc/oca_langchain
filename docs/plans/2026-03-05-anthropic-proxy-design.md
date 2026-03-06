# Anthropic API Proxy Design

Date: 2026-03-05

## Overview

Proxy Claude Code (Anthropic API style) calls to local chat completion API at `http://127.0.0.1:8450` using ccproxy as a standalone component.

## Architecture

```
Claude Code → ccproxy (localhost:4000) → oca_api (localhost:8450) → Backend LLM
                │
                └── Anthropic → OpenAI format conversion
```

## Components

### 1. ccproxy (Port 4000)
- Installed via `uv tool install claude-ccproxy --with 'litellm[proxy]'`
- Listens on `127.0.0.1:4000`
- Converts Anthropic Messages API format to OpenAI Chat Completions format
- Forwards to `http://127.0.0.1:8450/v1/chat/completions`

### 2. oca_api (Port 8450)
- Existing FastAPI service (unchanged)
- Receives OpenAI-format requests
- Proxies to backend LLM

## Model Mapping

Current Claude models (as of 2026-03-05):

| Claude Model | API ID | Target Model |
|--------------|--------|--------------|
| Claude Opus 4.6 | `claude-opus-4-6` | `oca/gpt-5.2` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | `oca/gpt-5.2` |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | `oca/llama4` |
| Claude Haiku 4.5 (alias) | `claude-haiku-4-5` | `oca/llama4` |

### Important Notes

1. **Why not oca/gpt-5.3-codex?**
   - `oca/gpt-5.3-codex` only supports the OpenAI Responses API endpoint (`/v1/responses`)
   - LiteLLM's Responses API support has issues
   - `oca/gpt-5.2` supports the Chat Completions endpoint (`/v1/chat/completions`) which works reliably

2. **API Base Configuration**
   - Must use `http://127.0.0.1:8450/v1` (with `/v1` suffix)
   - LiteLLM appends `/chat/completions` to the api_base

## Configuration

File: `~/.ccproxy/config.yaml`

```yaml
model_list:
  # Default model (sonnet → gpt-5.2)
  - model_name: default
    litellm_params:
      model: claude-sonnet-4-6

  # Claude Opus 4.6 → oca/gpt-5.2
  - model_name: claude-opus-4-6
    litellm_params:
      model: openai/oca/gpt-5.2
      api_base: http://127.0.0.1:8450/v1
      api_key: dummy-key

  # Claude Sonnet 4.6 → oca/gpt-5.2
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: openai/oca/gpt-5.2
      api_base: http://127.0.0.1:8450/v1
      api_key: dummy-key

  # Claude Haiku 4.5 (full ID) → oca/llama4
  - model_name: claude-haiku-4-5-20251001
    litellm_params:
      model: openai/oca/llama4
      api_base: http://127.0.0.1:8450/v1
      api_key: dummy-key

  # Claude Haiku 4.5 (alias) → oca/llama4
  - model_name: claude-haiku-4-5
    litellm_params:
      model: openai/oca/llama4
      api_base: http://127.0.0.1:8450/v1
      api_key: dummy-key

litellm_settings:
  drop_params: true
  callbacks:
    - ccproxy.handler

general_settings:
  forward_client_headers_to_llm_api: true
```

## Usage

### Start oca_api (Terminal 1)
```bash
cd /Users/yingdong/VSCode/oca_langchain
bash run_api.sh
```

### Start ccproxy (Terminal 2)
```bash
ccproxy start --detach
```

### Configure Claude Code
Environment variables are set in `~/.zshrc`:
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_API_KEY=dummy-key
```

### Useful Commands
```bash
# Check ccproxy status
ccproxy status

# View logs
ccproxy logs -f

# Stop ccproxy
ccproxy stop

# Restart ccproxy
ccproxy restart
```

## Future Considerations

1. **Use oca/gpt-5.3-codex**: Once LiteLLM's Responses API support is stable, switch to `oca/gpt-5.3-codex`
2. **Embed ccproxy source**: If satisfied with ccproxy quality, embed source code into project
3. **Add model versions**: As new Claude versions release, add to model_list
4. **Authentication**: Currently using dummy-key; can add real auth if needed
