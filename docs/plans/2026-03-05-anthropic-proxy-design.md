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
- Installed via `uv tool install ccproxy`
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
| Claude Opus 4.6 | `claude-opus-4-6` | `oca/gpt-5.3-codex` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | `oca/gpt-5.3-codex` |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | `oca/llama4` |
| Claude Haiku 4.5 (alias) | `claude-haiku-4-5` | `oca/llama4` |

## Configuration

File: `~/.ccproxy/config.yaml`

```yaml
model_list:
  # Opus 4.6 → gpt-5.3-codex
  - model_name: claude-opus-4-6
    litellm_params:
      model: openai/oca/gpt-5.3-codex
      api_base: http://127.0.0.1:8450
      api_key: dummy-key

  # Sonnet 4.6 → gpt-5.3-codex
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: openai/oca/gpt-5.3-codex
      api_base: http://127.0.0.1:8450
      api_key: dummy-key

  # Haiku 4.5 (full ID) → llama4
  - model_name: claude-haiku-4-5-20251001
    litellm_params:
      model: openai/oca/llama4
      api_base: http://127.0.0.1:8450
      api_key: dummy-key

  # Haiku 4.5 (alias) → llama4
  - model_name: claude-haiku-4-5
    litellm_params:
      model: openai/oca/llama4
      api_base: http://127.0.0.1:8450
      api_key: dummy-key

litellm_settings:
  host: 127.0.0.1
  port: 4000
  drop_params: true
  callbacks:
    - ccproxy.handler
```

## Usage

### Start oca_api (Terminal 1)
```bash
cd /Users/yingdong/VSCode/oca_langchain
bash run_api.sh
```

### Start ccproxy (Terminal 2)
```bash
ccproxy
```

### Configure Claude Code
Set environment variable:
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_API_KEY=dummy-key
```

## Future Considerations

1. **Embed ccproxy source**: If satisfied with ccproxy quality, embed source code into project
2. **Add model versions**: As new Claude versions release, add to model_list
3. **Authentication**: Currently using dummy-key; can add real auth if needed
