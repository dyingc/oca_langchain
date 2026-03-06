# Anthropic Proxy Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up ccproxy to translate Anthropic API calls to OpenAI format and forward to local oca_api at localhost:8450

**Architecture:** ccproxy (LiteLLM-based) listens on port 4000, receives Anthropic-format requests from Claude Code, converts to OpenAI format, forwards to oca_api on port 8450 which proxies to backend LLM.

**Tech Stack:** ccproxy (uv tool), LiteLLM, existing oca_api FastAPI service

---

## Prerequisites

- [ ] oca_api service running on port 8450
- [ ] uv package manager installed
- [ ] Terminal access

---

### Task 1: Install ccproxy

**Files:**
- N/A (system-wide tool installation)

**Step 1: Check if ccproxy is already installed**

Run: `ccproxy --version`
Expected: Either version info or "command not found"

**Step 2: Install ccproxy via uv**

Run: `uv tool install ccproxy`
Expected: Successfully installed ccproxy

**Step 3: Verify installation**

Run: `ccproxy --version`
Expected: Version info displayed

---

### Task 2: Verify Configuration File

**Files:**
- Verify: `~/.ccproxy/config.yaml` (already created)

**Step 1: Verify config directory exists**

Run: `ls -la ~/.ccproxy/`
Expected: config.yaml file listed

**Step 2: Verify config contents**

Run: `cat ~/.ccproxy/config.yaml`
Expected: YAML with model_list containing claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001, claude-haiku-4-5 mappings

---

### Task 3: Start Services and Test Integration

**Files:**
- N/A (runtime testing)

**Step 1: Start oca_api (Terminal 1)**

Run:
```bash
cd /Users/yingdong/VSCode/oca_langchain
source .venv/bin/activate
bash run_api.sh
```
Expected: Server running on http://127.0.0.1:8450

**Step 2: Start ccproxy (Terminal 2)**

Run: `ccproxy`
Expected: LiteLLM proxy running on http://127.0.0.1:4000

**Step 3: Test Anthropic to OpenAI translation**

Run:
```bash
curl http://127.0.0.1:4000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```
Expected: Response from backend LLM via oca_api

---

### Task 4: Configure Claude Code Environment

**Files:**
- Modify: shell environment or `.zshrc`/`.bashrc`

**Step 1: Set environment variables for Claude Code**

Add to shell config or export in terminal:
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_API_KEY=dummy-key
```

**Step 2: Verify environment**

Run: `echo $ANTHROPIC_BASE_URL`
Expected: `http://127.0.0.1:4000`

---

### Task 5: Document Usage

**Files:**
- Create: `docs/ccproxy-usage.md`

**Step 1: Create usage documentation**

```markdown
# Using ccproxy with Claude Code

## Quick Start

1. Start oca_api: `bash run_api.sh`
2. Start ccproxy: `ccproxy`
3. Set environment: `export ANTHROPIC_BASE_URL=http://127.0.0.1:4000`
4. Run Claude Code

## Model Mapping

- claude-opus-4-6 → oca/gpt-5.3-codex
- claude-sonnet-4-6 → oca/gpt-5.3-codex
- claude-haiku-4-5 → oca/llama4

## Ports

- 4000: ccproxy (Anthropic API)
- 8450: oca_api (OpenAI API)
```

**Step 2: Commit documentation**

Run:
```bash
git add docs/ccproxy-usage.md
git commit -m "docs: Add ccproxy usage guide"
```

---

## Verification Checklist

- [ ] ccproxy installed and runs without errors
- [ ] Config file correctly maps all 4 model IDs
- [ ] oca_api receives OpenAI-format requests from ccproxy
- [ ] Claude Code can connect to ccproxy
- [ ] All model mappings work (opus, sonnet, haiku)
