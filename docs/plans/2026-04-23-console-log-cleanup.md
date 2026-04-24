# Console Log Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep terminal output concise while preserving detailed LLM response logs in `logs/llm_api.log`.

**Architecture:** Split logger behavior by handler. Console receives summary-level operational records only, while the file handler continues to receive detailed compacted request/response bodies. Update verbose response log call sites to explicitly opt out of console display.

**Tech Stack:** Python, FastAPI, standard `logging`, pytest

---

### Task 1: Add failing tests for log summaries and handler filtering

**Files:**
- Modify: `tests/test_llm_logging_compaction.py`
- Test: `tests/test_llm_logging_compaction.py`

**Step 1: Write the failing tests**

- Add a test that the response summary helper returns only counts and no raw `content` or `tool_calls`
- Add a test that the logger configuration writes verbose records to file but not to console

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_llm_logging_compaction.py -v`

Expected: FAIL because the new summary helper and console filtering behavior do not exist yet

**Step 3: Write minimal implementation**

- Add a summary-only helper in `core/llm.py`
- Add console filtering support in `core/logger.py`
- Update verbose log call sites to use `extra={"console": False}`

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_llm_logging_compaction.py -v`

Expected: PASS

### Task 2: Update runtime logging call sites

**Files:**
- Modify: `core/llm.py`
- Modify: `responses_api.py`
- Test: `tests/test_llm_logging_compaction.py`

**Step 1: Write the failing test**

- Extend tests to assert verbose body logs are tagged for file-only handling

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_llm_logging_compaction.py -v`

Expected: FAIL due to old logging behavior

**Step 3: Write minimal implementation**

- Route detailed `[LLM RESPONSE] body=...` records to file only
- Keep concise `[LLM RESPONSE] ...` summary on console
- Demote Responses API event body dumps to file-only records

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_llm_logging_compaction.py -v`

Expected: PASS

### Task 3: Verify targeted regressions

**Files:**
- Test: `tests/test_llm_logging_compaction.py`

**Step 1: Run focused verification**

Run: `uv run pytest -q tests/test_llm_logging_compaction.py -v`

Expected: PASS

**Step 2: Review diff**

Run: `git diff -- core/logger.py core/llm.py responses_api.py tests/test_llm_logging_compaction.py docs/plans/2026-04-23-console-log-cleanup-design.md docs/plans/2026-04-23-console-log-cleanup.md`

Expected: Only logging cleanup changes
