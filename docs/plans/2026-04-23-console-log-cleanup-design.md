# Console Log Cleanup Design

**Date:** 2026-04-23
**Scope:** Chat completions and Responses API runtime logging

## Goal

Keep terminal console output clean while preserving detailed request/response data in the existing `logs/` file output.

## Problem

Current logging sends the same response body details to both file and console handlers. When `LOG_LEVEL=DEBUG`, the terminal shows full model output, tool call arguments, and other verbose payloads. This does not break functionality, but it makes the runtime console hard to scan.

## Requirements

- Keep detailed logs in `logs/llm_api.log`
- Keep console output concise
- Preserve operational stats in console logs
- Do not change API behavior or streaming payloads

## Design

### Handler split

Introduce separate logger handlers with different purposes:

- File handler:
  - Receives full logs at the configured runtime level
  - Continues writing detailed request/response bodies to `logs/llm_api.log`
- Console handler:
  - Receives concise operational logs only
  - Suppresses verbose body dump records

### Message classification

Add a lightweight mechanism so specific log calls can mark themselves as `console=False`.

- Verbose response body logs will be emitted with `extra={"console": False}`
- Summary logs keep the default `console=True`

### Response logging policy

For LLM response logging:

- Console:
  - request size
  - response size
  - content char count
  - tool call count
  - output item count
- File:
  - compacted response body
  - compacted tool call arguments
  - response event debug structures

## Files

- Modify: `core/logger.py`
- Modify: `core/llm.py`
- Modify: `responses_api.py`
- Modify: `tests/test_llm_logging_compaction.py`

## Risks

- Existing log calls must continue to appear in file logs
- Console filtering must not hide error logs
- Logger defaults must remain backward-compatible for non-verbose messages

## Verification

- Unit tests for log summary helpers
- Unit tests for console/file handler filtering
- Targeted pytest run for updated logging tests
