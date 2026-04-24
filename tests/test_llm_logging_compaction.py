import logging
from io import StringIO
from unittest.mock import Mock

import core.llm as llm
from core.llm import (
    _LOG_MAX_LIST_ITEMS,
    _LOG_MAX_STRING_CHARS,
    _build_response_log_obj,
    _build_response_log_summary,
    _log_llm_request_detail,
    _compact_for_log,
)
from core.logger import get_logger


def test_compact_for_log_truncates_long_string():
    raw = "a" * (_LOG_MAX_STRING_CHARS + 123)
    compact = _compact_for_log(raw)

    assert isinstance(compact, str)
    assert "truncated 123 chars" in compact
    assert compact.startswith("a" * _LOG_MAX_STRING_CHARS)


def test_compact_for_log_limits_list_items():
    raw = list(range(_LOG_MAX_LIST_ITEMS + 3))
    compact = _compact_for_log(raw)

    assert isinstance(compact, list)
    assert len(compact) == _LOG_MAX_LIST_ITEMS + 1
    assert compact[-1] == "<3 more items>"


def test_build_response_log_obj_adds_counts_and_compacts_body():
    long_content = "x" * (_LOG_MAX_STRING_CHARS + 10)
    tool_calls = [{
        "type": "function",
        "id": "call_1",
        "function": {
            "name": "search_in_files",
            "arguments": "y" * (_LOG_MAX_STRING_CHARS + 20),
        },
    }]

    log_obj = _build_response_log_obj(long_content, tool_calls)

    assert log_obj["content_chars"] == len(long_content)
    assert log_obj["tool_calls_count"] == 1
    assert "truncated 10 chars" in log_obj["content"]
    assert "truncated 20 chars" in log_obj["tool_calls"][0]["function"]["arguments"]


def test_build_response_log_summary_keeps_only_counts():
    content = "visible output that should not be in summary"
    tool_calls = [{
        "type": "function",
        "id": "call_1",
        "function": {
            "name": "search_in_files",
            "arguments": '{"query":"secret"}',
        },
    }]

    summary = _build_response_log_summary(content, tool_calls)

    assert summary == {
        "content_chars": len(content),
        "tool_calls_count": 1,
    }


def test_logger_routes_console_false_records_to_file_only(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FILE_PATH", str(tmp_path / "llm_api.log"))

    logger = get_logger("tests.console_filter")

    console_io = StringIO()
    stream_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            stream_handler = handler
            break

    assert stream_handler is not None
    stream_handler.setStream(console_io)

    logger.info("summary message")
    logger.info("verbose body", extra={"console": False})

    for handler in logger.handlers:
        handler.flush()

    console_output = console_io.getvalue()
    file_output = (tmp_path / "llm_api.log").read_text(encoding="utf-8")

    assert "summary message" in console_output
    assert "verbose body" not in console_output
    assert "summary message" in file_output
    assert "verbose body" in file_output


def test_log_llm_request_detail_marks_record_as_file_only(monkeypatch):
    mock_info = Mock()
    monkeypatch.setattr(llm.logger, "info", mock_info)

    _log_llm_request_detail(
        {"Authorization": "Bearer secret-token", "X-Test": "1"},
        {
            "model": "oca/gpt-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert mock_info.call_count == 1
    args, kwargs = mock_info.call_args

    assert args[0] == "[LLM REQUEST DETAIL] headers=%s payload=%s"
    assert "<redacted>" in args[1]
    assert '"model": "oca/gpt-5.2"' in args[2]
    assert kwargs["extra"] == {"console": False}
