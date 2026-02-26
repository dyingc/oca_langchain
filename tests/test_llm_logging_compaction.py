from core.llm import _LOG_MAX_LIST_ITEMS, _LOG_MAX_STRING_CHARS, _build_response_log_obj, _compact_for_log


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
