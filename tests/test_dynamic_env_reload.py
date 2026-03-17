from model_resolver import resolve_model_for_endpoint, FALLBACK_MODEL
from responses_passthrough import resolve_reasoning_effort, resolve_null_reasoning, enforce_pro_model_min_reasoning


def test_responses_api_model_reload_from_env_each_call(monkeypatch, tmp_path):
    """Env override via LLM_RESPONSES_MODEL_NAME is re-read on each call."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.1": ["RESPONSES"], "oca/gpt-5.2": ["RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.1"

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.2"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.2"


def test_responses_api_env_override_takes_priority_over_incoming_oca_model(monkeypatch, tmp_path):
    """When LLM_RESPONSES_MODEL_NAME is set, it takes priority even if incoming has oca/ prefix."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.9": ["RESPONSES"], "oca/gpt-5.1-codex": ["RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.9"\n', encoding="utf-8")
    # New behavior: env override takes priority, returns oca/gpt-5.9 (not the incoming oca/gpt-5.1-codex)
    assert resolve_model_for_endpoint("oca/gpt-5.1-codex", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.9"


def test_responses_api_model_falls_back_to_fallback_when_env_removed(monkeypatch, tmp_path):
    """When LLM_RESPONSES_MODEL_NAME is removed, resolution falls through to FALLBACK_MODEL if not in catalog."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    # gpt-4o not in catalog — will fall back to FALLBACK_MODEL
    catalog = {"oca/gpt-5.1": ["RESPONSES"], "oca/gpt-5.4": ["CHAT_COMPLETIONS", "RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.1"

    # Remove env override — incoming gpt-4o not in catalog → FALLBACK_MODEL
    env_file.write_text("# LLM_RESPONSES_MODEL_NAME removed\n", encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == FALLBACK_MODEL


def test_passthrough_model_reload_from_env_each_call(monkeypatch, tmp_path):
    """LLM_RESPONSES_MODEL_NAME is re-read from runtime_env on each call."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.1": ["RESPONSES"], "oca/gpt-5.2": ["RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.1"

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.2"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.2"


def test_passthrough_env_override_takes_priority_over_incoming_oca_model(monkeypatch, tmp_path):
    """When LLM_RESPONSES_MODEL_NAME is set, it takes priority even over oca/ incoming model."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.9": ["RESPONSES"], "oca/gpt-5.1-codex": ["RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.9"\n', encoding="utf-8")
    # New behavior: env override wins
    assert resolve_model_for_endpoint("oca/gpt-5.1-codex", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.9"


def test_passthrough_model_falls_back_when_env_removed(monkeypatch, tmp_path):
    """When LLM_RESPONSES_MODEL_NAME is removed and incoming not in catalog, returns FALLBACK_MODEL."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.1": ["RESPONSES"], "oca/gpt-5.4": ["CHAT_COMPLETIONS", "RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.1"

    env_file.write_text("# LLM_RESPONSES_MODEL_NAME removed\n", encoding="utf-8")
    # gpt-4o not in catalog → fallback
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == FALLBACK_MODEL


def test_passthrough_effort_reload_from_env_each_call(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_REASONING_STRENGTH", raising=False)

    env_file.write_text('LLM_REASONING_STRENGTH="high"\n', encoding="utf-8")
    assert resolve_reasoning_effort("low") == "high"

    env_file.write_text('LLM_REASONING_STRENGTH="minimal"\n', encoding="utf-8")
    assert resolve_reasoning_effort("low") == "minimal"


def test_passthrough_effort_falls_back_to_incoming_when_env_removed(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_REASONING_STRENGTH", raising=False)

    env_file.write_text('LLM_REASONING_STRENGTH="high"\n', encoding="utf-8")
    assert resolve_reasoning_effort("low") == "high"

    env_file.write_text("# LLM_REASONING_STRENGTH removed\n", encoding="utf-8")
    assert resolve_reasoning_effort("low") == "low"


def test_passthrough_missing_reasoning_treated_as_null(monkeypatch, tmp_path):
    """When reasoning key is completely absent, it should be filled like null."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_NON_REASONING_STRENGTH", raising=False)

    env_file.write_text('LLM_NON_REASONING_STRENGTH="medium"\n', encoding="utf-8")
    result = resolve_null_reasoning()
    assert result == {"effort": "medium", "summary": "auto"}


def test_passthrough_missing_reasoning_no_env(monkeypatch, tmp_path):
    """When reasoning key is absent and no env configured, returns None."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_NON_REASONING_STRENGTH", raising=False)

    env_file.write_text("# no reasoning config\n", encoding="utf-8")
    result = resolve_null_reasoning()
    assert result is None


def test_pro_model_promotes_low_effort_to_medium():
    """Pro model with effort 'low' should be promoted to 'medium'."""
    body = {"model": "oca/gpt-5.4-pro", "reasoning": {"effort": "low", "summary": "auto"}}
    enforce_pro_model_min_reasoning(body)
    assert body["reasoning"]["effort"] == "medium"


def test_pro_model_promotes_none_effort_to_medium():
    """Pro model with effort 'none' should be promoted to 'medium'."""
    body = {"model": "oca/gpt-5.4-pro", "reasoning": {"effort": "none"}}
    enforce_pro_model_min_reasoning(body)
    assert body["reasoning"]["effort"] == "medium"


def test_pro_model_promotes_minimal_effort_to_medium():
    """Pro model with effort 'minimal' should be promoted to 'medium'."""
    body = {"model": "oca/gpt-5.4-pro", "reasoning": {"effort": "minimal"}}
    enforce_pro_model_min_reasoning(body)
    assert body["reasoning"]["effort"] == "medium"


def test_pro_model_keeps_high_effort():
    """Pro model with effort 'high' should remain unchanged."""
    body = {"model": "oca/gpt-5.4-pro", "reasoning": {"effort": "high", "summary": "auto"}}
    enforce_pro_model_min_reasoning(body)
    assert body["reasoning"]["effort"] == "high"


def test_pro_model_keeps_xhigh_effort():
    """Pro model with effort 'xhigh' should remain unchanged."""
    body = {"model": "oca/gpt-5.4-pro", "reasoning": {"effort": "xhigh"}}
    enforce_pro_model_min_reasoning(body)
    assert body["reasoning"]["effort"] == "xhigh"


def test_pro_model_keeps_medium_effort():
    """Pro model with effort 'medium' should remain unchanged."""
    body = {"model": "oca/gpt-5.4-pro", "reasoning": {"effort": "medium"}}
    enforce_pro_model_min_reasoning(body)
    assert body["reasoning"]["effort"] == "medium"


def test_pro_model_adds_reasoning_when_missing():
    """Pro model with no reasoning should get medium effort added."""
    body = {"model": "oca/gpt-5.4-pro"}
    enforce_pro_model_min_reasoning(body)
    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}


def test_non_pro_model_keeps_low_effort():
    """Non-pro model with effort 'low' should remain unchanged."""
    body = {"model": "oca/gpt-5.4", "reasoning": {"effort": "low"}}
    enforce_pro_model_min_reasoning(body)
    assert body["reasoning"]["effort"] == "low"


def test_non_pro_model_no_reasoning_unchanged():
    """Non-pro model without reasoning should remain unchanged."""
    body = {"model": "oca/gpt-5.4"}
    enforce_pro_model_min_reasoning(body)
    assert "reasoning" not in body
