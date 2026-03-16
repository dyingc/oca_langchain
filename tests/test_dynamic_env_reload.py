from responses_api import resolve_model_name
from responses_passthrough import resolve_passthrough_model, resolve_reasoning_effort, resolve_null_reasoning, enforce_pro_model_min_reasoning


def test_responses_api_model_reload_from_env_each_call(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_api._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    env_file.write_text('LLM_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_name("gpt-4o") == "oca/gpt-5.1"

    env_file.write_text('LLM_MODEL_NAME="oca/gpt-5.2"\n', encoding="utf-8")
    assert resolve_model_name("gpt-4o") == "oca/gpt-5.2"


def test_responses_api_keeps_incoming_oca_model(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_api._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    env_file.write_text('LLM_MODEL_NAME="oca/gpt-5.9"\n', encoding="utf-8")
    assert resolve_model_name("oca/gpt-5.1-codex") == "oca/gpt-5.1-codex"


def test_responses_api_model_falls_back_to_incoming_when_env_removed(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_api._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    env_file.write_text('LLM_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_name("gpt-4o") == "oca/gpt-5.1"

    env_file.write_text("# LLM_MODEL_NAME removed\n", encoding="utf-8")
    assert resolve_model_name("gpt-4o") == "gpt-4o"


def test_passthrough_model_reload_from_env_each_call(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_passthrough_model("gpt-4o") == "oca/gpt-5.1"

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.2"\n', encoding="utf-8")
    assert resolve_passthrough_model("gpt-4o") == "oca/gpt-5.2"


def test_passthrough_keeps_incoming_oca_model(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.9"\n', encoding="utf-8")
    assert resolve_passthrough_model("oca/gpt-5.1-codex") == "oca/gpt-5.1-codex"


def test_passthrough_model_falls_back_when_env_removed(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_passthrough_model("gpt-4o") == "oca/gpt-5.1"

    env_file.write_text("# LLM_RESPONSES_MODEL_NAME removed\n", encoding="utf-8")
    assert resolve_passthrough_model("gpt-4o") == "oca/gpt-4o"


def test_passthrough_effort_reload_from_env_each_call(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_REASONING_STRENGTH", raising=False)

    env_file.write_text('LLM_REASONING_STRENGTH="high"\n', encoding="utf-8")
    assert resolve_reasoning_effort("low") == "high"

    env_file.write_text('LLM_REASONING_STRENGTH="minimal"\n', encoding="utf-8")
    assert resolve_reasoning_effort("low") == "minimal"


def test_passthrough_effort_falls_back_to_incoming_when_env_removed(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_REASONING_STRENGTH", raising=False)

    env_file.write_text('LLM_REASONING_STRENGTH="high"\n', encoding="utf-8")
    assert resolve_reasoning_effort("low") == "high"

    env_file.write_text("# LLM_REASONING_STRENGTH removed\n", encoding="utf-8")
    assert resolve_reasoning_effort("low") == "low"


def test_passthrough_missing_reasoning_treated_as_null(monkeypatch, tmp_path):
    """When reasoning key is completely absent, it should be filled like null."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_NON_REASONING_STRENGTH", raising=False)

    env_file.write_text('LLM_NON_REASONING_STRENGTH="medium"\n', encoding="utf-8")
    result = resolve_null_reasoning()
    assert result == {"effort": "medium", "summary": "auto"}


def test_passthrough_missing_reasoning_no_env(monkeypatch, tmp_path):
    """When reasoning key is absent and no env configured, returns None."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
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
