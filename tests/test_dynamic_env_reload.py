from responses_api import resolve_model_name
from responses_passthrough import resolve_passthrough_model, resolve_reasoning_effort, resolve_null_reasoning


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
