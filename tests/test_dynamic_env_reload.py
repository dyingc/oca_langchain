from responses_api import resolve_model_name
from responses_passthrough import resolve_passthrough_model, resolve_reasoning_effort


def test_responses_api_model_reload_from_env_each_call(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_api._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    env_file.write_text('LLM_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_name("gpt-4o") == "oca/gpt-5.1"

    env_file.write_text('LLM_MODEL_NAME="oca/gpt-5.2"\n', encoding="utf-8")
    assert resolve_model_name("gpt-4o") == "oca/gpt-5.2"


def test_passthrough_model_reload_from_env_each_call(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    env_file.write_text('LLM_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_passthrough_model("gpt-4o") == "oca/gpt-5.1"

    env_file.write_text('LLM_MODEL_NAME="oca/gpt-5.2"\n', encoding="utf-8")
    assert resolve_passthrough_model("gpt-4o") == "oca/gpt-5.2"


def test_passthrough_effort_reload_from_env_each_call(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_REASONING_STRENGTH", raising=False)

    env_file.write_text('LLM_REASONING_STRENGTH="high"\n', encoding="utf-8")
    assert resolve_reasoning_effort("low") == "high"

    env_file.write_text('LLM_REASONING_STRENGTH="minimal"\n', encoding="utf-8")
    assert resolve_reasoning_effort("low") == "minimal"
