import os
import pytest
from runtime_env import _get_runtime_env_value


def test_reads_value_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('MY_KEY="hello"\n', encoding="utf-8")
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("MY_KEY", raising=False)
    assert _get_runtime_env_value("MY_KEY") == "hello"


def test_returns_default_when_key_absent_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# no key\n", encoding="utf-8")
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("MY_KEY", raising=False)
    assert _get_runtime_env_value("MY_KEY", "fallback") == "fallback"


def test_falls_back_to_os_environ_when_file_missing(monkeypatch):
    monkeypatch.setattr("runtime_env._ENV_PATH", "/nonexistent/.env")
    monkeypatch.setenv("MY_KEY", "from_os")
    assert _get_runtime_env_value("MY_KEY") == "from_os"


def test_strips_whitespace_from_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('MY_KEY="  spaced  "\n', encoding="utf-8")
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("MY_KEY", raising=False)
    assert _get_runtime_env_value("MY_KEY") == "spaced"
