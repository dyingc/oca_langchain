from core.codex_config_generator import (
    choose_default_model,
    extract_model_catalog,
    fetch_model_info,
    get_access_token,
    render_config,
)


def test_extract_model_catalog_keeps_supported_api_metadata():
    payload = {
        "data": [
            {
                "litellm_params": {"model": "oca/gpt-5.3-codex"},
                "model_info": {
                    "supported_api_list": ["RESPONSES"],
                    "reasoning_effort_options": ["low", "medium", "high"],
                },
            },
            {
                "litellm_params": {"model": "oca/gpt-4.1"},
                "model_info": {
                    "supported_api_list": ["CHAT_COMPLETIONS", "RESPONSES"],
                    "reasoning_effort_options": [],
                },
            },
        ]
    }

    catalog = extract_model_catalog(payload)

    assert catalog["oca/gpt-5.3-codex"]["supported_api_list"] == ["RESPONSES"]
    assert catalog["oca/gpt-5.3-codex"]["reasoning_effort_options"] == ["low", "medium", "high"]
    assert catalog["oca/gpt-4.1"]["supported_api_list"] == ["CHAT_COMPLETIONS", "RESPONSES"]


def test_choose_default_model_prefers_gpt_5_3_codex_when_available():
    catalog = {
        "oca/gpt-4.1": {"supported_api_list": ["CHAT_COMPLETIONS", "RESPONSES"]},
        "oca/gpt-5.3-codex": {"supported_api_list": ["RESPONSES"]},
        "oca/gpt-5.4": {"supported_api_list": ["CHAT_COMPLETIONS", "RESPONSES"]},
    }

    assert choose_default_model(catalog) == "oca/gpt-5.3-codex"


def test_render_config_includes_only_available_profiles_and_sets_defaults():
    catalog = {
        "oca/gpt-4.1": {"supported_api_list": ["CHAT_COMPLETIONS", "RESPONSES"]},
        "oca/gpt-5.3-codex": {
            "supported_api_list": ["RESPONSES"],
            "reasoning_effort_options": ["low", "medium", "high"],
        },
        "oca/gpt-5.4": {"supported_api_list": ["CHAT_COMPLETIONS", "RESPONSES"]},
    }

    rendered = render_config(
        catalog=catalog,
        default_model="oca/gpt-5.3-codex",
        default_profile="gpt-5-3-codex",
    )

    assert 'model = "oca/gpt-5.3-codex"' in rendered
    assert 'profile = "gpt-5-3-codex"' in rendered
    assert '[profiles.gpt-5-3-codex]' in rendered
    assert '[profiles.gpt-5-4]' in rendered
    assert '[profiles.gpt-5-4-pro]' not in rendered
    assert 'review_model = "oca/gpt-5.3-codex"' in rendered


def test_get_access_token_uses_oauth_manager_refresh_logic(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'OAUTH_HOST="example.oraclecloud.com"',
                'OAUTH_CLIENT_ID="client-id"',
                'OAUTH_REFRESH_TOKEN="refresh-token"',
                'OAUTH_ACCESS_TOKEN="expired-access-token"',
                'OAUTH_ACCESS_TOKEN_EXPIRES_AT="2000-01-01T00:00:00+00:00"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    class FakeTokenManager:
        def __init__(self, dotenv_path: str, debug: bool = False):
            calls.append(f"init:{dotenv_path}:{debug}")

        def get_access_token(self) -> str:
            calls.append("get_access_token")
            return "fresh-access-token"

    monkeypatch.setattr("core.codex_config_generator.OCAOauth2TokenManager", FakeTokenManager)

    token = get_access_token(env_file)

    assert token == "fresh-access-token"
    assert calls == [f"init:{env_file}:False", "get_access_token"]


def test_fetch_model_info_uses_oauth_manager_request_transport():
    calls: list[object] = []

    class FakeResponse:
        def json(self) -> dict:
            return {"data": [{"litellm_params": {"model": "oca/gpt-5.3-codex"}, "model_info": {}}]}

    class FakeTokenManager:
        def request(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse()

    payload = fetch_model_info(
        token_manager=FakeTokenManager(),
        access_token="fresh-access-token",
        model_info_url="https://example.invalid/v1/model/info",
    )

    assert payload["data"][0]["litellm_params"]["model"] == "oca/gpt-5.3-codex"
    assert calls == [
        {
            "method": "GET",
            "url": "https://example.invalid/v1/model/info",
            "headers": {
                "Authorization": "Bearer fresh-access-token",
                "Accept": "application/json",
            },
            "request_timeout": 30,
        }
    ]
