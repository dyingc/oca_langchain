import warnings

from urllib3.exceptions import InsecureRequestWarning

from core.oauth2_token_manager import _request_with_warning_control


def test_request_with_warning_control_suppresses_loopback_warning(monkeypatch):
    def fake_request(*args, **kwargs):
        warnings.warn("unverified loopback request", InsecureRequestWarning)
        return "ok"

    monkeypatch.setattr("core.oauth2_token_manager.requests.request", fake_request)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _request_with_warning_control(
            "GET",
            "https://127.0.0.1/test",
            verify=False,
        )

    assert result == "ok"
    assert not any(issubclass(w.category, InsecureRequestWarning) for w in caught)


def test_request_with_warning_control_keeps_remote_warning(monkeypatch):
    def fake_request(*args, **kwargs):
        warnings.warn("unverified remote request", InsecureRequestWarning)
        return "ok"

    monkeypatch.setattr("core.oauth2_token_manager.requests.request", fake_request)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _request_with_warning_control(
            "GET",
            "https://example.test/api",
            verify=False,
        )

    assert result == "ok"
    assert any(issubclass(w.category, InsecureRequestWarning) for w in caught)
