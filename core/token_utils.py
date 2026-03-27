"""
Token refresh utilities.

Provides force_refresh_token() to force-refresh both the access token and refresh
token, with a configurable interval guard to avoid unnecessary churn.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import dotenv_values

from core.oauth2_token_manager import OCAOauth2TokenManager, get_key, set_key


def force_refresh_token(
    dotenv_path: str = ".env",
    interval_hours: float = 6.0,
    force: bool = True,
) -> str:
    """
    Force-refresh both access token and refresh token if needed.

    Persists OAUTH_LAST_FORCED_REFRESH (ISO timestamp) to .env so that callers
    with force=False can skip unnecessary refreshes when the token was recently
    rotated by another process.

    Args:
        dotenv_path: Path to the .env file.
        interval_hours: Minimum elapsed hours before a non-forced refresh is allowed.
        force: If True, refresh regardless of interval. If False, only refresh when
            (now - last_forced) >= interval_hours.

    Returns:
        The new access token.

    Raises:
        ValueError: If OAUTH_REFRESH_TOKEN is not found in .env.
        RuntimeError: If the token refresh call fails.
    """
    last_forced_raw = get_key(dotenv_path, "OAUTH_LAST_FORCED_REFRESH")

    should_refresh = force
    if not force and last_forced_raw:
        try:
            last_forced = datetime.fromisoformat(last_forced_raw).replace(tzinfo=timezone.utc)
            elapsed = datetime.now(timezone.utc) - last_forced
            should_refresh = elapsed >= timedelta(hours=interval_hours)
        except ValueError:
            should_refresh = True

    if not should_refresh:
        tm = OCAOauth2TokenManager(dotenv_path=dotenv_path, debug=False)
        token = tm.get_access_token()
        print(f"[token_utils] Token still valid (forced refresh within {interval_hours}h).")
        return token

    # Clear access token so _refresh_tokens() is triggered inside get_access_token()
    _clear_access_token(dotenv_path)

    tm = OCAOauth2TokenManager(dotenv_path=dotenv_path, debug=True)
    token = tm.get_access_token()

    set_key(dotenv_path, "OAUTH_LAST_FORCED_REFRESH", datetime.now(timezone.utc).isoformat())

    return token


def _clear_access_token(dotenv_path: str) -> None:
    """
    Remove OAUTH_ACCESS_TOKEN=... from .env, leaving the key present with no value.
    This ensures OCAOauth2TokenManager will not use a stale cached token and will
    call _refresh_tokens() on the next get_access_token() call.
    """
    env_file = Path(dotenv_path)
    lines = env_file.read_text().splitlines()
    cleared = []
    for line in lines:
        if line.startswith("OAUTH_ACCESS_TOKEN="):
            cleared.append("OAUTH_ACCESS_TOKEN=")
        else:
            cleared.append(line)
    env_file.write_text("\n".join(cleared) + "\n")
