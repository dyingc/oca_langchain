import os
from dotenv import dotenv_values

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _get_runtime_env_value(key: str, default: str = "") -> str:
    """Read env value with .env as source of truth when the file exists.

    This avoids stale os.environ values when a key is removed from .env
    while the process is still running.
    """
    if os.path.exists(_ENV_PATH):
        values = dotenv_values(_ENV_PATH)
        value = values.get(key)
        if value is None:
            return default
        return str(value).strip()
    return os.getenv(key, default).strip()
