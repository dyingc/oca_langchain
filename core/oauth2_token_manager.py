import os
import requests
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, AsyncIterator, Callable
from urllib.parse import urlparse
from enum import Enum

from dotenv import load_dotenv, get_key, set_key

class ConnectionMode(Enum):
    DIRECT = "direct"
    PROXY = "proxy"

class OCAOauth2TokenManager:
    """
    Manage OAuth2 tokens, including automatic refresh and persistence.
    Includes built-in network connectivity check and proxy fallback mechanism.
    """
    def __init__(self, dotenv_path: str = ".env", debug: bool = False):
        """
        Initialize Token Manager.
        - Load configuration
        - Try loading any existing valid Access Token from .env file
        """
        if not os.path.exists(dotenv_path):
            raise FileNotFoundError(f"Error: The specified .env file path does not exist -> {dotenv_path}")

        self.dotenv_path: str = dotenv_path
        load_dotenv(self.dotenv_path)

        self.host: Optional[str] = os.getenv("OAUTH_HOST")
        self.client_id: Optional[str] = os.getenv("OAUTH_CLIENT_ID")
        self.proxy_url: Optional[str] = os.getenv("HTTP_PROXY_URL")
        self._debug: bool = debug

        # Combine system CA bundle with additional certificates from MULTI_CA_BUNDLE (comma-separated)
        try:
            extra_cas_raw = os.getenv("MULTI_CA_BUNDLE", "")
            extra_cas = [p.strip() for p in extra_cas_raw.split(",") if p.strip()]
            if extra_cas:
                import certifi, tempfile, shutil, pathlib
                system_pem = certifi.where()
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
                tmp_path = tmp_file.name
                tmp_file.close()
                with open(system_pem, "rb") as f_sys, open(tmp_path, "wb") as f_out:
                    shutil.copyfileobj(f_sys, f_out)
                    for ca_path in extra_cas:
                        p = pathlib.Path(ca_path)
                        if p.exists():
                            try:
                                with open(ca_path, "rb") as f_ca:
                                    # Ensure there is a newline separator between bundles
                                    f_out.write(b"\n")
                                    shutil.copyfileobj(f_ca, f_out)
                            except Exception:
                                if self._debug:
                                    print(f"Warning: failed to append CA file: {ca_path}")
                        else:
                            if self._debug:
                                print(f"Warning: CA file not found, skipping: {ca_path}")
                # Respect explicit REQUESTS_CA_BUNDLE if already set
                if not os.getenv("REQUESTS_CA_BUNDLE"):
                    os.environ["REQUESTS_CA_BUNDLE"] = tmp_path
                # Also set SSL_CERT_FILE for httpx if not explicitly provided
                if not os.getenv("SSL_CERT_FILE"):
                    os.environ["SSL_CERT_FILE"] = tmp_path
                if self._debug:
                    print(f"Combined CA bundle created at {tmp_path}")
        except Exception:
            # Any failure in CA merge should not prevent startup; fall back to defaults
            pass

        if not all([self.host, self.client_id]):
            raise ValueError("Error: Please ensure .env contains both OAUTH_HOST and OAUTH_CLIENT_ID.")

        self.connection_mode: ConnectionMode = ConnectionMode.DIRECT
        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None

        # Network timeout: try to get from env, otherwise defaults to 2 seconds
        self.timeout: float = 2.0
        try:
            timeout_str = get_key(self.dotenv_path, "CONNECTION_TIMEOUT")
            if timeout_str:
                self.timeout = float(timeout_str)
        except Exception:
            pass

        # Try loading any existing access token from .env
        self._load_token_from_env()
        # Initialize connection mode from FORCE_PROXY env if set
        self._update_connection_mode_from_env()
        if self._debug:
            print("OcaOauth2TokenManager initialized successfully.")
            print(f"Current connection mode: {self.connection_mode.value}")

    def _load_token_from_env(self):
        """Try to load and validate Access Token from .env file."""
        token = get_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN")
        expires_at_str = get_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN_EXPIRES_AT")

        if token and expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now(timezone.utc) < expires_at:
                self.access_token = token
                self.expires_at = expires_at
                if self._debug:
                    print("Loaded a valid Access Token from .env file.")

    def _update_connection_mode_from_env(self) -> None:
        """
        Dynamically enforce proxy usage based on FORCE_PROXY env.
        If FORCE_PROXY=true, always switch to PROXY mode before sending the request.
        If FORCE_PROXY is false or unset, keep current connection_mode to preserve existing auto-logic.
        """
        # Reload dotenv so edits to .env take effect without restarting the process
        try:
            load_dotenv(self.dotenv_path, override=True)
        except Exception:
            pass
        # Refresh proxy_url if HTTP_PROXY_URL changed in environment
        try:
            current_proxy = os.getenv("HTTP_PROXY_URL")
            if current_proxy != self.proxy_url:
                self.proxy_url = current_proxy
                if self._debug:
                    print(f"HTTP_PROXY_URL updated to: {self.proxy_url or '<unset>'}")
        except Exception:
            pass
        try:
            force_proxy = os.getenv("FORCE_PROXY", "false").lower() == "true"
        except Exception:
            force_proxy = False
        if force_proxy and self.connection_mode != ConnectionMode.PROXY:
            if self._debug:
                print("FORCE_PROXY enabled — switching to proxy mode for all requests.")
            self.connection_mode = ConnectionMode.PROXY
        # 当处于 PROXY 模式时，显式在环境变量中写入 *HTTP(S)_PROXY 以兼容部分库的自动代理解析
        if self.connection_mode == ConnectionMode.PROXY and self.proxy_url:
            for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
                os.environ[k] = self.proxy_url
        else:
            # 清理遗留值，避免 DIRECT 模式误走代理
            for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
                os.environ.pop(k, None)

    def _get_proxies(self, mode: ConnectionMode) -> Optional[Dict[str, str]]:
        if mode == ConnectionMode.PROXY:
            if self.proxy_url:
                return {"http": self.proxy_url, "https": self.proxy_url}
            else:
                if self._debug:
                    print("Warning: Connection mode is PROXY but no proxy URL is configured.")
                return None
        return {} # direct connection

    def request(self, method: str, url: str, _do_retry: bool = True, request_timeout: Optional[float] = None, force_disable_verify: bool = False, **kwargs: Any) -> requests.Response:
        """
        Perform a synchronous request.
        Set force_disable_verify=True to explicitly disable SSL verification for this call.
        If _do_retry is True, it tries using the current self.connection_mode. If it fails, switches mode and retries once.
        If both direct and proxy fails in the same call, raises ConnectionError.
        If _do_retry is False, only try once; failure immediately raises ConnectionError.
        """
        # Dynamic check for FORCE_PROXY each request
        self._update_connection_mode_from_env()
        primary_mode = self.connection_mode
        secondary_mode = ConnectionMode.PROXY if primary_mode == ConnectionMode.DIRECT else ConnectionMode.DIRECT

        # First attempt
        if self._debug:
            print(f"Trying to connect to {url} with mode {primary_mode.value}")
        primary_proxies = self._get_proxies(primary_mode)

        if primary_mode == ConnectionMode.PROXY and primary_proxies is None:
            if self._debug:
                print("Cannot use proxy mode, switching to direct mode.")
            primary_mode, secondary_mode = secondary_mode, primary_mode
            primary_proxies = self._get_proxies(primary_mode)

        ca_bundle = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")

        # Global or per-call switch to disable SSL verification
        force_disable_env = os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true"
        force_disable = force_disable_verify or force_disable_env
        primary_verify = False if (force_disable or primary_mode == ConnectionMode.PROXY) else ca_bundle

        try:
            response = requests.request(method, url, timeout=request_timeout if request_timeout is not None else self.timeout, proxies=primary_proxies, verify=primary_verify, **kwargs)
            response.raise_for_status()
            if self._debug:
                print(f"Successfully connected to {url} with mode {primary_mode.value}.")
            return response
        except requests.exceptions.RequestException as e:
            if self._debug:
                print(f"Connection with {primary_mode.value} mode failed: {e}")
            if not _do_retry:
                raise ConnectionError(f"Unable to connect to {url}. Retry is disabled.") from e

            self.connection_mode = secondary_mode
            if self._debug:
                print(f"Connection mode switched to {self.connection_mode.value}, will use this for next request.")

            secondary_proxies = self._get_proxies(secondary_mode)
            if secondary_mode == ConnectionMode.PROXY and secondary_proxies is None:
                raise ConnectionError(f"Unable to connect to {url}. {primary_mode.value} mode failed and no proxy available for retry.") from e

            # Second attempt
            if self._debug:
                print(f"Retrying immediately with {secondary_mode.value} mode...")

            secondary_verify = False if (force_disable or secondary_mode == ConnectionMode.PROXY) else ca_bundle
            try:
                response = requests.request(method, url, timeout=request_timeout if request_timeout is not None else self.timeout, proxies=secondary_proxies, verify=secondary_verify, **kwargs)
                response.raise_for_status()
                if self._debug:
                    print(f"Retry with {secondary_mode.value} mode succeeded.")
                return response
            except requests.exceptions.RequestException as e2:
                if self._debug:
                    print(f"Retry with {secondary_mode.value} mode failed: {e2}")
                raise ConnectionError(f"Unable to connect to {url}. Both {primary_mode.value} and {secondary_mode.value} modes failed.") from e2

    async def async_stream_request(self, method: str, url: str, _do_retry: bool = True, request_timeout: Optional[float] = None, on_open: Optional[Callable[[httpx.Response], None]] = None, **kwargs: Any) -> AsyncIterator[str]:
        """
        Perform an asynchronous streaming request.
        If _do_retry is True, uses the same logic as sync: try primary mode, failover to alternate mode and retry.
        If _do_retry is False, try once only, on fail raise ConnectionError immediately.
        """
        # Dynamic check for FORCE_PROXY each async request
        self._update_connection_mode_from_env()
        primary_mode = self.connection_mode
        secondary_mode = ConnectionMode.PROXY if primary_mode == ConnectionMode.DIRECT else ConnectionMode.DIRECT

        # First attempt
        if self._debug:
            print(f"Trying async streaming request to {url} with mode {primary_mode.value}")
        primary_proxy_config = self.proxy_url if primary_mode == ConnectionMode.PROXY else None

        if primary_mode == ConnectionMode.PROXY and not primary_proxy_config:
            if self._debug:
                print("Cannot use proxy mode, switching to direct mode.")
            primary_mode, secondary_mode = secondary_mode, primary_mode
            primary_proxy_config = None

        try:
            ca_bundle = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
            if self._debug:
                print(f"Using CA bundle for httpx: {ca_bundle or '<default>'}")
                print(f"Using proxy for httpx: {'yes' if primary_proxy_config else 'no'}")

            # 直接让 httpx 处理 HTTPS CONNECT；无需手动重写 URL

            # 构造 httpx 传输层；避免 verify/proxy 重复并禁用环境代理变量
            # 使用 AsyncHTTPTransport 明确设置 proxy，避免某些版本不支持 AsyncClient.proxies
            transport = None
            if primary_proxy_config:
                transport = httpx.AsyncHTTPTransport(
                    proxy=primary_proxy_config,
                    verify=(False if primary_proxy_config else ca_bundle),
                    trust_env=False,
                )

            async with httpx.AsyncClient(
                transport=transport,
                verify=(False if primary_proxy_config else ca_bundle),
                trust_env=False,
            ) as client:
                async with client.stream(method, url, timeout=request_timeout if request_timeout is not None else self.timeout, **kwargs) as response:
                    response.raise_for_status()
                    if on_open is not None:
                        try:
                            on_open(response)
                        except Exception:
                            pass
                    if self._debug:
                        print(f"Async streaming connection to {url} with mode {primary_mode.value} succeeded.")
                    async for line in response.aiter_lines():
                        yield line
            return
        except httpx.RequestError as e:
            if self._debug:
                print(f"Async streaming connection with {primary_mode.value} mode failed: {e}")
            if not _do_retry:
                raise ConnectionError(f"Unable to connect to {url}. Retry is disabled.") from e

            self.connection_mode = secondary_mode
            if self._debug:
                print(f"Connection mode switched to {self.connection_mode.value}, will use this for next request.")

            secondary_proxy_config = self.proxy_url if secondary_mode == ConnectionMode.PROXY else None
            if secondary_mode == ConnectionMode.PROXY and not secondary_proxy_config:
                raise ConnectionError(f"Unable to connect to {url}. {primary_mode.value} mode failed and no proxy available for retry.") from e

            # Second attempt
            if self._debug:
                print(f"Retrying async streaming request immediately with mode {secondary_mode.value}...")
            try:
                ca_bundle = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
                if self._debug:
                    print(f"Using CA bundle for httpx: {ca_bundle or '<default>'}")
                    print(f"Using proxy for httpx: {'yes' if secondary_proxy_config else 'no'}")

                mounts = None
                if secondary_proxy_config:
                    transport = httpx.AsyncHTTPTransport(proxy=secondary_proxy_config, verify=ca_bundle)
                    mounts = {"http://": transport, "https://": transport}

                async with httpx.AsyncClient(mounts=mounts, verify=ca_bundle) as client:
                    async with client.stream(method, url, timeout=request_timeout if request_timeout is not None else self.timeout, **kwargs) as response:
                        response.raise_for_status()
                        if on_open is not None:
                            try:
                                on_open(response)
                            except Exception:
                                pass
                        if self._debug:
                            print(f"Async streaming retry with {secondary_mode.value} mode succeeded.")
                        async for line in response.aiter_lines():
                            yield line
                return
            except httpx.RequestError as e2:
                if self._debug:
                    print(f"Async streaming retry with {secondary_mode.value} mode failed: {e2}")
                raise ConnectionError(f"Unable to connect to {url}. Both {primary_mode.value} and {secondary_mode.value} async streaming requests failed.") from e2

    def _refresh_tokens(self) -> None:
        """
        Use the Refresh Token to acquire new Access and Refresh tokens.
        Persists new tokens to the .env file.
        """
        current_refresh_token = get_key(self.dotenv_path, "OAUTH_REFRESH_TOKEN")
        if not current_refresh_token:
            raise ValueError(f"Error: OAUTH_REFRESH_TOKEN not found in {self.dotenv_path} file.")

        token_url = f"https://{self.host}/oauth2/v1/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": current_refresh_token,
        }

        if self._debug:
            print(f"Sending request to {token_url} to refresh token...")
        try:
            response = self.request(
                method="POST",
                url=token_url,
                headers=headers,
                data=data,
                _do_retry=False,
                force_disable_verify=True
            )
        except ConnectionError as e:
            if self._debug:
                print(f"Failed to refresh token: {e}")
            raise

        response_data = response.json()

        self.access_token = response_data["access_token"]
        expires_in = response_data["expires_in"]
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
        if self._debug:
            print("Access Token has been updated in memory.")

        set_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN", self.access_token)
        set_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN_EXPIRES_AT", self.expires_at.isoformat())
        if self._debug:
            print(f"Access Token and expiry have been written to {self.dotenv_path}.")

        if "refresh_token" in response_data:
            new_refresh_token = response_data["refresh_token"]
            if set_key(self.dotenv_path, "OAUTH_REFRESH_TOKEN", new_refresh_token):
                if self._debug:
                    print(f"Refresh Token has been updated in {self.dotenv_path}.")
            else:
                if self._debug:
                    print(f"Warning: Failed to update {self.dotenv_path} with new Refresh Token!")

    def get_access_token(self) -> str:
        """
        Obtain a valid Access Token. If the current token is invalid or expired, automatically refresh.
        """
        if self.access_token and self.expires_at and datetime.now(timezone.utc) < self.expires_at:
            if self._debug:
                print("Using valid Access Token from memory cache.")
            return self.access_token

        if self._debug:
            print("Access Token expired or not found, starting refresh process...")
        self._refresh_tokens()
        if self.access_token:
            return self.access_token
        else:
            raise ValueError("Failed to obtain valid Access Token after refresh.")
