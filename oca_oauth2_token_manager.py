import os
import requests
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, AsyncIterator
from urllib.parse import urlparse
from enum import Enum

from dotenv import load_dotenv, get_key, set_key

class ConnectionMode(Enum):
    DIRECT = "direct"
    PROXY = "proxy"

class OCAOauth2TokenManager:
    """
    管理 OAuth2 令牌，包括自动刷新和持久化。
    内置网络连通性检查和代理回退机制。
    """
    def __init__(self, dotenv_path: str = ".env"):
        """
        初始化 Token 管理器。
        - 加载配置
        - 尝试从 .env 文件加载已有的有效 Access Token
        """
        if not os.path.exists(dotenv_path):
            raise FileNotFoundError(f"错误：指定的 .env 文件路径不存在 -> {dotenv_path}")

        self.dotenv_path: str = dotenv_path
        load_dotenv(self.dotenv_path)

        self.host: Optional[str] = os.getenv("OAUTH_HOST")
        self.client_id: Optional[str] = os.getenv("OAUTH_CLIENT_ID")
        self.proxy_url: Optional[str] = os.getenv("HTTP_PROXY_URL")

        if not all([self.host, self.client_id]):
            raise ValueError("错误：请确保 .env 文件中包含 OAUTH_HOST 和 OAUTH_CLIENT_ID。")

        self.connection_mode: ConnectionMode = ConnectionMode.DIRECT
        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None

        # 网络超时时间：优先从env获取，否则默认2秒
        self.timeout: float = 20.0
        try:
            timeout_str = get_key(self.dotenv_path, "CONNECTION_TIMEOUT")
            if timeout_str:
                self.timeout = float(timeout_str)
        except Exception:
            pass

        # 尝试从 .env 加载已存在的 access token
        self._load_token_from_env()
        print("OcaOauth2TokenManager 初始化成功。")
        print(f"当前连接模式: {self.connection_mode.value}")

    def _load_token_from_env(self):
        """尝试从 .env 文件加载并验证 Access Token。"""
        token = get_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN")
        expires_at_str = get_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN_EXPIRES_AT")

        if token and expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now(timezone.utc) < expires_at:
                self.access_token = token
                self.expires_at = expires_at
                print("从 .env 文件加载了有效的 Access Token。")

    def _get_proxies(self, mode: ConnectionMode) -> Optional[Dict[str, str]]:
        if mode == ConnectionMode.PROXY:
            if self.proxy_url:
                return {"http": self.proxy_url, "https": self.proxy_url}
            else:
                print("警告: 连接模式为 PROXY，但未配置代理 URL。")
                return None
        return {} # direct connection

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """
        执行一个同步请求。
        它会根据当前的 self.connection_mode 尝试连接。如果失败，它会切换模式并立即重试一次。
        两次连续失败（即，在同一次调用中直连和代理都失败）将引发 ConnectionError。
        """
        primary_mode = self.connection_mode
        secondary_mode = ConnectionMode.PROXY if primary_mode == ConnectionMode.DIRECT else ConnectionMode.DIRECT

        # 尝试第一次
        print(f"正在尝试使用 {primary_mode.value} 模式连接 {url}")
        primary_proxies = self._get_proxies(primary_mode)
        
        if primary_mode == ConnectionMode.PROXY and primary_proxies is None:
            print("无法使用代理模式，切换到直连模式。")
            primary_mode, secondary_mode = secondary_mode, primary_mode
            primary_proxies = self._get_proxies(primary_mode)

        try:
            response = requests.request(method, url, timeout=self.timeout, proxies=primary_proxies, **kwargs)
            response.raise_for_status()
            print(f"使用 {primary_mode.value} 模式连接 {url} 成功。")
            return response
        except requests.exceptions.RequestException as e:
            print(f"使用 {primary_mode.value} 模式连接失败: {e}")
            self.connection_mode = secondary_mode
            print(f"连接模式已切换至 {self.connection_mode.value}，下次请求将使用此模式。")

            secondary_proxies = self._get_proxies(secondary_mode)
            if secondary_mode == ConnectionMode.PROXY and secondary_proxies is None:
                raise ConnectionError(f"无法连接到 {url}。{primary_mode.value} 模式失败且无代理可供重试。") from e

            # 尝试第二次
            print(f"立即使用 {secondary_mode.value} 模式重试...")
            try:
                response = requests.request(method, url, timeout=self.timeout, proxies=secondary_proxies, **kwargs)
                response.raise_for_status()
                print(f"使用 {secondary_mode.value} 模式重试成功。")
                return response
            except requests.exceptions.RequestException as e2:
                print(f"使用 {secondary_mode.value} 模式重试失败: {e2}")
                raise ConnectionError(f"无法连接到 {url}。{primary_mode.value} 和 {secondary_mode.value} 模式均失败。") from e2

    async def async_stream_request(self, method: str, url: str, **kwargs: Any) -> AsyncIterator[str]:
        """
        执行一个异步流式请求。
        逻辑与同步的 request 方法相同：尝试主模式，失败则切换并用备用模式重试。
        """
        primary_mode = self.connection_mode
        secondary_mode = ConnectionMode.PROXY if primary_mode == ConnectionMode.DIRECT else ConnectionMode.DIRECT

        # 尝试第一次
        print(f"正在尝试使用 {primary_mode.value} 模式的异步流式请求连接 {url}")
        primary_proxy_config = self.proxy_url if primary_mode == ConnectionMode.PROXY else None

        if primary_mode == ConnectionMode.PROXY and not primary_proxy_config:
            print("无法使用代理模式，切换到直连模式。")
            primary_mode, secondary_mode = secondary_mode, primary_mode
            primary_proxy_config = None

        try:
            async with httpx.AsyncClient(proxy=primary_proxy_config) as client:
                async with client.stream(method, url, timeout=self.timeout, **kwargs) as response:
                    response.raise_for_status()
                    print(f"使用 {primary_mode.value} 模式的流式连接 {url} 成功。")
                    async for line in response.aiter_lines():
                        yield line
            return
        except httpx.RequestError as e:
            print(f"使用 {primary_mode.value} 模式的流式连接失败: {e}")
            self.connection_mode = secondary_mode
            print(f"连接模式已切换至 {self.connection_mode.value}，下次请求将使用此模式。")

            secondary_proxy_config = self.proxy_url if secondary_mode == ConnectionMode.PROXY else None
            if secondary_mode == ConnectionMode.PROXY and not secondary_proxy_config:
                raise ConnectionError(f"无法连接到 {url}。{primary_mode.value} 模式失败且无代理可供重试。") from e

            # 尝试第二次
            print(f"立即使用 {secondary_mode.value} 模式重试流式请求...")
            try:
                async with httpx.AsyncClient(proxy=secondary_proxy_config) as client:
                    async with client.stream(method, url, timeout=self.timeout, **kwargs) as response:
                        response.raise_for_status()
                        print(f"使用 {secondary_mode.value} 模式的流式重试成功。")
                        async for line in response.aiter_lines():
                            yield line
                return
            except httpx.RequestError as e2:
                print(f"使用 {secondary_mode.value} 模式的流式重试失败: {e2}")
                raise ConnectionError(f"无法连接到 {url}。{primary_mode.value} 和 {secondary_mode.value} 模式的流式请求均失败。") from e2

    def _refresh_tokens(self) -> None:
        """
        使用 Refresh Token 换取新的 Access Token 和 Refresh Token。
        并将新 Token 持久化到 .env 文件。
        """
        current_refresh_token = get_key(self.dotenv_path, "OAUTH_REFRESH_TOKEN")
        if not current_refresh_token:
            raise ValueError(f"错误：在 {self.dotenv_path} 文件中找不到 OAUTH_REFRESH_TOKEN。")

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

        print(f"正在向 {token_url} 发送请求以刷新令牌...")
        try:
            response = self.request(
                method="POST",
                url=token_url,
                headers=headers,
                data=data
            )
        except ConnectionError as e:
            print(f"刷新令牌失败: {e}")
            raise

        response_data = response.json()

        self.access_token = response_data["access_token"]
        expires_in = response_data["expires_in"]
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
        print("Access Token 已在内存中更新。")

        set_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN", self.access_token)
        set_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN_EXPIRES_AT", self.expires_at.isoformat())
        print(f"Access Token 和过期时间已写入 {self.dotenv_path}。")

        if "refresh_token" in response_data:
            new_refresh_token = response_data["refresh_token"]
            if set_key(self.dotenv_path, "OAUTH_REFRESH_TOKEN", new_refresh_token):
                print(f"Refresh Token 已在 {self.dotenv_path} 文件中更新。")
            else:
                print(f"警告: 更新 {self.dotenv_path} 文件失败！")

    def get_access_token(self) -> str:
        """
        获取一个有效的 Access Token。如果当前令牌无效或已过期，则自动刷新。
        """
        if self.access_token and self.expires_at and datetime.now(timezone.utc) < self.expires_at:
            print("使用内存中缓存的有效 Access Token。")
            return self.access_token

        print("Access Token 已过期或不存在，正在启动刷新流程...")
        self._refresh_tokens()
        if self.access_token:
            return self.access_token
        else:
            raise ValueError("刷新后未能获取有效的 Access Token。")