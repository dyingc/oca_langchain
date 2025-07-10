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

def request_with_auto_proxy(method: str, url: str, timeout: float, proxy_url: Optional[str], **kwargs: Any) -> requests.Response:
    """
    封装 requests 请求，先尝试直连，如果失败且配置了代理，则自动使用代理重试。
    """
    # 尝试直连
    print(f"正在尝试直连 {url}")
    try:
        response = requests.request(method, url, timeout=timeout, proxies={}, **kwargs)
        response.raise_for_status()
        print(f"直连 {url} 成功。")
        return response
    except requests.exceptions.RequestException as e:
        print(f"直连 {url} 失败: {e}")
        if proxy_url:
            # 如果直连失败且配置了代理，则尝试使用代理
            proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
            print(f"直连失败，尝试通过代理 {proxy_url} 连接 {url}")
            try:
                response = requests.request(method, url, timeout=timeout, proxies=proxies, **kwargs)
                response.raise_for_status()
                print(f"通过代理 {url} 成功。")
                return response
            except requests.exceptions.RequestException as proxy_e:
                print(f"通过代理 {url} 失败: {proxy_e}")
                raise ConnectionError(
                    f"无法连接到 {url}。直连和代理模式均失败。请检查您的网络设置或代理配置。"
                ) from proxy_e
        else:
            # 如果没有配置代理，则直接抛出直连失败的异常
            raise ConnectionError(
                f"无法连接到 {url}。请检查您的网络设置。"
            ) from e

async def async_request_with_auto_proxy(method: str, url: str, timeout: float, proxy_url: Optional[str], **kwargs: Any) -> httpx.Response:
    """
    封装 httpx 异步请求，先尝试直连，如果失败且配置了代理，则自动使用代理重试。
    """
    # 尝试直连
    print(f"正在尝试直连 {url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        print(f"直连 {url} 成功。")
        return response
    except httpx.RequestError as e:
        print(f"直连 {url} 失败: {e}")
        if proxy_url:
            # 如果直连失败且配置了代理，则尝试使用代理
            print(f"直连失败，尝试通过代理 {proxy_url} 连接 {url}")
            try:
                async with httpx.AsyncClient(proxy=proxy_url) as client:
                    response = await client.request(method, url, timeout=timeout, **kwargs)
                response.raise_for_status()
                print(f"通过代理 {url} 成功。")
                return response
            except httpx.RequestError as proxy_e:
                print(f"通过代理 {url} 失败: {proxy_e}")
                raise ConnectionError(
                    f"无法连接到 {url}。直连和代理模式均失败。请检查您的网络设置或代理配置。"
                ) from proxy_e
        else:
            # 如果没有配置代理，则直接抛出直连失败的异常
            raise ConnectionError(
                f"无法连接到 {url}。请检查您的网络设置。"
            ) from e

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

        # proxies 字段不再在初始化时设置，而是由 request_with_auto_proxy 动态处理
        self.proxies: Dict[str, str] = {}
        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None

        # 网络超时时间：优先从env获取，否则默认2秒
        self.timeout: float = 2.0
        try:
            timeout_str = get_key(self.dotenv_path, "CONNECTION_TIMEOUT")
            if timeout_str:
                self.timeout = float(timeout_str)
        except Exception:
            pass

        # 尝试从 .env 加载已存在的 access token
        self._load_token_from_env()
        print("OcaOauth2TokenManager 初始化成功。")

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
            # 使用新的 request_with_auto_proxy 函数
            response = request_with_auto_proxy(
                method="POST",
                url=token_url,
                timeout=self.timeout,
                proxy_url=self.proxy_url,
                headers=headers,
                data=data
            )
            response.raise_for_status()
        except ConnectionError as e:
            print(f"请求失败: {e}")
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
