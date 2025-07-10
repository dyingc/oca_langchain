import os
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from dotenv import load_dotenv, get_key, set_key

class OcaOauth2TokenManager:
    """
    管理 OAuth2 令牌，包括自动刷新和持久化。
    内置网络连通性检查和代理回退机制。
    """
    def __init__(self, dotenv_path: str = ".env"):
        """
        初始化 Token 管理器。
        - 加载配置
        - 检测网络连通性并确定是否使用代理
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

        self.proxies: Dict[str, str] = {}
        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None

        # 执行网络检查并设置代理
        self._check_network_and_set_proxies()

        # 尝试从 .env 加载已存在的 access token
        self._load_token_from_env()
        print("OcaOauth2TokenManager 初始化成功。")

    def _check_network_and_set_proxies(self):
        """检查网络连通性，如果直连失败则尝试使用代理。"""
        test_url = f"https://{self.host}" # 使用认证主机作为测试目标
        print(f"正在测试网络连通性，目标: {test_url}")

        try:
            # 尝试直接连接 (设置一个较短的超时时间)
            requests.get(test_url, timeout=5)
            print("网络直连成功，无需代理。")
            self.proxies = {}
            return
        except requests.exceptions.RequestException as e:
            print(f"网络直连失败: {e}")

        # 如果直连失败，并且配置了代理URL，则尝试使用代理
        if self.proxy_url:
            print(f"直连失败，尝试使用代理: {self.proxy_url}")
            # 为 httpx 和 requests 创建兼容的代理字典
            proxies = {
                "http://": self.proxy_url,
                "https://": self.proxy_url,
            }
            try:
                # requests 也能理解这种格式的 key
                requests.get(test_url, proxies=proxies, timeout=10)
                print("通过代理连接成功。")
                self.proxies = proxies
                return
            except requests.exceptions.RequestException as e:
                print(f"通过代理连接失败: {e}")
                raise ConnectionError(
                    f"无法通过代理 {self.proxy_url} 连接到 {test_url}。请检查您的网络和代理设置。"
                ) from e

        # 如果没有配置代理或代理也失败了
        raise ConnectionError(
            f"无法连接到 {test_url}。请检查您的网络设置或配置一个有效的 HTTP_PROXY_URL。"
        )

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
            # 在所有请求中使用已确定的代理配置
            response = requests.post(token_url, headers=headers, data=data, timeout=10, proxies=self.proxies)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
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
