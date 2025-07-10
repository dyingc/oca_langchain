import os
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv, get_key, set_key

class Oauth2TokenManager:
    """
    管理 OAuth2 令牌，包括自动刷新和持久化 Access Token 及 Refresh Token。
    """
    def __init__(self, dotenv_path: str = ".env"):
        """
        初始化 Token 管理器。
        会尝试从 .env 文件加载已有的有效 Access Token。
        """
        if not os.path.exists(dotenv_path):
            raise FileNotFoundError(f"错误：指定的 .env 文件路径不存在 -> {dotenv_path}")
        
        self.dotenv_path: str = dotenv_path
        load_dotenv(self.dotenv_path)
        
        self.host: Optional[str] = os.getenv("OAUTH_HOST")
        self.client_id: Optional[str] = os.getenv("OAUTH_CLIENT_ID")

        if not all([self.host, self.client_id]):
            raise ValueError("错误：请确保 .env 文件中包含 OAUTH_HOST 和 OAUTH_CLIENT_ID。")

        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        
        # 尝试从 .env 加载已存在的 access token
        self._load_token_from_env()
        print("Oauth2TokenManager 初始化成功。")

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
            response = requests.post(token_url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            raise

        response_data = response.json()

        # 更新内存中的 Access Token 和过期时间
        self.access_token = response_data["access_token"]
        expires_in = response_data["expires_in"]
        # 使用带时区的 UTC 时间，以保证比较的准确性
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
        print("Access Token 已在内存中更新。")

        # 将新的 Access Token 和过期时间持久化到 .env 文件
        set_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN", self.access_token)
        set_key(self.dotenv_path, "OAUTH_ACCESS_TOKEN_EXPIRES_AT", self.expires_at.isoformat())
        print(f"Access Token 和过期时间已写入 {self.dotenv_path}。")

        # 检查并更新 .env 文件中的 Refresh Token (Refresh Token Rotation)
        if "refresh_token" in response_data:
            new_refresh_token = response_data["refresh_token"]
            if set_key(self.dotenv_path, "OAUTH_REFRESH_TOKEN", new_refresh_token):
                print(f"Refresh Token 已在 {self.dotenv_path} 文件中更新。")
            else:
                print(f"警告: 更新 {self.dotenv_path} 文件失败！")

    def get_access_token(self) -> str:
        """
        获取一个有效的 Access Token。如果当前令牌无效或已过期，则自动刷新。

        Returns:
            一个有效的 Access Token 字符串。
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