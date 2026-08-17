"""应用配置：支持环境变量和 .env 文件，默认值适合本地开发。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """集中管理应用配置，字段名对应环境变量。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Voice Compare API"
    version: str = "0.2.0"
    upload_dir: Path = BASE_DIR / "uploads"
    static_dir: Path = BASE_DIR / "static"
    db_path: Path = BASE_DIR / "data" / "files.db"
    max_upload_size_mb: int = 500
    allowed_origins: list[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
    ffprobe_path: str | None = None
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """返回缓存的配置实例，避免每次请求都重新读取。"""
    return Settings()
