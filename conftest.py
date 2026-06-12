"""pytest 配置 — 加载 .env + asyncio 模式。"""
from pathlib import Path

import pytest


def pytest_configure(config):
    config.option.asyncio_mode = "auto"
    _load_dotenv()


def _load_dotenv():
    """加载项目根目录的 .env 文件（如果 python-dotenv 可用）。"""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        pass
