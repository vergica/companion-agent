"""配置加载 — env 变量 > config.yaml > 默认值。

用法::

    from companion_agent.config import load_config
    cfg = load_config()
    print(cfg.llm.base_url)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 配置结构
# ---------------------------------------------------------------------------


@dataclass
class LlmConfig:
    base_url: str = "https://api.deepseek.com/anthropic"
    api_key: str = ""
    model: str = "deepseek-v4-pro"
    max_tokens: int = 4096


@dataclass
class CompanionConfig:
    dream_trigger_rounds: int = 30
    max_history_rounds: int = 100


@dataclass
class ProactiveConfig:
    tick_interval: int = 1800     # 检查间隔（秒）
    start_hour: int = 7
    end_hour: int = 23


@dataclass
class BotConfig:
    delay_per_char: float = 0.2   # 每条消息发送前等待秒数 × 长度


@dataclass
class Config:
    llm: LlmConfig = field(default_factory=LlmConfig)
    companion: CompanionConfig = field(default_factory=CompanionConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    bot: BotConfig = field(default_factory=BotConfig)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env(value: str) -> str:
    """替换字符串中的 ${VAR} 为环境变量值。"""
    def _repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return _ENV_VAR_RE.sub(_repl, value)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def load_config() -> Config:
    """加载配置，优先级：环境变量 > config.yaml > 默认值。"""
    cfg = Config()

    # --- config.yaml ---
    _load_yaml(cfg)

    # --- 环境变量覆盖 ---
    _apply_env(cfg)

    return cfg


def _load_yaml(cfg: Config) -> None:
    """从 config.yaml 加载（如果存在）。"""
    for candidate in [
        Path("config.yaml"),
        Path("workspace/config.yaml"),
    ]:
        if candidate.exists():
            _merge_yaml(cfg, candidate)
            return


def _merge_yaml(cfg: Config, path: Path) -> None:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception:
        return

    if "llm" in data:
        for k in ("base_url", "api_key", "model", "max_tokens"):
            if k in data["llm"]:
                val = data["llm"][k]
                if isinstance(val, str):
                    val = _resolve_env(val)
                setattr(cfg.llm, k, val)

    if "companion" in data:
        for k in ("dream_trigger_rounds", "max_history_rounds"):
            if k in data["companion"]:
                setattr(cfg.companion, k, data["companion"][k])

    if "proactive" in data:
        for k in ("tick_interval", "start_hour", "end_hour"):
            if k in data["proactive"]:
                setattr(cfg.proactive, k, data["proactive"][k])

    if "bot" in data:
        if "delay_per_char" in data["bot"]:
            val = data["bot"]["delay_per_char"]
            if isinstance(val, str):
                val = _resolve_env(val)
            cfg.bot.delay_per_char = float(val)


def _apply_env(cfg: Config) -> None:
    """环境变量覆盖 config 字段。"""
    # LLM
    if v := _env("ANTHROPIC_API_KEY"):
        cfg.llm.api_key = v
    if v := _env("ANTHROPIC_BASE_URL"):
        cfg.llm.base_url = v
    if v := _env("ANTHROPIC_MODEL"):
        cfg.llm.model = v
    cfg.llm.max_tokens = _env_int("ANTHROPIC_MAX_TOKENS", cfg.llm.max_tokens)

    # Companion
    cfg.companion.dream_trigger_rounds = _env_int(
        "COMPANION_DREAM_TRIGGER_ROUNDS", cfg.companion.dream_trigger_rounds,
    )
    cfg.companion.max_history_rounds = _env_int(
        "COMPANION_MAX_HISTORY_ROUNDS", cfg.companion.max_history_rounds,
    )

    # Proactive
    cfg.proactive.tick_interval = _env_int(
        "PROACTIVE_TICK_INTERVAL", cfg.proactive.tick_interval,
    )
    cfg.proactive.start_hour = _env_int("PROACTIVE_START_HOUR", cfg.proactive.start_hour)
    cfg.proactive.end_hour = _env_int("PROACTIVE_END_HOUR", cfg.proactive.end_hour)

    # Bot
    cfg.bot.delay_per_char = _env_float("BOT_DELAY_PER_CHAR", cfg.bot.delay_per_char)
