"""companion-agent 入口 — 组装并启动所有模块。

用法::

    python -m companion_agent

环境变量:
    ANTHROPIC_API_KEY     DeepSeek/Anthropic API key（必需）
    ANTHROPIC_BASE_URL    默认 https://api.deepseek.com/anthropic
    ANTHROPIC_MODEL       默认 deepseek-v4-pro
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from companion_agent.agent.engine import Engine
from companion_agent.bot import Bot
from companion_agent.config import load_config
from companion_agent.llm import LlmClient
from companion_agent.memory import Session, UserProfile
from companion_agent.proactive import Proactive
from weixin_bot.auth.login import login
from weixin_bot.messaging.send import send_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

_WORKSPACE = Path("workspace")
_SOUL_PATH = _WORKSPACE / "SOUL.md"
_USER_MD_PATH = _WORKSPACE / "USER.md"
_SESSION_PATH = _WORKSPACE / "sessions" / "default.jsonl"


async def main() -> None:
    # ---- 配置 ----
    cfg = load_config()
    if not cfg.llm.api_key:
        print("错误: 未设置 ANTHROPIC_API_KEY 环境变量")
        sys.exit(1)

    # ---- 核心模块 ----
    llm = LlmClient(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        max_tokens=cfg.llm.max_tokens,
    )

    session = Session(_SESSION_PATH)
    user_profile = UserProfile(_USER_MD_PATH)
    user_profile.ensure_exists()

    if _SOUL_PATH.exists():
        soul = _SOUL_PATH.read_text(encoding="utf-8")
    else:
        print(f"警告: {_SOUL_PATH} 不存在，使用默认人格")
        soul = "# 角色\n\n你是用户的个人陪伴 AI，温暖、真诚、善于倾听。"

    engine = Engine(
        soul=soul,
        session=session,
        user_profile=user_profile,
        llm=llm,
        dream_trigger_rounds=cfg.companion.dream_trigger_rounds,
        max_history_rounds=cfg.companion.max_history_rounds,
    )

    # ---- 微信登录 ----
    print("正在登录微信...")
    login_result = await login()

    base_url = login_result["base_url"]
    token = login_result["bot_token"]
    account_id = login_result["account_id"]
    print(f"登录成功 account_id={account_id}")

    # ---- 微信适配层 ----
    bot = Bot(
        engine=engine,
        base_url=base_url,
        token=token,
        delay_per_char=cfg.bot.delay_per_char,
    )

    # ---- 用户名册（缓存首个消息的发送者，供 proactive 使用）----
    user_state: dict = {"id": ""}

    async def _proactive_send(text: str) -> None:
        to = user_state.get("id", "")
        if not to:
            return
        await send_text(
            to=to, text=text,
            base_url=base_url, token=token,
        )

    proactive = Proactive(
        engine=engine,
        session=session,
        send_callback=_proactive_send,
        tick_interval=cfg.proactive.tick_interval,
        start_hour=cfg.proactive.start_hour,
        end_hour=cfg.proactive.end_hour,
    )

    # ---- 消息处理器（闭包捕获 loop 以获取 context_token）----
    loop = None  # 先占位，闭包内引用外层变量

    async def on_message(msg: dict) -> None:
        from weixin_bot.messaging.inbound import parse_message

        m = parse_message(msg)
        # 缓存用户名册
        if m.from_user and not user_state["id"]:
            user_state["id"] = m.from_user

        # context token
        ctx = (
            await loop.ctx_tokens.get(
                user_id=m.from_user,
                base_url=base_url,
                auth_token=token,
            )
            or m.context_token
        )

        try:
            await bot.handle(m, ctx)
        except Exception:
            logger.exception("消息处理异常")

    # ---- 启动 ----
    await proactive.start()
    print("bot 已就绪，等待消息...")

    from weixin_bot.monitor.loop import MonitorLoop

    loop = MonitorLoop(
        base_url=base_url,
        token=token,
        account_id=account_id,
        on_message=on_message,
    )

    try:
        await loop.run()
    finally:
        await proactive.stop()
        print("已退出")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
