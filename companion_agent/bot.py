"""微信适配层 — 收消息 → engine → 发回复。

纯胶水代码，不包含业务逻辑。wiring（创建 Engine、Session 等）在 __main__.py 完成。

用法::

    bot = Bot(engine=engine, base_url=..., token=...)
    # 作为 MonitorLoop 的 on_message handler:
    loop = MonitorLoop(..., on_message=bot.as_handler(loop))
"""

from __future__ import annotations

import asyncio
import logging

from weixin_bot.messaging.inbound import InboundMessage, parse_message
from weixin_bot.messaging.send import send_text
from weixin_bot.messaging.typing import TypingIndicator

from companion_agent.agent.engine import Engine

logger = logging.getLogger(__name__)

# 消息拆分后每条发送前的等待秒数 × 消息长度
_DELAY_PER_CHAR = 0.2


class Bot:
    """微信消息的接收 → 处理 → 回复管道。

    不负责 wiring、登录、MonitorLoop 创建。
    """

    def __init__(
        self,
        engine: Engine,
        base_url: str,
        token: str,
        *,
        delay_per_char: float = _DELAY_PER_CHAR,
    ) -> None:
        self._engine = engine
        self._base_url = base_url
        self._token = token
        self._delay_per_char = delay_per_char

    # ------------------------------------------------------------------
    # 消息处理
    # ------------------------------------------------------------------

    async def handle(self, msg: InboundMessage, context_token: str) -> None:
        """处理一条 WeChat 消息。

        msg: parse_message 的产出
        context_token: 从 loop.ctx_tokens.get() 获取的 token
        """
        text = msg.text.strip()
        if not text:
            return

        # typing indicator 覆盖 engine 思考 + 发送全过程
        async with TypingIndicator(
            base_url=self._base_url,
            token=self._token,
            ilink_user_id=msg.from_user,
            context_token=context_token,
        ):
            # 调 engine
            replies = await self._engine.handle(text)

            # 逐条发送，首条不等（LLM 思考已花时间），后续按长度等
            for i, reply in enumerate(replies):
                if i > 0:
                    await asyncio.sleep(len(reply) * self._delay_per_char)
                await send_text(
                    to=msg.from_user,
                    text=reply,
                    base_url=self._base_url,
                    token=self._token,
                    context_token=context_token,
                )

    # ------------------------------------------------------------------
    # 回调生成
    # ------------------------------------------------------------------

    def as_handler(self, loop):
        """生成 MonitorLoop 的 on_message 回调。

        闭包捕获 loop 以获取 ctx_tokens。
        """
        bot = self

        async def on_message(msg: dict) -> None:
            m = parse_message(msg)
            # 优先用 loop 维护的 ctx_token（自动刷新），fallback 到消息自带
            ctx = (
                await loop.ctx_tokens.get(
                    user_id=m.from_user,
                    base_url=self._base_url,
                    auth_token=self._token,
                )
                or m.context_token
            )
            try:
                await bot.handle(m, ctx)
            except Exception:
                logger.exception("handle message failed")

        return on_message
