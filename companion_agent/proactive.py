"""主动消息 — 后台定时检测，模拟系统消息触发 AI 主动联系用户。

不需要额外的 LLM 决策——直接调 engine.handle()，AI 通过 <silent>
自行判断是否该说话。

用法::

    proactive = Proactive(
        engine=engine,
        session=session,
        send_callback=bot.send_text,
        tick_interval=1800,   # 30 分钟检查一次
    )
    await proactive.start()
    # ...
    await proactive.stop()
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import Awaitable, Callable

from companion_agent.agent.engine import Engine
from companion_agent.memory.session import Session

# ---------------------------------------------------------------------------
# 系统消息模板
# ---------------------------------------------------------------------------


def _build_proactive_message(
    idle_hours: int, idle_mins: int, now: datetime,
) -> str:
    """构建模拟的系统消息，触发 AI 决策是否主动联系。"""
    time_str = now.strftime("%H:%M")
    return (
        f"[SYSTEM_PROACTIVE_CHECK]\n"
        f"距上次互动已过 {idle_hours} 小时 {idle_mins} 分钟。"
        f"现在是 {time_str}，属于可发消息的时间。\n"
        f"请判断是否主动联系用户：\n"
        f"- 要联系 → 直接输出你想说的话\n"
        f"- 不联系 → 输出 <silent>\n"
        f"记住你是主动发起对话的一方，开场要自然。"
    )


def _first_time_message(now: datetime) -> str:
    """首次启动时的系统消息——强制主动打招呼。"""
    time_str = now.strftime("%H:%M")
    return (
        f"[SYSTEM_PROACTIVE_CHECK]\n"
        f"这是你启动后第一次接触用户。现在是 {time_str}。\n"
        f"你必须主动打招呼，介绍你自己，开启你们的第一次对话。\n"
        f"可以聊聊当下的时间、天气或者一个简单的问候，让对方感到你在。"
    )


# ---------------------------------------------------------------------------
# Proactive
# ---------------------------------------------------------------------------


class Proactive:
    """后台主动消息调度器。

    每隔 tick_interval 秒检查一次，满足条件时模拟系统消息调 engine。
    """

    def __init__(
        self,
        engine: Engine,
        session: Session,
        send_callback: Callable[[str], Awaitable[None]],
        *,
        tick_interval: int = 1800,   # 30 分钟
        start_hour: int = 7,
        end_hour: int = 23,
    ) -> None:
        self._engine = engine
        self._session = session
        self._send = send_callback
        self._tick_interval = tick_interval
        self._start_hour = start_hour
        self._end_hour = end_hour
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """启动后台定时循环。重复调用无害。"""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """停止后台循环。重复调用无害。"""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """主循环：每 tick_interval 秒 tick 一次。"""
        while True:
            await asyncio.sleep(self._tick_interval)
            try:
                await self._tick()
            except Exception:
                # 异常不崩循环，下个周期重试
                pass

    async def _tick(self) -> None:
        """一次检查 + 决策 + 发送。"""
        now = datetime.now()

        # 1. 时间段检查
        if not (self._start_hour <= now.hour <= self._end_hour):
            return

        # 2. 空闲阈值：每次随机 60–300 分钟
        threshold = random.randint(60, 300)

        # 3. 计算空闲时长
        last_ts = self._session.last_message_time
        if last_ts is None:
            # 无历史 → 首次启动，直接触发
            idle_minutes = float("inf")
        else:
            last = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
            idle_minutes = (now - last).total_seconds() / 60.0

        if idle_minutes < threshold:
            return

        # 4. 组装系统消息
        if last_ts is None:
            prompt = _first_time_message(now)
        else:
            idle_hours = int(idle_minutes // 60)
            idle_mins = int(idle_minutes % 60)
            prompt = _build_proactive_message(idle_hours, idle_mins, now)

        # 5. 调 engine（不存伪造的用户消息）
        replies = await self._engine.handle(prompt, save_user_message=False)

        # 6. 逐条发送
        for reply in replies:
            await self._send(reply)
