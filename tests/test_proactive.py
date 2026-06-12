"""proactive.py 测试 — mock engine + session，纯函数 + 集成。"""

from pathlib import Path

import pytest

from companion_agent.proactive import _build_proactive_message, _first_time_message
from companion_agent.proactive import Proactive


# ---------------------------------------------------------------------------
# 纯函数：系统消息模板
# ---------------------------------------------------------------------------


class TestBuildProactiveMessage:
    def test_with_idle_time(self):
        from datetime import datetime
        now = datetime(2026, 6, 13, 14, 30)
        msg = _build_proactive_message(2, 35, now)
        assert "2 小时 35 分钟" in msg
        assert "14:30" in msg
        assert "<silent>" in msg

    def test_zero_minutes(self):
        from datetime import datetime
        now = datetime(2026, 6, 13, 8, 0)
        msg = _build_proactive_message(0, 0, now)
        assert msg  # 不崩就行


class TestFirstTimeMessage:
    def test_startup_message(self):
        from datetime import datetime
        now = datetime(2026, 6, 13, 9, 15)
        msg = _first_time_message(now)
        assert "09:15" in msg
        assert "首次" in msg or "第一次" in msg
        assert "必须" in msg         # 强制说话
        assert "主动打招呼" in msg    # 正面指令


# ---------------------------------------------------------------------------
# Proactive tick 逻辑（mock engine + send_callback）
# ---------------------------------------------------------------------------


class MockEngine:
    """记录调用，返回预设回复。"""

    def __init__(self, replies: list[str] | None = None):
        self.replies = replies or []
        self.calls: list[dict] = []

    async def handle(self, text: str, *, save_user_message: bool = True) -> list[str]:
        self.calls.append({"text": text, "save_user_message": save_user_message})
        return list(self.replies)


class MockSession:
    """记录消息时间戳。"""

    def __init__(self, last_message_time: str | None = None, message_count: int = 0):
        self._last_message_time = last_message_time
        self.message_count = message_count

    @property
    def last_message_time(self) -> str | None:
        return self._last_message_time


@pytest.fixture
def workspace(tmpdir):
    return Path(str(tmpdir))


@pytest.mark.asyncio
class TestProactiveTick:

    async def test_calls_engine_with_save_user_message_false(self):
        """tick 触发后调 engine.handle 时 save_user_message=False。"""
        session = MockSession()  # 无历史 → 立即触发
        engine = MockEngine(replies=["你好呀"])
        sent: list[str] = []

        async def _send(text: str) -> None:
            sent.append(text)

        # 不启动后台循环，直接调 _tick
        proactive = Proactive(
            engine=engine, session=session,
            send_callback=_send,
            start_hour=0, end_hour=23,
        )
        await proactive._tick()

        assert len(engine.calls) == 1
        assert engine.calls[0]["save_user_message"] is False
        assert sent == ["你好呀"]

    async def test_engine_returns_silent(self):
        """engine 返回 [] → 不发消息。"""
        session = MockSession()
        engine = MockEngine(replies=[])  # 静默
        sent: list[str] = []

        async def _send(text: str) -> None:
            sent.append(text)

        proactive = Proactive(
            engine=engine, session=session,
            send_callback=_send,
            start_hour=0, end_hour=23,
        )
        await proactive._tick()

        assert sent == []

    async def test_multi_segment_reply(self):
        """多段回复逐条发送。"""
        session = MockSession()
        engine = MockEngine(replies=["第一句", "第二句", "第三句"])
        sent: list[str] = []

        async def _send(text: str) -> None:
            sent.append(text)

        proactive = Proactive(
            engine=engine, session=session,
            send_callback=_send,
            start_hour=0, end_hour=23,
        )
        await proactive._tick()

        assert sent == ["第一句", "第二句", "第三句"]
