"""bot.py 测试 — mock 微信 API + engine。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from companion_agent.bot import Bot
from weixin_bot.messaging.inbound import InboundMessage


# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------

class MockEngine:
    """记录调用，返回预设回复。"""

    def __init__(self, replies: list[str] | None = None):
        self.replies = replies or []
        self.calls: list[str] = []

    async def handle(self, text: str, *, save_user_message: bool = True) -> list[str]:
        self.calls.append(text)
        return list(self.replies)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_msg(text: str) -> InboundMessage:
    return InboundMessage(
        from_user="test_user@im.wechat",
        text=text,
        context_token="ctx123",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return MockEngine(replies=["嗨！你好呀。"])


@pytest.fixture
def bot(engine):
    return Bot(
        engine=engine,
        base_url="https://test.example.com",
        token="test-token",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBotHandle:

    @pytest.mark.asyncio
    async def test_normal_flow(self, bot, engine):
        """正常流程：收消息 → engine 处理 → 发回复。"""
        sent_texts: list[str] = []

        with (
            patch("companion_agent.bot.TypingIndicator", new=MagicMock()),
            patch("companion_agent.bot.send_text", new=AsyncMock()) as mock_send,
        ):
            mock_send.side_effect = lambda **kw: sent_texts.append(kw["text"])

            msg = _make_msg("你好")
            await bot.handle(msg, "ctx-token")

        assert engine.calls == ["你好"]
        assert sent_texts == ["嗨！你好呀。"]

    @pytest.mark.asyncio
    async def test_empty_message_skips(self, bot, engine):
        """空消息不调 engine。"""
        with patch("companion_agent.bot.send_text", new=AsyncMock()) as mock_send:
            msg = _make_msg("   ")
            await bot.handle(msg, "ctx-token")

        assert engine.calls == []
        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_segment(self, bot, engine):
        """多段回复分段发送。"""
        engine.replies = ["第一句", "第二句", "第三句"]
        sent: list[str] = []

        with (
            patch("companion_agent.bot.TypingIndicator", new=MagicMock()),
            patch("companion_agent.bot.send_text", new=AsyncMock()) as mock_send,
        ):
            mock_send.side_effect = lambda **kw: sent.append(kw["text"])
            msg = _make_msg("测试")
            await bot.handle(msg, "ctx-token")

        assert sent == ["第一句", "第二句", "第三句"]

    @pytest.mark.asyncio
    async def test_silent_no_send(self, bot, engine):
        """engine 返回 [] → 不发消息。"""
        engine.replies = []
        sent: list[str] = []

        with (
            patch("companion_agent.bot.TypingIndicator", new=MagicMock()),
            patch("companion_agent.bot.send_text", new=AsyncMock()) as mock_send,
        ):
            mock_send.side_effect = lambda **kw: sent.append(kw["text"])
            msg = _make_msg("测试")
            await bot.handle(msg, "ctx-token")

        assert sent == []

    @pytest.mark.asyncio
    async def test_typing_indicator_created(self, bot, engine):
        """验证 TypingIndicator 被创建（context manager 正常使用）。"""
        mock_typing_cls = MagicMock()

        with (
            patch("companion_agent.bot.TypingIndicator", new=mock_typing_cls),
            patch("companion_agent.bot.send_text", new=AsyncMock()),
        ):
            msg = _make_msg("测试")
            await bot.handle(msg, "ctx-token")

        # TypingIndicator 被实例化
        mock_typing_cls.assert_called_once()
        # 参数正确
        _, kwargs = mock_typing_cls.call_args
        assert kwargs["ilink_user_id"] == "test_user@im.wechat"
        assert kwargs["context_token"] == "ctx-token"
