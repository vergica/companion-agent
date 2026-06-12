"""engine.py 测试 — mock LLM，真实 session/user_profile。"""

from pathlib import Path

import pytest

from companion_agent.agent.engine import Engine, _is_silent, _split_reply
from companion_agent.llm import LlmResponse


SOUL = "# 角色\n你是小伴，温暖的陪伴AI。\n"


class MockLlm:
    """模拟 LLM，聊天返回预设回复（LlmResponse），Dream 调用返回合法 USER.md。"""

    def __init__(self, reply: str = "默认回复", thinking: str = ""):
        self.reply = reply
        self.thinking = thinking
        self.call_count = 0
        self.last_system = ""
        self.last_messages = []

    async def chat(self, system_prompt: str, messages: list[dict]) -> LlmResponse:
        self.call_count += 1
        self.last_system = system_prompt
        self.last_messages = list(messages)
        # Dream 的 system prompt 含"演化规则"，正常 chat 不含
        if "演化规则" in system_prompt:
            return LlmResponse(text=_valid_user_md())
        return LlmResponse(text=self.reply, thinking=self.thinking)


def _valid_user_md() -> str:
    return "\n".join(
        f"## {s}\n内容\n[last_updated: 2026-06-13]"
        for s in ["基本信息", "性格与特质", "生活图景", "兴趣与偏好", "当前状态", "我们之间"]
    )


@pytest.fixture
def workspace(tmpdir):
    return Path(str(tmpdir))


@pytest.fixture
def session(workspace):
    from companion_agent.memory.session import Session
    return Session(workspace / "sessions" / "test.jsonl")


@pytest.fixture
def user_profile(workspace):
    from companion_agent.memory.user_profile import UserProfile
    return UserProfile(workspace / "USER.md")


@pytest.fixture
def llm():
    return MockLlm(reply="嗨！我在呢。")


@pytest.fixture
def engine(session, user_profile, llm):
    return Engine(
        soul=SOUL,
        session=session,
        user_profile=user_profile,
        llm=llm,
        dream_trigger_rounds=3,     # 设小一点方便测 Dream 触发
        max_history_rounds=10,
    )


# ---------------------------------------------------------------------------
# _is_silent 纯函数
# ---------------------------------------------------------------------------


class TestIsSilent:
    def test_lowercase(self):
        assert _is_silent("<silent>") is True

    def test_uppercase(self):
        assert _is_silent("<SILENT>") is True

    def test_mixed_case(self):
        assert _is_silent("<Silent>") is True

    def test_with_whitespace(self):
        assert _is_silent("  <silent>  ") is True

    def test_not_silent(self):
        assert _is_silent("你好") is False

    def test_empty_string(self):
        assert _is_silent("") is False


# ---------------------------------------------------------------------------
# _split_reply 纯函数
# ---------------------------------------------------------------------------


class TestSplitReply:
    def test_single_line(self):
        assert _split_reply("你好") == ["你好"]

    def test_multi_line(self):
        text = "今天天气真好。\n对了，你上次说的那本书我看完了。\n真的很好看！"
        assert _split_reply(text) == [
            "今天天气真好。",
            "对了，你上次说的那本书我看完了。",
            "真的很好看！",
        ]

    def test_filters_empty_lines(self):
        text = "第一段\n\n\n第二段\n"
        assert _split_reply(text) == ["第一段", "第二段"]

    def test_all_empty(self):
        assert _split_reply("\n\n\n") == []

    def test_empty_string(self):
        assert _split_reply("") == []

    def test_strips_whitespace(self):
        text = "  你好  \n  再见  "
        assert _split_reply(text) == ["你好", "再见"]


# ---------------------------------------------------------------------------
# Engine 基础
# ---------------------------------------------------------------------------


class TestEngineBasic:

    @pytest.mark.asyncio
    async def test_returns_list(self, engine, llm):
        """handle() 返回 list[str]。"""
        reply = await engine.handle("你好")
        assert reply == ["嗨！我在呢。"]
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_passes_system_prompt(self, engine, llm):
        """system prompt 包含 SOUL、回复规则、关于用户。"""
        await engine.handle("测试")
        assert "小伴" in llm.last_system
        assert "回复规则" in llm.last_system
        assert "关于用户" in llm.last_system

    @pytest.mark.asyncio
    async def test_passes_messages(self, engine, llm):
        """messages 以当前用户消息结尾。"""
        await engine.handle("你好")
        assert llm.last_messages[-1] == {"role": "user", "content": "你好"}

    @pytest.mark.asyncio
    async def test_saves_to_session(self, engine, session):
        """对话写入 session（user + assistant）。"""
        assert session.get_message_count() == 0
        await engine.handle("消息")
        assert session.get_message_count() == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_save_user_message_false(self, engine, session, llm):
        """save_user_message=False → 不存 user，但存 assistant。"""
        llm.reply = "主动问候"
        before = session.get_message_count()
        await engine.handle("[SYSTEM] 检查", save_user_message=False)
        # 只多了 assistant
        assert session.get_message_count() == before + 1
        messages = session._read_messages()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "主动问候"

    @pytest.mark.asyncio
    async def test_context_accumulates(self, engine, session, llm):
        """多轮对话：历史逐渐累积。"""
        llm.reply = "回复1"
        await engine.handle("消息1")

        llm.reply = "回复2"
        await engine.handle("消息2")

        # 第二轮时 messages 应包含第一轮的历史
        assert len(llm.last_messages) >= 3  # user1, asst1, user2

    @pytest.mark.asyncio
    async def test_empty_message(self, engine):
        """空消息不崩。"""
        reply = await engine.handle("")
        assert isinstance(reply, list)


# ---------------------------------------------------------------------------
# 静默机制
# ---------------------------------------------------------------------------


class TestEngineSilent:

    @pytest.mark.asyncio
    async def test_silent_returns_empty(self, engine, llm):
        """LLM 输出 <silent> → 返回空列表。"""
        llm.reply = "<silent>"
        reply = await engine.handle("用户消息")
        assert reply == []

    @pytest.mark.asyncio
    async def test_silent_case_insensitive(self, engine, llm):
        """<SILENT> / <Silent> 都识别。"""
        for variant in ["<SILENT>", "<Silent>", "  <silent>  "]:
            llm.reply = variant
            reply = await engine.handle("hi")
            assert reply == []

    @pytest.mark.asyncio
    async def test_silent_still_saves_both_messages(self, engine, session, llm):
        """静默轮也保存 user 和 assistant（含思考过程）。"""
        llm.reply = "<silent>"
        llm.thinking = "用户看起来心情不太好，我决定先不打扰。"
        before = session.get_message_count()
        await engine.handle("用户说了什么")
        # user + assistant 都保存了
        assert session.get_message_count() == before + 2

    @pytest.mark.asyncio
    async def test_silent_saves_thinking(self, engine, session, llm):
        """静默轮的思考过程也被保存。"""
        llm.reply = "<silent>"
        llm.thinking = "用户在倾诉，我应该沉默倾听。"
        await engine.handle("测试")
        messages = session._read_messages()
        # 找最后一条 assistant 消息
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "<silent>"
        assert assistant_msgs[0].get("thinking") == "用户在倾诉，我应该沉默倾听。"


# ---------------------------------------------------------------------------
# 多段回复拆分
# ---------------------------------------------------------------------------


class TestEngineSplit:

    @pytest.mark.asyncio
    async def test_multi_line_reply(self, engine, llm):
        """多行回复按换行拆分。"""
        llm.reply = "第一句\n第二句\n第三句"
        reply = await engine.handle("你好")
        assert reply == ["第一句", "第二句", "第三句"]

    @pytest.mark.asyncio
    async def test_session_stores_original_text(self, engine, session, llm):
        """session 存的是原始 LLM 输出，不是拆分后的数组。"""
        llm.reply = "A\nB\nC"
        await engine.handle("测试")
        rounds = session.get_recent_rounds(100)
        assistant_msg = [m for m in rounds if m["role"] == "assistant"][0]
        assert assistant_msg["content"] == "A\nB\nC"


# ---------------------------------------------------------------------------
# Dream 触发
# ---------------------------------------------------------------------------


class TestEngineDream:

    @pytest.mark.asyncio
    async def test_dream_triggers_after_threshold(self, engine, session, user_profile, llm):
        """累计达到 dream_trigger_rounds 后触发 Dream。"""
        user_profile.ensure_exists()
        before = user_profile.read()

        # 发 3 条消息（刚好达到阈值）
        for i in range(3):
            llm.reply = f"回复{i}"
            await engine.handle(f"消息{i}")

        after = user_profile.read()
        # Dream 被触发过 → USER.md 应该变了
        assert after != before

    @pytest.mark.asyncio
    async def test_dream_not_triggered_early(self, engine, session, user_profile, llm):
        """未达阈值时不触发 Dream。"""
        user_profile.ensure_exists()
        before = user_profile.read()

        # 只发 2 条（阈值是 3）
        for i in range(2):
            llm.reply = f"回复{i}"
            await engine.handle(f"消息{i}")

        after = user_profile.read()
        # 没触发 Dream → USER.md 不变
        assert after == before
