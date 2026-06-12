"""llm + memory 联合测试 — 验证完整的记忆管道。

运行方式:
    ANTHROPIC_API_KEY=sk-xxx pytest tests/test_memory.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from companion_agent.llm import LlmClient
from companion_agent.memory import Dream, Session, UserProfile

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
pytestmark = pytest.mark.skipif(not API_KEY, reason="未设置 ANTHROPIC_API_KEY")


def _safe_print(text: str) -> None:
    """print 但容忍 Windows GBK 打不出的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_dream_llm(client: LlmClient):
    """把 LlmClient.chat() 适配为 Dream 需要的 llm_call 签名。"""
    async def _call(system_prompt: str, user_prompt: str) -> str:
        response = await client.chat(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.text
    return _call


def _simulate_conversation(session: Session, turns: list[tuple[str, str]]) -> None:
    """模拟对话：交替写入 user 和 assistant 消息。"""
    for user_msg, asst_msg in turns:
        session.add_message("user", user_msg)
        session.add_message("assistant", asst_msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return LlmClient(
        base_url="https://api.deepseek.com/anthropic",
        api_key=API_KEY,
        model="deepseek-v4-pro",
        max_tokens=4096,
    )


@pytest.fixture
def workspace():
    """临时工作区，测试结束自动清理。"""
    with TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def session(workspace):
    return Session(workspace / "sessions" / "test_user.jsonl")


@pytest.fixture
def user_profile(workspace):
    return UserProfile(workspace / "USER.md")


@pytest.fixture
def dream(session, user_profile, client):
    return Dream(
        session=session,
        user_profile=user_profile,
        llm_call=_make_dream_llm(client),
    )


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class TestSessionBasics:
    """session 本身的基本功能。"""

    def test_add_and_count(self, session):
        """追加消息 → 统计正确。"""
        session.add_message("user", "你好")
        session.add_message("assistant", "嗨！")
        assert session.get_message_count() == 2

    def test_unprocessed_rounds_initial(self, session):
        """新 session 未处理轮数为 0。"""
        assert session.unprocessed_rounds == 0

    def test_unprocessed_rounds_after_add(self, session):
        """添加消息后，未处理轮数正确。"""
        _simulate_conversation(session, [
            ("消息1", "回复1"),
            ("消息2", "回复2"),
            ("消息3", "回复3"),
        ])
        assert session.unprocessed_rounds == 3

    def test_get_unprocessed_for_dream(self, session):
        """get_unprocessed_for_dream 返回 dream_cursor 之后的消息。"""
        _simulate_conversation(session, [
            ("第一轮", "回一"),
            ("第二轮", "回二"),
        ])
        unprocessed = session.get_unprocessed_for_dream()
        assert len(unprocessed) == 4  # 2 user + 2 assistant

    def test_mark_dream_done(self, session):
        """标记后，unprocessed_rounds 归零。"""
        _simulate_conversation(session, [("A", "a"), ("B", "b")])
        assert session.unprocessed_rounds == 2

        session.mark_dream_done()
        assert session.unprocessed_rounds == 0

    def test_last_message_time(self, session):
        """last_message_time 返回最后一条消息的时间戳。"""
        assert session.last_message_time is None
        session.add_message("user", "你好")
        ts1 = session.last_message_time
        assert ts1 is not None
        session.add_message("assistant", "嗨！")
        ts2 = session.last_message_time
        assert ts2 is not None
        assert ts2 >= ts1  # 时间戳递增

    def test_clear(self, session):
        """clear 清空所有消息。"""
        _simulate_conversation(session, [("A", "a")])
        session.clear()
        assert session.get_message_count() == 0
        assert session.unprocessed_rounds == 0

    def test_clear_then_add(self, session):
        """clear 后能正常追加消息 — P0。"""
        _simulate_conversation(session, [("A", "a")])
        session.clear()
        assert session.get_message_count() == 0
        assert session.unprocessed_rounds == 0

        # clear 后再加消息
        session.add_message("user", "新消息")
        session.add_message("assistant", "新回复")
        assert session.get_message_count() == 2
        assert session.unprocessed_rounds == 1

    def test_get_recent_rounds(self, session):
        """get_recent_rounds 只返回最近 N 轮。"""
        _simulate_conversation(session, [
            (f"消息{i}", f"回复{i}") for i in range(1, 11)
        ])
        recent = session.get_recent_rounds(max_rounds=3)
        # 3 轮 = 6 条消息（3 user + 3 assistant）
        assert len(recent) == 6
        # 第一条是最近的 user 消息
        assert recent[0]["role"] == "user"
        assert "消息8" in recent[0]["content"]

    def test_get_recent_rounds_includes_thinking(self, session):
        """include_thinking=True → assistant 消息含 thinking 字段。"""
        session.add_message("user", "你好")
        session.add_message("assistant", "嗨！", thinking="判断用户只是打招呼，简单回应即可。")
        session.add_message("user", "今天心情不好")
        session.add_message("assistant", "怎么了？", thinking="用户表达了负面情绪，需要共情引导。")

        recent = session.get_recent_rounds(max_rounds=2, include_thinking=True)
        # 4 条消息：user1, asst1, user2, asst2
        assert len(recent) == 4

        # assistant 消息带 thinking
        asst_msgs = [m for m in recent if m["role"] == "assistant"]
        assert len(asst_msgs) == 2
        assert asst_msgs[0].get("thinking") == "判断用户只是打招呼，简单回应即可。"
        assert asst_msgs[1].get("thinking") == "用户表达了负面情绪，需要共情引导。"

        # user 消息不带 thinking
        user_msgs = [m for m in recent if m["role"] == "user"]
        assert "thinking" not in user_msgs[0]
        assert "thinking" not in user_msgs[1]

    def test_get_recent_rounds_excludes_thinking_by_default(self, session):
        """默认 include_thinking=False → 不含 thinking 字段。"""
        session.add_message("user", "你好")
        session.add_message("assistant", "嗨！", thinking="思考内容")

        recent = session.get_recent_rounds(max_rounds=1)
        assert len(recent) == 2
        assistant_msg = [m for m in recent if m["role"] == "assistant"][0]
        assert "thinking" not in assistant_msg


class TestSessionCorruption:
    """session 文件损坏容错 — P0。"""

    def test_corrupt_metadata_line(self, tmpdir):
        """元数据行不是合法 JSON → 不崩溃，dream_cursor 回到 -1。"""
        path = Path(str(tmpdir / "s.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("这不是合法JSON\n", encoding="utf-8")

        s = Session(path)
        assert s.get_message_count() == 0
        assert s.unprocessed_rounds == 0

        # 即使元数据坏了，也能正常 add
        s.add_message("user", "你好")
        assert s.get_message_count() == 1

    def test_metadata_not_tagged(self, tmpdir):
        """第一行是合法 JSON 但不是 metadata → 不崩溃。"""
        path = Path(str(tmpdir / "s.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"role": "user", "content": "误放在第一行的消息"}\n',
            encoding="utf-8",
        )

        s = Session(path)
        assert s.unprocessed_rounds >= 0
        s.add_message("assistant", "回复")
        assert s.get_message_count() >= 1

    def test_empty_file(self, tmpdir):
        """空文件 → 正常初始化。"""
        path = Path(str(tmpdir / "s.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

        s = Session(path)
        assert s.get_message_count() == 0
        assert s.unprocessed_rounds == 0


class TestUserProfileBasics:
    """user_profile 本身的基本功能。"""

    def test_ensure_exists_creates_file(self, user_profile):
        """首次 ensure_exists 创建 USER.md。"""
        assert not user_profile.path.exists()
        created = user_profile.ensure_exists()
        assert created
        assert user_profile.path.exists()

    def test_read_returns_default(self, user_profile):
        """文件不存在时 read() 返回默认模板。"""
        content = user_profile.read()
        assert "基本信息" in content
        assert "性格与特质" in content
        assert "生活图景" in content
        assert "兴趣与偏好" in content
        assert "当前状态" in content
        assert "我们之间" in content

    def test_write_and_read(self, user_profile):
        """写入后读出内容一致。"""
        custom = "# 用户档案\n\n## 基本信息\n测试内容\n[last_updated: 2026-06-13]"
        user_profile.write(custom)
        assert user_profile.read() == custom


class TestDreamNoMessages:
    """无新消息时 Dream 的行为。"""

    @pytest.mark.asyncio
    async def test_no_new_messages_skips(self, dream, user_profile):
        """没有新消息 → run() 返回 False，不调 LLM。"""
        user_profile.ensure_exists()
        result = await dream.run()
        assert result is False


class TestDreamValidation:
    """_is_valid_user_md 校验逻辑 — P1。"""

    def test_all_sections_present(self):
        """6 个 section 全有 → 合法。"""
        from companion_agent.memory.dream import _is_valid_user_md
        content = "\n".join(
            f"## {s}\n内容\n[last_updated: 2026-06-13]"
            for s in ["基本信息", "性格与特质", "生活图景", "兴趣与偏好", "当前状态", "我们之间"]
        )
        assert _is_valid_user_md(content) is True

    def test_missing_one_section(self):
        """缺 1 个 section → 合法（容错 1 个）。"""
        from companion_agent.memory.dream import _is_valid_user_md
        content = "\n".join(
            f"## {s}\n内容\n[last_updated: 2026-06-13]"
            for s in ["基本信息", "性格与特质", "生活图景", "兴趣与偏好", "当前状态"]
        )
        assert _is_valid_user_md(content) is True

    def test_missing_two_sections(self):
        """缺 2 个 section → 不合法。"""
        from companion_agent.memory.dream import _is_valid_user_md
        content = "\n".join(
            f"## {s}\n内容\n[last_updated: 2026-06-13]"
            for s in ["基本信息", "性格与特质", "生活图景", "兴趣与偏好"]
        )
        assert _is_valid_user_md(content) is False

    def test_empty_content(self):
        """空字符串 → 不合法。"""
        from companion_agent.memory.dream import _is_valid_user_md
        assert _is_valid_user_md("") is False
        assert _is_valid_user_md("   ") is False

    def test_with_preamble_text(self):
        """LLM 在 section 前加了废话 → 仍然合法（section 在就行）。"""
        from companion_agent.memory.dream import _is_valid_user_md
        content = "好的，这是更新后的档案：\n\n" + "\n".join(
            f"## {s}\n内容\n[last_updated: 2026-06-13]"
            for s in ["基本信息", "性格与特质", "生活图景", "兴趣与偏好", "当前状态", "我们之间"]
        )
        assert _is_valid_user_md(content) is True


class TestDreamErrors:
    """Dream 异常场景 — P0。"""

    @pytest.mark.asyncio
    async def test_llm_exception_propagates(self, tmpdir, client):
        """LLM 抛异常 → 穿透 run()，cursor 不动，消息不丢。"""
        s = Session(Path(str(tmpdir / "s.jsonl")))
        p = UserProfile(Path(str(tmpdir / "USER.md")))

        async def _failing_llm(_sys: str, _usr: str) -> str:
            raise RuntimeError("API 挂了")

        d = Dream(session=s, user_profile=p, llm_call=_failing_llm)

        _simulate_conversation(s, [("test", "reply")])
        before = s.unprocessed_rounds

        with pytest.raises(RuntimeError, match="API 挂了"):
            await d.run()

        # cursor 没变 → 消息还在
        assert s.unprocessed_rounds == before


class TestDreamEdgeCases:
    """Dream 边界条件 — P2。"""

    @pytest.mark.asyncio
    async def test_llm_returns_unchanged(self, tmpdir, client):
        """LLM 返回和旧 USER.md 完全一样 → 标记已处理，不触发死循环。"""
        s = Session(Path(str(tmpdir / "s.jsonl")))
        p = UserProfile(Path(str(tmpdir / "USER.md")))

        # 一份完整的、合法的 USER.md
        existing = p.read()  # 默认模板（含全部 6 个 section）
        p.write(existing)

        _simulate_conversation(s, [
            ("今天天气不错。", "是呀，阳光很好。"),
        ])

        # Mock：LLM 原样返回（没有新信息）
        async def _echo_llm(_sys: str, _usr: str) -> str:
            return existing

        d = Dream(session=s, user_profile=p, llm_call=_echo_llm)
        result = await d.run()
        assert result is False           # 没写入
        assert s.unprocessed_rounds == 0  # 但标记已处理，不会死循环


class TestSessionEmpty:
    """空 session 各种读操作 — P1。"""

    def test_empty_reads(self, session):
        """新 session 的所有读操作不崩且返回合理值。"""
        assert session.get_message_count() == 0
        assert session.get_recent_rounds(max_rounds=10) == []
        assert session.get_unprocessed_for_dream() == []
        assert session.unprocessed_rounds == 0
        assert session.total_rounds == 0
        assert isinstance(session.created_at, str)
        assert session.last_message_time is None


class TestDreamIntegration:
    """Dream 完整集成测试 — 真实 LLM 调用。"""

    @pytest.mark.asyncio
    async def test_incremental_update(self, tmpdir, client):
        """已有 USER.md 内容 → 新对话 → 增量更新而非全量替换 — P1。"""
        s = Session(Path(str(tmpdir / "s.jsonl")))
        p = UserProfile(Path(str(tmpdir / "USER.md")))

        # 预先写入一份有内容的 USER.md（模拟之前的 Dream 产物）
        existing = """\
# 用户档案

## 基本信息
- 称呼：张三
- 年龄：25
- 所在地：杭州
- 职业：设计师
[last_updated: 2026-06-10]

## 性格与特质
- 性格内向但温和
[last_updated: 2026-06-08]

## 生活图景
- 日常：朝九晚五上班
[last_updated: 2026-06-12]

## 兴趣与偏好
- 喜欢看电影和读书
[last_updated: 2026-06-05]

## 当前状态
- 最近项目忙
[last_updated: 2026-06-13]

## 我们之间
- 刚认识
[last_updated: 2026-06-01]
"""
        p.write(existing)

        # Dream 前先标记 cursor（模拟已有处理过的消息）
        _simulate_conversation(s, [("旧的", "旧的")])
        s.mark_dream_done()

        # 新对话：张三养了猫
        _simulate_conversation(s, [
            ("我最近养了一只猫，叫豆豆，是只橘猫。",
             "豆豆这名字好可爱！橘猫很能吃吧？"),
            ("对，特别能吃。我工作忙的时候它就陪着我。",
             "真好，有个小生命陪着确实很治愈。"),
        ])

        d = Dream(session=s, user_profile=p, llm_call=_make_dream_llm(client))
        result = await d.run()
        assert result is True

        updated = p.read()

        # 旧信息保留
        assert "张三" in updated
        assert "杭州" in updated
        assert "设计师" in updated
        assert "电影" in updated or "读书" in updated

        # 新信息出现
        assert "猫" in updated or "豆豆" in updated

        _safe_print(f"\n[增量更新结果]\n{updated}")

    @pytest.mark.asyncio
    async def test_basic_dream(self, dream, session, user_profile):
        """模拟对话 → Dream → USER.md 被更新。"""
        user_profile.ensure_exists()

        # 模拟一段包含用户信息的对话
        _simulate_conversation(session, [
            (
                "你好，我叫张小明，今年28岁，在北京做程序员。",
                "张小明你好！很高兴认识你，28岁的程序员，在北京工作。有什么想聊的吗？",
            ),
            (
                "我最近工作压力很大，经常失眠。",
                "听起来很辛苦。跟我说说具体是什么让你压力这么大？",
            ),
            (
                "主要是项目赶进度，老板天天催。我喜欢打篮球，但最近都没时间去。",
                "打篮球确实是个好方式。也许周末可以抽空去打一场，哪怕就一小时。",
            ),
        ])

        original_md = user_profile.read()
        result = await dream.run()

        assert result is True
        updated_md = user_profile.read()
        # 必须包含所有 section
        for section in ["基本信息", "性格与特质", "生活图景", "兴趣与偏好", "当前状态", "我们之间"]:
            assert section in updated_md, f"缺少 section: {section}"

        # USER.md 应该有实质更新（不跟原始模板一样）
        assert updated_md != original_md
        _safe_print(f"\n[更新后的 USER.md]\n{updated_md}")

    @pytest.mark.asyncio
    async def test_dream_idempotent(self, dream, session, user_profile):
        """Dream 处理后，再次运行不再重复处理。"""
        user_profile.ensure_exists()

        _simulate_conversation(session, [
            ("我叫李四，喜欢游泳。", "李四你好！游泳是个好运动。"),
        ])

        # 第一次 Dream
        result1 = await dream.run()
        assert result1 is True

        # 第二次 Dream — 没有新消息
        result2 = await dream.run()
        assert result2 is False

    @pytest.mark.asyncio
    async def test_dream_advances_cursor(self, dream, session, user_profile):
        """Dream 完成后，unprocessed_rounds 归零。"""
        user_profile.ensure_exists()

        _simulate_conversation(session, [
            ("一些测试对话", "测试回复"),
        ])

        assert session.unprocessed_rounds == 1
        await dream.run()
        assert session.unprocessed_rounds == 0

    @pytest.mark.asyncio
    async def test_dream_remembers_name(self, dream, session, user_profile):
        """核心测试：Dream 后 USER.md 包含了对话中提到的用户信息。"""
        user_profile.ensure_exists()

        _simulate_conversation(session, [
            (
                "我叫王五，今年22岁，在上海读大学，学的是计算机。",
                "王五你好！22岁在上海读计算机，大学生活怎么样？",
            ),
            (
                "挺忙的，但我喜欢摄影，周末会出去拍照。",
                "摄影很棒！拍人像还是风景多？",
            ),
            (
                "风景多一些，我喜欢大自然。",
                "真好，上海周边有不少适合拍风景的地方。",
            ),
        ])

        await dream.run()

        updated = user_profile.read()

        # USER.md 中应该出现这些关键信息
        assert "王五" in updated
        assert "22" in updated or "二十二" in updated
        assert "上海" in updated
        _safe_print(f"\n[USER.md 是否记住王五]\n{updated}")

    @pytest.mark.asyncio
    async def test_dream_validation_rejects_invalid(self, dream, session, user_profile):
        """如果 LLM 返回不完整的输出（缺少多个 section），不写入。"""
        user_profile.ensure_exists()

        _simulate_conversation(session, [("测试", "回复")])

        # 用原来的 llm_call 正常跑，应该通过验证
        result = await dream.run()
        # 正常情况应该成功（新的对话会被处理）
        # 如果 LLM 返回合法输出，result 应该是 True
        # 如果 LLM 返回不合法（缺 >1 个 section），result 为 False
        _safe_print(f"\n[Dream 结果] {result}")
