"""prompt.py 测试 — 纯函数，无外部依赖。"""

from companion_agent.agent.prompt import build_system_prompt, build_messages


SOUL = """# 角色

你是小伴，一个温暖的个人陪伴AI。
你善于倾听，也会适时给出建议。"""

USER = """# 用户档案

## 基本信息
- 称呼：小明
- 年龄：25
[last_updated: 2026-06-10]"""


class TestBuildSystemPrompt:

    def test_contains_soul(self):
        prompt = build_system_prompt(soul=SOUL, user_profile=USER)
        assert "小伴" in prompt
        assert "倾听" in prompt

    def test_contains_user_profile(self):
        prompt = build_system_prompt(soul=SOUL, user_profile=USER)
        assert "小明" in prompt
        assert "25" in prompt

    def test_contains_sections(self):
        """四个 section 都存在：soul / 回复规则 / 用户 / 当前时间。"""
        prompt = build_system_prompt(soul=SOUL, user_profile=USER)
        assert "角色" in prompt        # soul section
        assert "回复规则" in prompt     # reply rules
        assert "<silent>" in prompt    # silence marker
        assert "禁止" in prompt         # format bans (markdown/latex/...)
        assert "允许" in prompt         # positive: emoji, 口癖, etc.
        assert "关于用户" in prompt     # user section
        assert "当前时间" in prompt     # time section

    def test_empty_user_profile(self):
        """USER.md 为空时不影响拼接。"""
        prompt = build_system_prompt(soul=SOUL, user_profile="")
        assert "关于用户" not in prompt  # 空的就不加这个 section

    def test_contains_today_date(self):
        prompt = build_system_prompt(soul=SOUL, user_profile=USER)
        # 应该包含类似 "2026年06月13日" 的内容
        assert "2026" in prompt


class TestBuildMessages:

    def test_appends_current_message(self):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "嗨！"},
        ]
        messages = build_messages(history, "今天好累")
        assert len(messages) == 3
        assert messages[-1] == {"role": "user", "content": "今天好累"}

    def test_empty_history(self):
        """没有历史时，就只包含当前消息。"""
        messages = build_messages([], "你好")
        assert messages == [{"role": "user", "content": "你好"}]

    def test_history_unmodified(self):
        """传入的 history 不会被修改（副作用检查）。"""
        history = [{"role": "user", "content": "hi"}]
        original = len(history)
        build_messages(history, "new")
        assert len(history) == original
