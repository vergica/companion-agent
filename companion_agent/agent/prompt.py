"""prompt 组装 — 纯函数，不依赖任何外部模块。

将 SOUL.md、USER.md、对话历史和当前消息拼接为 LLM 可用的
system_prompt 和 messages 列表。
"""

from __future__ import annotations

from datetime import datetime

# ---------------------------------------------------------------------------
# prompt 片段常量
# ---------------------------------------------------------------------------

# 前置声明——放在 SOUL 之前，确保 LLM 第一时间知道这是聊天场景
_CONTEXT = """\
# 场景

你正在微信上和用户实时聊天。你的每一条回复就是打字发出去的消息。
你只能输出用户会在他手机屏幕上看到的文字。不要描写动作/表情/语气，不要用任何第三人称叙事（如"某某叹了口气"、"某某冷冷地说"）。把你的情绪和态度直接融入对话里。"""

_REPLY_RULES = """\
# 回复规则

- 如果判断当前不需要回复，只输出 `<silent>`，不要有任何其他内容。
- 如果要说话，正常输出你的回复。
- 禁止 markdown、Latex、HTML、代码块、链接、列表等等。
- 允许使用标点，口癖，换行符，emoji等**日常聊天**会出现的内容"""


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def build_system_prompt(
    soul: str,
    user_profile: str,
) -> str:
    """组装 system prompt。

    结构::

        场景约束（聊天场景 + 禁止旁白）
        ---
        SOUL.md（agent 人格）
        ---
        回复规则
        ---
        USER.md（对用户的认知）
        ---
        当前时间
    """
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    parts = [
        _CONTEXT,
        soul.strip(),
        _REPLY_RULES,
        _build_user_section(user_profile),
        f"# 当前时间\n\n{now}",
    ]
    return "\n\n---\n\n".join(p for p in parts if p)


def build_messages(
    history: list[dict],
    current_message: str,
) -> list[dict]:
    """组装 messages 列表，用于 LLM 请求。

    参数:
        history: session.get_recent_rounds() 的返回值
        current_message: 用户刚发的消息

    返回::

        [
            {"role": "user", "content": "历史消息1"},
            {"role": "assistant", "content": "历史回复1"},
            ...
            {"role": "user", "content": "当前消息"},
        ]
    """
    messages = list(history)
    messages.append({"role": "user", "content": current_message})
    return messages


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------

def _build_user_section(user_profile: str) -> str:
    """把 USER.md 包装为 system prompt 的一部分。"""
    if not user_profile.strip():
        return ""
    return f"# 关于用户\n\n以下是你对用户的了解，由记忆系统维护：\n\n{user_profile.strip()}"


