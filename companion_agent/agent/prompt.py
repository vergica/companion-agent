"""prompt 组装 — 纯函数，不依赖任何外部模块。

将 SOUL.md、USER.md、对话历史和当前消息拼接为 LLM 可用的
system_prompt 和 messages 列表。
"""

from __future__ import annotations

from datetime import datetime

# ---------------------------------------------------------------------------
# prompt 片段常量
# ---------------------------------------------------------------------------

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


