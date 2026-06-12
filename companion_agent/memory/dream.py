"""Dream 记忆整合 — 读对话、更新 USER.md。

依赖注入 user_profile、session 和 llm_call（避免耦合具体 LLM SDK）。
触发条件由外部（agent 或 bot）判断，dream 自身只负责"执行一次整合"。

整合流程::

    1. 读取 session 中 dream_cursor 之后的消息
    2. 如果没有新消息，跳过（返回 False）
    3. 读取当前 USER.md
    4. 组装 Dream prompt（演化规则 + 当前档案 + 新对话）
    5. 调用 LLM → 获取新版 USER.md
    6. 覆盖写入 USER.md
    7. 更新 session.dream_cursor
"""

from __future__ import annotations

from typing import Awaitable, Callable

from .session import Session
from .user_profile import UserProfile

# LLM 调用签名：async def llm(system_prompt: str, user_prompt: str) -> str
LlmCall = Callable[[str, str], Awaitable[str]]

# ---------------------------------------------------------------------------
# Dream system prompt — 记忆演化规则
# ---------------------------------------------------------------------------

_DREAM_SYSTEM_PROMPT = """\
你是个人陪伴 AI 的记忆系统。你的任务是根据最近的对话，维护用户的结构化档案（USER.md）。

档案包含以下 section：

## 基本信息
  - 称呼、大致年龄、所在地、职业领域
  - 用户与 AI 的关系定位（朋友/知己/陪伴者等）

## 性格与特质
  - 性格描述、思维风格、沟通特点
  - 情绪模式（容易被什么影响、如何调节）

## 生活图景
  - 日常节奏、工作/学习状况
  - 生活环境、重要的人事物

## 兴趣与偏好
  - 兴趣爱好、喜欢的领域
  - 聊天风格偏好（需要鼓励/需要倾听/需要调侃）
  - 忌讳话题

## 当前状态
  - 最近在做什么、关注什么
  - 当前心情/困扰/期待
  - 近期重要事件

## 我们之间
  - 重要的对话历史节点
  - 内部笑话、特殊称呼、共同记忆
  - 关系进展

---

演化规则 — 更新档案时必须遵守：

1. **新增**：对话中出现的关于用户的新信息，添加到对应 section。
   只记录用户明确表达或强烈暗示的内容，不要编造。

2. **更新**：如果新信息与已有信息**矛盾**，用新信息替换旧的。
   保留旧版本作为 HTML 注释：<!-- 旧: ... -->

3. **强化**：如果新信息**印证**了已有信息，提升可信度标记：
   low → medium → high

4. **降级**：如果用户行为与已有认知**明显不符**，降低可信度：
   high → medium → low

5. **清理**：标记为 `confidence: low` 且超过 30 天未更新的信息可以删除。

6. **保留**：「旧」不等于「不重要」。不要因为是旧信息就删除，
   有些旧信息是理解用户的基础。

---

元数据格式：

- 每个 section 末尾必须带 `[last_updated: YYYY-MM-DD]`
- 信息可信时加 `[confidence: high]`
- 不确定时加 `[confidence: low]`
- 中等把握加 `[confidence: medium]` 或不写

---

输出完整的 USER.md。不要跳过任何 section。
- 有新信息的 section：更新内容，并将 `last_updated` 改为今天的日期
- 没有新信息的 section：保持原样，包括原来的 `last_updated` 日期也不要改

只输出 markdown，不要任何解释或前言。"""


def _build_user_prompt(current_user_md: str, messages: list[dict]) -> str:
    """组装 Dream 的 user prompt：当前档案 + 新对话。"""
    conv_text = _format_messages_for_dream(messages)
    return f"""\
## 当前 USER.md

{current_user_md}

## 最近的对话（自上次更新以来）

{conv_text}

请根据以上对话更新 USER.md。"""


def _format_messages_for_dream(messages: list[dict]) -> str:
    """把消息列表格式化为 Dream 可读的对话文本。

    AI 的思考过程也包含在内，帮助 Dream 理解模型当时的判断依据。
    """
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        thinking = m.get("thinking", "")
        ts = m.get("timestamp", "")
        if role == "user":
            lines.append(f"[{ts}] 用户: {content}")
        else:
            if thinking:
                lines.append(f"[{ts}] AI 思考: {thinking}")
            lines.append(f"[{ts}] AI: {content}")
    return "\n".join(lines)


_USER_MD_SECTIONS = [
    "基本信息", "性格与特质", "生活图景",
    "兴趣与偏好", "当前状态", "我们之间",
]


def _is_valid_user_md(content: str) -> bool:
    """检查 LLM 输出是否像一个合法的 USER.md。"""
    if not content or not content.strip():
        return False
    # 最多允许缺失 1 个 section（即至少含 5 个）
    found = sum(1 for s in _USER_MD_SECTIONS if s in content)
    return len(_USER_MD_SECTIONS) - found <= 1


# ---------------------------------------------------------------------------
# Dream runner
# ---------------------------------------------------------------------------

class Dream:
    """执行一次记忆整合。

    用法::

        dream = Dream(
            session=session,
            user_profile=user_profile,
            llm_call=my_llm_function,
        )
        updated = await dream.run()
        if updated:
            print("USER.md 已更新")
    """

    def __init__(
        self,
        session: Session,
        user_profile: UserProfile,
        llm_call: LlmCall,
    ) -> None:
        self.session = session
        self.user_profile = user_profile
        self.llm_call = llm_call

    async def run(self) -> bool:
        """执行一次 Dream 整合。有新消息时更新 USER.md，返回 True。

        无新消息时直接返回 False，不调用 LLM。
        """
        messages = self.session.get_unprocessed_for_dream()
        if not messages:
            return False

        current_md = self.user_profile.read()
        user_prompt = _build_user_prompt(current_md, messages)

        new_md = await self.llm_call(_DREAM_SYSTEM_PROMPT, user_prompt)
        new_md = new_md.strip()

        # 校验输出：必须包含足够的 section 才算合法
        if not _is_valid_user_md(new_md):
            # 输出不合法，不标记已处理，下次 Dream 仍会重试这批消息
            return False

        # 内容没有实质变化，仍然标记已处理（这批对话确实没有新信息）
        if new_md == current_md.strip():
            self.session.mark_dream_done()
            return False

        self.user_profile.write(new_md)
        self.session.mark_dream_done()
        return True
