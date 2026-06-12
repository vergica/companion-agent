"""引擎 — 编排一次对话的完整流程。

将 prompt、llm、session、dream 串联起来，是 agent 模块对外的唯一入口。

流程::

    user_text → build prompt → call llm → save session → check dream
    → split reply → return list[str]

LLM 可以通过输出 ``<silent>`` 来决定本轮不回复（静默轮）。
正常回复按换行拆分为多条消息，由 bot 层逐条发送。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from companion_agent.agent.prompt import build_system_prompt, build_messages
from companion_agent.memory.dream import Dream
from companion_agent.memory.session import Session
from companion_agent.memory.user_profile import UserProfile

if TYPE_CHECKING:
    from companion_agent.llm import LlmClient

# ---------------------------------------------------------------------------
# 静默标记
# ---------------------------------------------------------------------------

_SILENT = "<silent>"


def _is_silent(text: str) -> bool:
    """模型决定本轮不回复。大小写不敏感。"""
    return text.strip().lower() == _SILENT


# ---------------------------------------------------------------------------
# 回复拆分
# ---------------------------------------------------------------------------


def _split_reply(text: str) -> list[str]:
    """按换行拆分回复，过滤空行。"""
    return [line.strip() for line in text.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """一次对话的完整编排。

    用法::

        engine = Engine(
            soul=Path("workspace/SOUL.md").read_text(),
            session=Session(Path("workspace/sessions/user.jsonl")),
            user_profile=UserProfile(Path("workspace/USER.md")),
            llm=LlmClient(...),
        )
        replies = await engine.handle("今天好累啊")
        for msg in replies:
            await bot.send(msg)
    """

    def __init__(
        self,
        soul: str,
        session: Session,
        user_profile: UserProfile,
        llm: LlmClient,
        *,
        dream_trigger_rounds: int = 30,
        max_history_rounds: int = 100,
    ) -> None:
        self._soul = soul
        self._session = session
        self._user_profile = user_profile
        self._llm = llm
        self._dream_trigger_rounds = dream_trigger_rounds
        self._max_history_rounds = max_history_rounds

    async def handle(self, text: str, *, save_user_message: bool = True) -> list[str]:
        """处理一条用户消息，返回 agent 的回复列表。

        返回空列表 ``[]`` 表示本轮不回复（静默轮）。
        save_user_message=False 时不存储用户消息（用于 proactive 系统模拟）。
        副作用：写入 session（含思考过程）、可能触发 Dream 更新 USER.md。
        """
        # 1. 组装 prompt
        system_prompt = build_system_prompt(
            soul=self._soul,
            user_profile=self._user_profile.read(),
        )
        messages = build_messages(
            history=self._session.get_recent_rounds(self._max_history_rounds),
            current_message=text,
        )

        # 2. 调 LLM
        response = await self._llm.chat(
            system_prompt=system_prompt,
            messages=messages,
        )

        # 3. 用户消息写入 session（proactive 模拟消息不存储）
        if save_user_message:
            self._session.add_message("user", text)

        # 4. 存原始回复到 session（含思考过程，未拆分）
        #    先存再检测静默——即使不说话，思考过程也有价值
        self._session.add_message(
            "assistant",
            response.text,
            thinking=response.thinking or None,
        )

        # 5. 静默检测——模型决定本轮不说话
        if _is_silent(response.text):
            return []

        # 6. 检查是否需要 Dream 整合
        if self._session.unprocessed_rounds >= self._dream_trigger_rounds:
            dream = Dream(
                session=self._session,
                user_profile=self._user_profile,
                llm_call=_make_dream_llm(self._llm),
            )
            await dream.run()

        # 7. 拆分并返回
        return _split_reply(response.text)


def _make_dream_llm(llm: LlmClient):
    """把 LlmClient.chat() 适配为 Dream 需要的 llm_call 签名。"""
    async def _call(system_prompt: str, user_prompt: str) -> str:
        response = await llm.chat(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.text
    return _call
