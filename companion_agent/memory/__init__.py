"""记忆系统 — 用户认知档案 + 对话记录 + Dream 记忆整合。

用法::

    from companion_agent.memory import UserProfile, Session, Dream

    user_profile = UserProfile(Path("workspace/USER.md"))
    session = Session(Path("workspace/sessions/user_xxx.jsonl"))
    dream = Dream(session=session, user_profile=user_profile, llm_call=my_llm)
"""

from .dream import Dream
from .session import Session
from .user_profile import UserProfile

__all__ = ["UserProfile", "Session", "Dream"]
