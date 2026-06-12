"""USER.md 读写 — agent 对用户的认知档案。

纯文件 I/O，无任何外部依赖。USER.md 是 Dream 记忆整合的输出目标，
也是每次对话 system prompt 的核心组成部分。

USER.md 采用半结构化 markdown，包含 6 个固定 section。
每个 section 末尾可携带元数据行：[last_updated: ...], [confidence: ...]
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# USER.md 的默认模板 — 第一次运行时生成
# ---------------------------------------------------------------------------

_DEFAULT_USER_MD = """\
# 用户档案

## 基本信息
<!-- 称呼、大致年龄、所在地、职业领域 -->
<!-- 与 agent 的关系定位 -->
[last_updated: unknown]

## 性格与特质
<!-- 性格描述、思维风格、沟通特点 -->
<!-- 情绪模式 -->
[last_updated: unknown]

## 生活图景
<!-- 日常节奏、工作/学习状况 -->
<!-- 生活环境、重要的人事物 -->
[last_updated: unknown]

## 兴趣与偏好
<!-- 兴趣爱好、喜欢的领域 -->
<!-- 聊天风格偏好（鼓励/倾听/调侃） -->
<!-- 忌讳话题 -->
[last_updated: unknown]

## 当前状态
<!-- 最近在做什么、关注什么 -->
<!-- 当前心情/困扰/期待 -->
[last_updated: unknown]

## 我们之间
<!-- 重要的对话历史节点 -->
<!-- 内部笑话、特殊称呼、共同记忆 -->
[last_updated: unknown]
"""


class UserProfile:
    """USER.md 的读写门面。

    用法::

        profile = UserProfile(Path("workspace/USER.md"))
        content = profile.read()        # -> str
        profile.write(new_content)      # 覆盖写入
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def read(self) -> str:
        """读取 USER.md 全文。文件不存在时返回默认模板。"""
        if not self.path.exists():
            return _DEFAULT_USER_MD
        return self.path.read_text(encoding="utf-8")

    @property
    def exists(self) -> bool:
        """USER.md 是否已存在（即用户是否已被 Dream 更新过）。"""
        return self.path.exists()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def write(self, content: str) -> None:
        """覆盖写入 USER.md（原子写入：先写临时文件再替换）。"""
        tmp = self.path.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.path)

    def ensure_exists(self) -> bool:
        """确保 USER.md 存在；首次创建时写入默认模板。返回是否为新创建。"""
        if self.path.exists():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write(_DEFAULT_USER_MD)
        return True
