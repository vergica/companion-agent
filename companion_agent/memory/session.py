"""会话管理 — 单个用户的原始对话记录。

纯文件 I/O，无依赖。每条消息存一行 JSONL。
首行是元数据，其中 dream_cursor 标记 Dream 已处理到的消息位置。

文件格式::

    {"_type": "metadata", "created_at": "...", "dream_cursor": 5}
    {"role": "user", "content": "今天好累啊", "timestamp": "2026-06-13 14:30:00"}
    {"role": "assistant", "content": "怎么了？跟我说说", "timestamp": "2026-06-13 14:30:05"}

一个 "round" = 一条 user 消息 + 其后紧跟的所有 assistant 消息（可能多条）。
dream_cursor 记录的是"已处理到的消息索引"（0-based，指向最后一条已处理消息）。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path


class Session:
    """单个用户会话的读写门面。

    用法::

        sess = Session(Path("workspace/sessions/user123.jsonl"))
        sess.add_message("user", "你好")
        sess.add_message("assistant", "嗨！今天怎么样？")

        # 取最近 50 轮对话给 LLM
        messages = sess.get_recent_rounds(50)

        # Dream 用：取未处理的消息
        unprocessed = sess.get_unprocessed_for_dream()
        # ...
        sess.mark_dream_done()
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 追加消息
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str, *, thinking: str | None = None) -> None:
        """追加一条消息。role 为 "user" 或 "assistant"。

        thinking 仅用于 assistant 消息，存储模型的思考过程。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        record: dict = {
            "role": role,
            "content": content,
            "timestamp": _now(),
        }
        if thinking:
            record["thinking"] = thinking

        with self._lock:
            if not self.path.exists():
                self._write_metadata({"created_at": _now(), "dream_cursor": -1})
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 读取 — 给 LLM context 用
    # ------------------------------------------------------------------

    def get_recent_rounds(
        self, max_rounds: int, *, include_thinking: bool = False,
    ) -> list[dict]:
        """取最近 N 轮对话，直接用于 LLM messages 数组。

        返回格式: [{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}, ...]
        保证以 user 开头。

        include_thinking=False 时不带思考过程（默认，省 token）。
        """
        messages = self._read_messages()

        # 从后往前数 max_rounds 个 user 消息
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            return []

        if max_rounds <= 0:
            return []
        start = user_indices[-min(max_rounds, len(user_indices))]
        sliced = messages[start:]

        # 转为 LLM 格式
        result: list[dict] = []
        for m in sliced:
            entry = {"role": m["role"], "content": m["content"]}
            if include_thinking and m.get("thinking"):
                entry["thinking"] = m["thinking"]
            result.append(entry)
        return result

    def get_message_count(self) -> int:
        """消息总数（不含元数据行）。"""
        return len(self._read_messages())

    # ------------------------------------------------------------------
    # 读取 — 给 Dream 用
    # ------------------------------------------------------------------

    def get_unprocessed_for_dream(self) -> list[dict]:
        """返回 dream_cursor 之后的所有消息。

        Dream 用这些消息作为素材来更新 USER.md。
        返回包含 role、content、timestamp 的完整字典列表。
        """
        cursor = self._read_dream_cursor()
        messages = self._read_messages()
        # cursor 是"已处理到第几条"（0-based），取它之后的
        return messages[cursor + 1:]

    def mark_dream_done(self) -> None:
        """标记所有消息已被 Dream 处理——将 dream_cursor 设为最后一条消息的索引。"""
        messages = self._read_messages()
        new_cursor = len(messages) - 1  # 最后一条的索引
        if new_cursor < 0:
            return
        self._update_dream_cursor(new_cursor)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    @property
    def unprocessed_rounds(self) -> int:
        """Dream 未处理的对话轮数。

        用于触发判定：超过 dream_trigger_rounds 就触发 Dream。
        """
        cursor = self._read_dream_cursor()
        messages = self._read_messages()
        unprocessed = messages[cursor + 1:]
        return sum(1 for m in unprocessed if m.get("role") == "user")

    @property
    def total_rounds(self) -> int:
        """总对话轮数（user 消息总数）。"""
        return sum(1 for m in self._read_messages() if m.get("role") == "user")

    @property
    def last_message_time(self) -> str | None:
        """最后一条消息的时间戳。无消息时返回 None。"""
        messages = self._read_messages()
        if not messages:
            return None
        return messages[-1].get("timestamp", None)

    @property
    def created_at(self) -> str:
        """会话创建时间。"""
        meta = self._read_metadata()
        return meta.get("created_at", "unknown")

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """清空会话，重置 dream_cursor。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # 直接截断文件，只写一行元数据
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"_type": "metadata", "created_at": _now(), "dream_cursor": -1},
                    ensure_ascii=False,
                ) + "\n")

    # ------------------------------------------------------------------
    # 内部：元数据读写
    # ------------------------------------------------------------------

    def _read_metadata(self) -> dict:
        """读取元数据行。"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
                if first:
                    data = json.loads(first)
                    if data.get("_type") == "metadata":
                        return data
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return {}

    def _write_metadata(self, meta: dict) -> None:
        """覆盖写入元数据行。文件不存在则创建。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta["_type"] = "metadata"
        # 先读已有消息行
        messages = []
        if self.path.exists():
            messages = self._read_messages()
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                for m in messages:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def _read_dream_cursor(self) -> int:
        meta = self._read_metadata()
        cursor = meta.get("dream_cursor", -1)
        return cursor if isinstance(cursor, int) else -1

    def _update_dream_cursor(self, new_cursor: int) -> None:
        """更新 dream_cursor（保留其他元数据字段）。"""
        meta = self._read_metadata()
        meta["dream_cursor"] = new_cursor
        self._write_metadata(meta)

    # ------------------------------------------------------------------
    # 内部：消息行读取
    # ------------------------------------------------------------------

    def _read_messages(self) -> list[dict]:
        """读取所有消息行（跳过元数据行和空行），返回内部格式列表。"""
        messages = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("_type") == "metadata":
                        continue
                    messages.append(data)
        except FileNotFoundError:
            pass
        return messages


def _now() -> str:
    """当前时间 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
