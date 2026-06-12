"""LLM 客户端 — 封装 Anthropic SDK，提供统一的异步调用接口。

项目中唯一直接 import anthropic 的模块。其他模块（agent、dream、proactive）
都通过 LlmClient 调用 LLM，不感知底层 SDK 细节。

LlmResponse 是一个纯数据类，可安全导入（不触发 anthropic import）。
"""

from __future__ import annotations

import anthropic


class LlmResponse:
    """LLM 一次调用的完整返回。

    - thinking: 模型的思考过程（DeepSeek v4-pro 的推理链），可能为空字符串
    - text: 模型的最终输出文本
    """

    __slots__ = ("thinking", "text")

    def __init__(self, *, thinking: str = "", text: str = "") -> None:
        self.thinking = thinking
        self.text = text

    def __repr__(self) -> str:
        return f"LlmResponse(thinking={len(self.thinking)}chars, text={len(self.text)}chars)"


class LlmClient:
    """Anthropic 兼容接口的 LLM 客户端。

    用法::

        client = LlmClient(
            base_url="https://api.deepseek.com/anthropic",
            api_key="sk-xxx",
            model="deepseek-v4-pro",
            max_tokens=4096,
        )
        response = await client.chat(
            system_prompt="你是用户的陪伴 AI...",
            messages=[{"role": "user", "content": "你好"}],
        )
        print(response.text)       # 最终输出
        print(response.thinking)    # 思考过程
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            base_url=base_url,
            api_key=api_key,
        )

    async def chat(self, system_prompt: str, messages: list[dict]) -> LlmResponse:
        """发送请求，返回 LlmResponse（含 thinking + text）。"""
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )
        thinking, text = self._extract(response)
        return LlmResponse(thinking=thinking, text=text)

    @staticmethod
    def _extract(response) -> tuple[str, str]:
        """从 anthropic 响应中提取 思考过程 和 输出文本。

        返回 ``(thinking, text)`` 二元组。
        """
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
        return "".join(thinking_parts), "".join(text_parts)
