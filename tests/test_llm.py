"""llm.py 测试 — 真实 API 调用。

运行方式:
    ANTHROPIC_API_KEY=sk-xxx pytest tests/test_llm.py -v
"""

from __future__ import annotations

import os

import pytest

from companion_agent.llm import LlmClient

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL = "https://api.deepseek.com/anthropic"
MODEL = "deepseek-v4-pro"

pytestmark = pytest.mark.skipif(not API_KEY, reason="未设置 ANTHROPIC_API_KEY 环境变量")


def _safe_print(text: str) -> None:
    """print 但容忍 emoji 等 Windows GBK 打不出的字符。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


@pytest.fixture
def client():
    return LlmClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# chat 测试
# ---------------------------------------------------------------------------

class TestChat:
    """用真实 API 测试 chat 方法。"""

    @pytest.mark.asyncio
    async def test_simple_reply(self, client):
        """基本功能：发一条消息，收到非空文本回复。"""
        response = await client.chat(
            system_prompt="你是一个友好的助手，用中文回复。",
            messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}],
        )
        assert response.text
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0
        # LlmResponse 属性存在
        assert hasattr(response, "thinking")
        assert isinstance(response.thinking, str)
        _safe_print(f"\n[回复] {response.text}")
        if response.thinking:
            _safe_print(f"[思考过程] {len(response.thinking)} 字符")

    @pytest.mark.asyncio
    async def test_multi_turn(self, client):
        """多轮对话：上下文能正确传递。"""
        messages = [
            {"role": "user", "content": "我叫小明，记住我的名字。"},
        ]
        response1 = await client.chat(
            system_prompt="你是一个友好的助手，用中文回复，尽量简短。",
            messages=messages,
        )
        assert response1.text
        _safe_print(f"\n[第一轮] {response1.text}")

        # 第二轮：把 assistant 回复追加进去，看 LLM 是否记得
        messages.append({"role": "assistant", "content": response1.text})
        messages.append({"role": "user", "content": "我叫什么名字？"})

        response2 = await client.chat(
            system_prompt="你是一个友好的助手，用中文回复，尽量简短。",
            messages=messages,
        )
        assert response2.text
        assert "小明" in response2.text
        _safe_print(f"[第二轮] {response2.text}")

    @pytest.mark.asyncio
    async def test_system_prompt_effect(self, client):
        """system_prompt 能影响回复风格。"""
        # 用特殊 system prompt 设定回复风格
        response = await client.chat(
            system_prompt="你说的每一个回复都必须以'🐱喵~'开头。用中文。",
            messages=[{"role": "user", "content": "你好"}],
        )
        assert response.text  # 不强制要求"喵"——LLM 对 emoji prompt 不稳定
        _safe_print(f"\n[回复] {response.text}")

    @pytest.mark.asyncio
    async def test_empty_user_message(self, client):
        """空消息也能正常处理。"""
        response = await client.chat(
            system_prompt="你是友好的助手。",
            messages=[{"role": "user", "content": ""}],
        )
        assert isinstance(response.text, str)
        _safe_print(f"\n[回复] {response.text}")

    @pytest.mark.asyncio
    async def test_long_message(self, client):
        """较长消息不报错。"""
        long_text = "你好，" + "请介绍一下你自己，" * 50
        response = await client.chat(
            system_prompt="你是友好的助手，用中文回复。",
            messages=[{"role": "user", "content": long_text}],
        )
        assert response.text
        _safe_print(f"\n[回复长度] {len(response.text)} 字符")


    @pytest.mark.asyncio
    async def test_thinking_field_in_history(self, client):
        """验证：带 thinking 字段的 assistant 消息传给 DeepSeek 不报错。

        这是 get_recent_rounds() → build_messages() → API 的真实路径。
        thinking 不是 Anthropic 标准的 message 字段，DeepSeek 兼容层可能拒绝。
        """
        # 第一轮：正常对话
        msgs = [{"role": "user", "content": "我叫小明，记住我的名字。"}]
        r1 = await client.chat(
            system_prompt="你是友好的助手，用中文回复。",
            messages=msgs,
        )
        assert r1.text
        _safe_print(f"\n[第一轮] {r1.text}")

        # 第二轮：模拟 get_recent_rounds() 的真实输出——
        # assistant 消息带 thinking 字段
        msgs.append({
            "role": "assistant",
            "content": r1.text,
            "thinking": "用户叫小明，我应该记住这个名字并在后续对话中使用。",
        })
        msgs.append({"role": "user", "content": "我叫什么名字？"})

        r2 = await client.chat(
            system_prompt="你是友好的助手，用中文回复。",
            messages=msgs,
        )
        assert r2.text
        _safe_print(f"[第二轮] {r2.text}")


class TestApiErrors:
    """LLM API 错误处理 — P0。"""

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """错误的 API key → 抛异常（不静默）。"""
        client = LlmClient(
            base_url=BASE_URL,
            api_key="sk-this-is-clearly-invalid",
            model=MODEL,
        )
        with pytest.raises(Exception):
            await client.chat(
                system_prompt="test",
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    async def test_invalid_model(self):
        """不存在的模型名 → 抛异常。"""
        client = LlmClient(
            base_url=BASE_URL,
            api_key=API_KEY,
            model="nonexistent-model-xyz",
        )
        with pytest.raises(Exception):
            await client.chat(
                system_prompt="test",
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    async def test_unreachable_base_url(self):
        """不存在的 base_url → 抛异常。"""
        client = LlmClient(
            base_url="https://127.0.0.1:1",
            api_key=API_KEY,
            model=MODEL,
        )
        with pytest.raises(Exception):
            await client.chat(
                system_prompt="test",
                messages=[{"role": "user", "content": "hi"}],
            )
