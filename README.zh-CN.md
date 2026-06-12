# companion-agent

> [English](README.md)

个人微信陪伴 AI —— 基于 DeepSeek v4-pro + weixin-bot-python，记住你的一切，建立深度情感羁绊。

## 架构

```
微信消息 → bot.py → engine.py → LLM (DeepSeek)
                         ↑
              prompt.py  ←  SOUL.md + USER.md
                         ↓
                   session.jsonl  →  Dream  →  USER.md
                         ↓
                   proactive.py (后台主动消息)
```

三层记忆：

| 层 | 存储 | 读写 | 用途 |
|----|------|------|------|
| 人格 | `workspace/SOUL.md` | 人手写 | system prompt 核心 |
| 认知 | `workspace/USER.md` | Dream 自动维护 | 对用户的了解 |
| 对话 | `workspace/sessions/default.jsonl` | engine 自动写入 | 历史上下文 |

## 快速开始

### 前置条件

- Python ≥ 3.12
- DeepSeek API key（或任何 Anthropic 兼容接口）
- 微信 iLink bot 账号

### 安装

```bash
git clone https://github.com/vergica/companion-agent.git
cd companion-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 核心依赖
pip install anthropic pyyaml python-dotenv

# weixin-bot-python 从 GitHub 安装
pip install git+https://github.com/vergica/weixin-bot-python.git

# 开发工具（可选）
pip install pytest pytest-asyncio
```

### 配置

1. 设置 API key：

```bash
# 方式一：环境变量（推荐）
export ANTHROPIC_API_KEY=sk-your-key

# 方式二：.env 文件（开发用）
echo ANTHROPIC_API_KEY=sk-your-key > .env
```

2. 创建人格文件：`cp workspace/SOUL.md.template workspace/SOUL.md`，按需修改。

3. 可选：创建 `config.yaml` 覆盖默认配置。

### 运行

```bash
python -m companion_agent
```

### 配置项

所有配置都有默认值，可通过环境变量或 `config.yaml` 覆盖。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ANTHROPIC_API_KEY` | — | **必需** |
| `ANTHROPIC_BASE_URL` | `https://api.deepseek.com/anthropic` | |
| `ANTHROPIC_MODEL` | `deepseek-v4-pro` | |
| `ANTHROPIC_MAX_TOKENS` | `4096` | |
| `COMPANION_DREAM_TRIGGER_ROUNDS` | `30` | 累计 N 轮新对话后触发记忆整合 |
| `COMPANION_MAX_HISTORY_ROUNDS` | `100` | 每次注入 LLM 的对话轮数 |
| `PROACTIVE_TICK_INTERVAL` | `1800` | 主动消息检查间隔（秒） |
| `PROACTIVE_START_HOUR` | `7` | 允许主动消息的起始时间 |
| `PROACTIVE_END_HOUR` | `23` | 允许主动消息的结束时间 |
| `BOT_DELAY_PER_CHAR` | `0.2` | 多段消息间延迟系数 |

优先级：**环境变量 > config.yaml > 默认值**

`config.yaml` 示例：

```yaml
llm:
  base_url: "https://api.deepseek.com/anthropic"
  api_key: "${ANTHROPIC_API_KEY}"   # 从环境变量读取
  model: "deepseek-v4-pro"
  max_tokens: 4096

companion:
  dream_trigger_rounds: 30
  max_history_rounds: 100

proactive:
  tick_interval: 1800
  start_hour: 7
  end_hour: 23

bot:
  delay_per_char: 0.2
```

`${VAR}` 占位符自动替换为环境变量值。

## 测试

```bash
# 全量
pytest -v

# 仅非 API 测试（无需网络）
pytest tests/test_prompt.py tests/test_engine.py tests/test_proactive.py tests/test_bot.py tests/test_config.py -v

# 仅 API 测试（需要 ANTHROPIC_API_KEY）
pytest tests/test_llm.py tests/test_memory.py -v
```

## 项目结构

```
companion-agent/
├── pyproject.toml
├── conftest.py                  # pytest 配置
├── companion_agent/
│   ├── __init__.py
│   ├── __main__.py              # 入口：组装 + 启动
│   ├── config.py                # 配置加载
│   ├── llm.py                   # Anthropic SDK 封装
│   ├── bot.py                   # 微信适配层
│   ├── proactive.py             # 主动消息模块
│   ├── agent/
│   │   ├── engine.py            # 对话编排
│   │   └── prompt.py            # 纯函数：拼 prompt
│   └── memory/
│       ├── session.py           # 对话记录 (JSONL)
│       ├── dream.py             # 记忆整合
│       └── user_profile.py      # USER.md 读写
├── tests/
│   ├── test_prompt.py           # 纯函数
│   ├── test_engine.py           # mock LLM
│   ├── test_bot.py              # mock 微信
│   ├── test_proactive.py        # mock engine
│   ├── test_config.py           # env/yaml
│   ├── test_llm.py              # 真实 API
│   └── test_memory.py           # 真实 API + I/O
└── workspace/
    ├── SOUL.md                  # 你的 AI 人格（不入 git，从 template 复制）
    └── SOUL.md.template         # 人格模板
```

## 许可

MIT
