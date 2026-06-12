# companion-agent

> [中文版本](README.zh-CN.md)

A personal WeChat companion AI — built on DeepSeek v4-pro + weixin-bot-python. Remembers everything about you and builds meaningful emotional bonds.

## Architecture

```
WeChat msg → bot.py → engine.py → LLM (DeepSeek)
                        ↑
             prompt.py ← SOUL.md + USER.md
                        ↓
                  session.jsonl → Dream → USER.md
                        ↓
                  proactive.py (background pings)
```

Three-layer memory:

| Layer | Storage | Writer | Purpose |
|-------|---------|--------|---------|
| Persona | `workspace/SOUL.md` | You | System prompt core |
| Knowledge | `workspace/USER.md` | Dream (auto) | What we know about the user |
| History | `workspace/sessions/default.jsonl` | Engine (auto) | Conversation context |

## Quick Start

### Prerequisites

- Python ≥ 3.12
- DeepSeek API key (or any Anthropic-compatible endpoint)
- WeChat iLink bot account

### Install

```bash
git clone https://github.com/vergica/companion-agent.git
cd companion-agent

# Create virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Core dependencies
pip install anthropic pyyaml python-dotenv

# weixin-bot-python (from GitHub)
pip install git+https://github.com/vergica/weixin-bot-python.git

# Dev tools (optional)
pip install pytest pytest-asyncio
```

### Configure

1. Set your API key:

```bash
# Option A: environment variable (recommended)
export ANTHROPIC_API_KEY=sk-your-key

# Option B: .env file (for development)
echo ANTHROPIC_API_KEY=sk-your-key > .env
```

2. Create your persona: `cp workspace/SOUL.md.template workspace/SOUL.md` and edit as desired.

3. Optional: create a `config.yaml` to override defaults.

### Run

```bash
python -m companion_agent
```

### Configuration Reference

All settings have defaults. Override via env vars or `config.yaml`.

| Env Variable | Default | Description |
|--------------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required** |
| `ANTHROPIC_BASE_URL` | `https://api.deepseek.com/anthropic` | |
| `ANTHROPIC_MODEL` | `deepseek-v4-pro` | |
| `ANTHROPIC_MAX_TOKENS` | `4096` | |
| `COMPANION_DREAM_TRIGGER_ROUNDS` | `30` | Rounds before memory consolidation |
| `COMPANION_MAX_HISTORY_ROUNDS` | `100` | Conversation rounds fed to LLM |
| `PROACTIVE_TICK_INTERVAL` | `1800` | Proactive check interval (seconds) |
| `PROACTIVE_START_HOUR` | `7` | Earliest hour for proactive messages |
| `PROACTIVE_END_HOUR` | `23` | Latest hour for proactive messages |
| `BOT_DELAY_PER_CHAR` | `0.2` | Multi-message send delay coefficient |

Priority: **env vars > config.yaml > defaults**

Sample `config.yaml`:

```yaml
llm:
  base_url: "https://api.deepseek.com/anthropic"
  api_key: "${ANTHROPIC_API_KEY}"   # resolved from environment
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

`${VAR}` placeholders are replaced with environment variable values at load time.

## Tests

```bash
# All tests
pytest -v

# Non-API only (no network needed)
pytest tests/test_prompt.py tests/test_engine.py tests/test_proactive.py tests/test_bot.py tests/test_config.py -v

# API tests (requires ANTHROPIC_API_KEY)
pytest tests/test_llm.py tests/test_memory.py -v
```

## Project Structure

```
companion-agent/
├── pyproject.toml
├── conftest.py                  # pytest config
├── companion_agent/
│   ├── __init__.py
│   ├── __main__.py              # Entry: wiring + startup
│   ├── config.py                # Config loading
│   ├── llm.py                   # Anthropic SDK wrapper
│   ├── bot.py                   # WeChat adapter
│   ├── proactive.py             # Proactive messaging
│   ├── agent/
│   │   ├── engine.py            # Conversation orchestration
│   │   └── prompt.py            # Pure functions: prompt assembly
│   └── memory/
│       ├── session.py           # Chat log (JSONL)
│       ├── dream.py             # Memory consolidation
│       └── user_profile.py      # USER.md I/O
├── tests/
│   ├── test_prompt.py           # Pure functions
│   ├── test_engine.py           # Mock LLM
│   ├── test_bot.py              # Mock WeChat
│   ├── test_proactive.py        # Mock engine
│   ├── test_config.py           # Env / yaml
│   ├── test_llm.py              # Real API
│   └── test_memory.py           # Real API + I/O
└── workspace/
    ├── SOUL.md                  # Your persona (gitignored; copy from template)
    └── SOUL.md.template         # Persona template
```

## License

MIT
