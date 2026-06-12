"""config.py 测试。"""

from __future__ import annotations

from pathlib import Path

from companion_agent.config import load_config


class TestDefaults:
    """默认值。"""

    def test_all_defaults(self):
        cfg = load_config()
        assert cfg.llm.base_url == "https://api.deepseek.com/anthropic"
        assert cfg.llm.model == "deepseek-v4-pro"
        assert cfg.llm.max_tokens == 4096
        assert cfg.companion.dream_trigger_rounds == 30
        assert cfg.companion.max_history_rounds == 100
        assert cfg.proactive.tick_interval == 1800
        assert cfg.proactive.start_hour == 7
        assert cfg.proactive.end_hour == 23
        assert cfg.bot.delay_per_char == 0.2


class TestEnvOverride:
    """环境变量覆盖。"""

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://custom.api.com")
        monkeypatch.setenv("ANTHROPIC_MODEL", "custom-model")
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "8192")
        monkeypatch.setenv("COMPANION_DREAM_TRIGGER_ROUNDS", "50")
        monkeypatch.setenv("PROACTIVE_START_HOUR", "8")
        monkeypatch.setenv("PROACTIVE_END_HOUR", "22")
        monkeypatch.setenv("BOT_DELAY_PER_CHAR", "0.3")

        cfg = load_config()

        assert cfg.llm.api_key == "sk-test"
        assert cfg.llm.base_url == "https://custom.api.com"
        assert cfg.llm.model == "custom-model"
        assert cfg.llm.max_tokens == 8192
        assert cfg.companion.dream_trigger_rounds == 50
        assert cfg.proactive.start_hour == 8
        assert cfg.proactive.end_hour == 22
        assert cfg.bot.delay_per_char == 0.3

    def test_env_partial_override(self, monkeypatch):
        """只覆盖部分字段，其余保持默认。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-partial")
        cfg = load_config()

        assert cfg.llm.api_key == "sk-partial"
        assert cfg.llm.model == "deepseek-v4-pro"  # 默认
        assert cfg.companion.dream_trigger_rounds == 30  # 默认

    def test_invalid_int_fallback(self, monkeypatch):
        """非数字环境变量 → 回退默认。"""
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "not-a-number")
        cfg = load_config()
        assert cfg.llm.max_tokens == 4096  # 默认


class TestYamlLoad:
    """config.yaml 加载。"""

    def test_yaml_merge(self, tmpdir, monkeypatch):
        from companion_agent.config import Config, _merge_yaml

        yaml_path = Path(str(tmpdir)) / "config.yaml"
        yaml_path.write_text("""\
llm:
  base_url: "https://yaml.api.com"
  api_key: "${YAML_API_KEY}"
  model: "yaml-model"
companion:
  dream_trigger_rounds: 20
""", encoding="utf-8")

        monkeypatch.setenv("YAML_API_KEY", "sk-from-yaml-ref")

        cfg = Config()
        _merge_yaml(cfg, yaml_path)

        assert cfg.llm.base_url == "https://yaml.api.com"
        assert cfg.llm.api_key == "sk-from-yaml-ref"
        assert cfg.llm.model == "yaml-model"
        assert cfg.companion.dream_trigger_rounds == 20
