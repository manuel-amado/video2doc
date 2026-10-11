"""Unit tests for Video2Doc configuration management and Settings model."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from video2doc.config import Settings, clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def mock_environment():
    """Isolate config unit tests from E2E integration fixtures."""
    yield


class TestSettingsDefaults:
    """Tests checking default settings against interface specifications."""

    def test_default_values(self):
        settings = Settings()
        assert settings.ollama_url == "http://localhost:11434"
        assert settings.vision_model == "llama3.2-vision"
        assert settings.text_model == "qwen2.5-coder:14b"
        assert settings.whisper_model == "turbo"
        assert settings.whisper_device == "auto"
        assert settings.whisper_language is None
        assert settings.scene_threshold == 0.3
        assert settings.debounce_seconds == 3.0
        assert settings.max_keyframes == 10
        assert settings.default_template == "sop"
        assert settings.output_dir == Path("output")
        assert settings.timeout_seconds == 300.0
        assert settings.model_num_ctx == 32768
        assert settings.model_temperature == 0.1
        assert settings.keep_frames is True
        assert settings.embed_images is True

    def test_properties_and_aliases(self):
        settings = Settings()
        assert settings.model_vision == settings.vision_model
        assert settings.model_text == settings.text_model
        assert settings.max_frames == settings.max_keyframes

    def test_alias_initialization(self):
        settings = Settings(
            model_vision="alt-vision-model",
            model_text="alt-text-model",
            max_frames=25,
        )
        assert settings.vision_model == "alt-vision-model"
        assert settings.text_model == "alt-text-model"
        assert settings.max_keyframes == 25


class TestSettingsValidation:
    """Tests checking schema validation and type coercion."""

    def test_output_dir_coercion_from_string(self):
        settings = Settings(output_dir="custom_output_dir")
        assert isinstance(settings.output_dir, Path)
        assert settings.output_dir == Path("custom_output_dir")

    def test_scene_threshold_bounds(self):
        with pytest.raises(ValidationError):
            Settings(scene_threshold=-0.1)
        with pytest.raises(ValidationError):
            Settings(scene_threshold=1.1)

        valid_lower = Settings(scene_threshold=0.0)
        assert valid_lower.scene_threshold == 0.0
        valid_upper = Settings(scene_threshold=1.0)
        assert valid_upper.scene_threshold == 1.0

    def test_debounce_seconds_positive(self):
        with pytest.raises(ValidationError):
            Settings(debounce_seconds=-1.0)
        valid = Settings(debounce_seconds=0.0)
        assert valid.debounce_seconds == 0.0

    def test_max_keyframes_bounds(self):
        with pytest.raises(ValidationError):
            Settings(max_keyframes=0)
        valid = Settings(max_keyframes=1)
        assert valid.max_keyframes == 1

    def test_timeout_seconds_positive(self):
        with pytest.raises(ValidationError):
            Settings(timeout_seconds=0.0)
        with pytest.raises(ValidationError):
            Settings(timeout_seconds=-5.0)

    def test_whisper_device_literal(self):
        valid_cuda = Settings(whisper_device="cuda")
        assert valid_cuda.whisper_device == "cuda"
        valid_cpu = Settings(whisper_device="cpu")
        assert valid_cpu.whisper_device == "cpu"
        valid_auto = Settings(whisper_device="auto")
        assert valid_auto.whisper_device == "auto"

        with pytest.raises(ValidationError):
            Settings(whisper_device="vulkan")


class TestSettingsEnvironmentResolution:
    """Tests checking precedence across environment variables and .env files."""

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("VIDEO2DOC_OLLAMA_URL", "http://remote-gpu:11434")
        monkeypatch.setenv("VIDEO2DOC_VISION_MODEL", "llava:13b")
        monkeypatch.setenv("VIDEO2DOC_SCENE_THRESHOLD", "0.45")

        settings = Settings()
        assert settings.ollama_url == "http://remote-gpu:11434"
        assert settings.vision_model == "llava:13b"
        assert settings.scene_threshold == 0.45

    def test_constructor_arg_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("VIDEO2DOC_OLLAMA_URL", "http://remote-gpu:11434")
        settings = Settings(ollama_url="http://explicit-host:11434")
        assert settings.ollama_url == "http://explicit-host:11434"

    def test_custom_env_prefix(self, monkeypatch):
        monkeypatch.setenv("CUSTOMAPP_TEXT_MODEL", "custom-gpt:latest")
        settings = Settings(_env_prefix="CUSTOMAPP_")
        assert settings.text_model == "custom-gpt:latest"

    def test_env_file_loading(self, tmp_path):
        env_file = tmp_path / "test.env"
        env_file.write_text(
            "VIDEO2DOC_OLLAMA_URL=http://envfile-host:11434\n"
            "VIDEO2DOC_MAX_KEYFRAMES=42\n"
            "VIDEO2DOC_DEFAULT_TEMPLATE=lecture\n",
            encoding="utf-8",
        )

        settings = Settings(_env_file=str(env_file))
        assert settings.ollama_url == "http://envfile-host:11434"
        assert settings.max_keyframes == 42
        assert settings.default_template == "lecture"

    def test_extra_fields_ignored(self):
        settings = Settings(unknown_future_field="value")
        assert not hasattr(settings, "unknown_future_field")


class TestGetSettingsFactory:
    """Tests checking the get_settings accessor and caching behavior."""

    def test_get_settings_singleton_and_clear(self):
        clear_settings_cache()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

        clear_settings_cache()
        s3 = get_settings()
        assert s3 is not s1
        assert s3 == s1

    def test_get_settings_with_overrides(self):
        custom = get_settings(vision_model="override-model")
        default = get_settings()
        assert custom.vision_model == "override-model"
        assert default.vision_model == "llama3.2-vision"
        assert custom is not default

    def test_get_settings_with_custom_env_file(self, tmp_path):
        env_file = tmp_path / "custom.env"
        env_file.write_text("VIDEO2DOC_DEFAULT_TEMPLATE=executive\n", encoding="utf-8")
        settings = get_settings(env_file=env_file)
        assert settings.default_template == "executive"
