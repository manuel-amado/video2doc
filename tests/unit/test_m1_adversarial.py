"""Adversarial stress-testing suite for Milestone 1: Packaging, Git Hygiene & Config Core.

Tests edge cases, boundary conditions, invalid inputs, environment overrides,
non-existent files, and exception hierarchy contracts.
"""

from pathlib import Path
from unittest.mock import patch
import os
import pytest
from pydantic import ValidationError

from video2doc.config import Settings, clear_settings_cache, get_settings
from video2doc.exceptions import (
    MediaCorruptError,
    MissingDependencyError,
    ModelNotFoundError,
    OllamaConnectionError,
    TemplateNotFoundError,
    Video2DocError,
)


@pytest.fixture(autouse=True)
def mock_environment():
    """Ensure isolation from global integration mocks."""
    yield


class TestConfigAdversarialBoundaries:
    """Adversarial stress tests for configuration schema boundaries."""

    def test_scene_threshold_negative_boundary(self):
        """Negative values just below 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(scene_threshold=-0.000001)

    def test_scene_threshold_upper_boundary_violation(self):
        """Values just above 1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(scene_threshold=1.000001)

    def test_scene_threshold_extreme_bounds(self):
        """Extreme numbers (-1000.0, 1000.0) must raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(scene_threshold=-1000.0)
        with pytest.raises(ValidationError):
            Settings(scene_threshold=1000.0)

    def test_scene_threshold_exact_boundaries(self):
        """Exact boundaries 0.0 and 1.0 must be accepted."""
        s_zero = Settings(scene_threshold=0.0)
        assert s_zero.scene_threshold == 0.0
        s_one = Settings(scene_threshold=1.0)
        assert s_one.scene_threshold == 1.0

    def test_debounce_seconds_negative_boundary(self):
        """Negative debounce seconds must raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(debounce_seconds=-0.0001)
        with pytest.raises(ValidationError):
            Settings(debounce_seconds=-10.0)

    def test_debounce_seconds_zero_and_integer_coercion(self):
        """Zero debounce and integer coercion must be supported."""
        s_zero = Settings(debounce_seconds=0.0)
        assert s_zero.debounce_seconds == 0.0
        s_int = Settings(debounce_seconds=5)
        assert s_int.debounce_seconds == 5.0
        assert isinstance(s_int.debounce_seconds, float)

    def test_whisper_device_invalid_literals(self):
        """Non-supported device identifiers must be strictly rejected."""
        invalid_devices = ["gpu", "tpu", "directml", "mps", "vulkan", "", "123", "CUDA", "CPU", "Auto"]
        for dev in invalid_devices:
            with pytest.raises(ValidationError):
                Settings(whisper_device=dev)

    def test_whisper_device_valid_literals(self):
        """Supported device identifiers must be strictly accepted."""
        for dev in ["cuda", "cpu", "auto"]:
            s = Settings(whisper_device=dev)
            assert s.whisper_device == dev

    def test_max_keyframes_bounds(self):
        """max_keyframes must be >= 1 and <= 1000."""
        with pytest.raises(ValidationError):
            Settings(max_keyframes=0)
        with pytest.raises(ValidationError):
            Settings(max_keyframes=-5)
        with pytest.raises(ValidationError):
            Settings(max_keyframes=1001)

        s_min = Settings(max_keyframes=1)
        assert s_min.max_keyframes == 1
        s_max = Settings(max_keyframes=1000)
        assert s_max.max_keyframes == 1000

    def test_timeout_seconds_strict_positivity(self):
        """timeout_seconds must be strictly > 0.0."""
        with pytest.raises(ValidationError):
            Settings(timeout_seconds=0.0)
        with pytest.raises(ValidationError):
            Settings(timeout_seconds=-0.1)

        s = Settings(timeout_seconds=0.001)
        assert s.timeout_seconds == 0.001


class TestConfigAdversarialStringAndPathHandling:
    """Stress testing empty strings, path parsing, and aliases."""

    def test_output_dir_empty_string_coercion(self):
        """Empty string for output_dir coerces into Path('.')."""
        s = Settings(output_dir="")
        assert isinstance(s.output_dir, Path)
        assert s.output_dir == Path("")

    def test_output_dir_nested_path_parsing(self):
        """Complex path string parses correctly into Path object."""
        s = Settings(output_dir="nested/sub/directory/output")
        assert isinstance(s.output_dir, Path)
        assert s.output_dir == Path("nested/sub/directory/output")

    def test_alias_interoperability_and_precedence(self):
        """Explicit model_vision alias maps correctly, canonical vision_model takes precedence."""
        s_alias = Settings(model_vision="custom-vision-model")
        assert s_alias.vision_model == "custom-vision-model"
        assert s_alias.model_vision == "custom-vision-model"

        # Canonical takes precedence if both provided
        s_both = Settings(vision_model="canonical-model", model_vision="alias-model")
        assert s_both.vision_model == "canonical-model"

    def test_empty_string_configuration_values(self):
        """Empty strings are accepted at config level; downstream validation handles them."""
        s = Settings(ollama_url="", vision_model="", text_model="", default_template="")
        assert s.ollama_url == ""
        assert s.vision_model == ""
        assert s.text_model == ""
        assert s.default_template == ""


class TestConfigAdversarialEnvironmentAndFiles:
    """Stress testing environment variable overrides and non-existent files."""

    def test_nonexistent_env_file_degrades_gracefully(self, tmp_path):
        """Passing non-existent .env file should not crash; it loads default settings."""
        nonexistent = tmp_path / "does_not_exist_xyz.env"
        assert not nonexistent.exists()

        settings = get_settings(env_file=nonexistent)
        assert settings.ollama_url == "http://localhost:11434"
        assert settings.scene_threshold == 0.3

    def test_empty_env_file_loads_defaults(self, tmp_path):
        """Empty .env file loads default values without corruption."""
        empty_env = tmp_path / "empty.env"
        empty_env.write_text("", encoding="utf-8")

        settings = get_settings(env_file=empty_env)
        assert settings.ollama_url == "http://localhost:11434"

    def test_env_var_override_invalid_type_raises(self, monkeypatch):
        """Non-numeric string in numeric env var must raise ValidationError."""
        monkeypatch.setenv("VIDEO2DOC_SCENE_THRESHOLD", "not_a_number")
        with pytest.raises(ValidationError):
            Settings()

    def test_env_var_override_out_of_bounds_raises(self, monkeypatch):
        """Out-of-bounds float in env var must raise ValidationError."""
        monkeypatch.setenv("VIDEO2DOC_SCENE_THRESHOLD", "2.5")
        with pytest.raises(ValidationError):
            Settings()

    def test_env_var_override_case_insensitivity(self, monkeypatch):
        """Environment variables are resolved case-insensitively."""
        monkeypatch.setenv("video2doc_ollama_url", "http://case-insensitive-host:11434")
        settings = Settings()
        assert settings.ollama_url == "http://case-insensitive-host:11434"

    def test_constructor_overrides_beat_environment(self, monkeypatch):
        """Constructor arguments take highest precedence over environment variables."""
        monkeypatch.setenv("VIDEO2DOC_OLLAMA_URL", "http://env-host:11434")
        monkeypatch.setenv("VIDEO2DOC_SCENE_THRESHOLD", "0.8")

        settings = Settings(ollama_url="http://explicit-host:11434", scene_threshold=0.1)
        assert settings.ollama_url == "http://explicit-host:11434"
        assert settings.scene_threshold == 0.1


class TestExceptionsAdversarialHierarchy:
    """Stress testing exception hierarchy, structured details, and chaining."""

    def test_all_exceptions_inherit_from_video2doc_error(self):
        """All domain exceptions must inherit from Video2DocError and Exception."""
        exception_classes = [
            MissingDependencyError,
            OllamaConnectionError,
            ModelNotFoundError,
            MediaCorruptError,
            TemplateNotFoundError,
        ]
        for exc_cls in exception_classes:
            assert issubclass(exc_cls, Video2DocError)
            assert issubclass(exc_cls, Exception)

    def test_structured_details_are_always_dictionaries(self):
        """All exception instances must provide a non-None details dictionary."""
        instances = [
            Video2DocError("base error"),
            MissingDependencyError("ffmpeg"),
            OllamaConnectionError("http://127.0.0.1:11434"),
            ModelNotFoundError("vision-model"),
            MediaCorruptError("video.mp4"),
            TemplateNotFoundError("sop"),
        ]
        for inst in instances:
            assert isinstance(inst.details, dict)

    def test_details_keys_contract(self):
        """Structured details dictionary must contain expected domain metadata keys."""
        e1 = MissingDependencyError("ffprobe", remediation="Install ffprobe")
        assert e1.details == {"dependency": "ffprobe", "remediation": "Install ffprobe"}

        e2 = OllamaConnectionError("http://localhost:11434", reason="Connection refused")
        assert e2.details == {"url": "http://localhost:11434", "reason": "Connection refused"}

        e3 = ModelNotFoundError("custom-model", available_models=["m1", "m2"])
        assert e3.details == {"model_name": "custom-model", "available_models": ["m1", "m2"]}

        e4 = MediaCorruptError("sample.mp4", reason="Invalid header")
        assert e4.details == {"media_path": "sample.mp4", "reason": "Invalid header"}

        e5 = TemplateNotFoundError("custom.yaml", searched_paths=["/path/a", "/path/b"])
        assert e5.details == {"template_name": "custom.yaml", "searched_paths": ["/path/a", "/path/b"]}

    def test_catch_block_polymorphism(self):
        """Single 'except Video2DocError' block must catch every subclass."""
        subclasses = [
            MissingDependencyError("dep"),
            OllamaConnectionError("url"),
            ModelNotFoundError("model"),
            MediaCorruptError("path"),
            TemplateNotFoundError("tmpl"),
        ]
        for exc in subclasses:
            caught = False
            try:
                raise exc
            except Video2DocError as caught_exc:
                caught = True
                assert caught_exc is exc
                assert len(str(caught_exc)) > 0
            assert caught is True

    def test_exception_chaining_preserves_root_cause(self):
        """Exception chaining via 'from' must preserve __cause__ and details."""
        root_cause = TimeoutError("Socket timed out after 300s")
        try:
            try:
                raise root_cause
            except TimeoutError as te:
                raise OllamaConnectionError("http://ollama:11434", reason=str(te)) from te
        except Video2DocError as caught:
            assert caught.__cause__ is root_cause
            assert caught.details["reason"] == "Socket timed out after 300s"
