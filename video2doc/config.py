"""Configuration management for Video2Doc using Pydantic Settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings resolved from CLI flags, env vars, .env files, and defaults.

    Precedence order:
      1. Explicit constructor arguments / CLI overrides
      2. Environment variables (prefixed with VIDEO2DOC_ or custom prefix)
      3. .env file
      4. Default values defined on the model
    """

    model_config = SettingsConfigDict(
        env_prefix="VIDEO2DOC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core AI & Service Configuration
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local or remote Ollama server",
    )
    vision_model: str = Field(
        default="llama3.2-vision",
        description="Vision LLM model identifier for keyframe analysis",
    )
    text_model: str = Field(
        default="qwen2.5-coder:14b",
        description="Text LLM model identifier for document synthesis",
    )
    whisper_model: str = Field(
        default="turbo",
        description="OpenAI Whisper model variant (e.g. tiny, base, small, medium, turbo, large-v3)",
    )
    whisper_device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Compute device for Whisper execution ('cuda', 'cpu', 'auto')",
    )
    whisper_language: str | None = Field(
        default=None,
        description="Language code for Whisper transcription (ISO 639-1) or None for auto-detection",
    )

    # Keyframe Extraction Parameters
    scene_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="FFmpeg scene detection sensitivity threshold (0.0 to 1.0)",
    )
    debounce_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Minimum elapsed seconds between extracted keyframes to avoid redundant frames",
    )
    max_keyframes: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of keyframes to extract from a video",
    )

    # Synthesis & Generation
    default_template: str = Field(
        default="sop",
        description="Default template identifier or path ('sop', 'lecture', 'executive', etc.)",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Target directory for generated Markdown documentation and assets",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description="HTTP request timeout in seconds for Ollama API calls",
    )
    model_num_ctx: int = Field(
        default=32768,
        ge=2048,
        description="Context window token size for Ollama inference",
    )
    model_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Generation temperature for LLM responses",
    )
    keep_frames: bool = Field(
        default=True,
        description="Whether to keep extracted keyframe image files alongside Markdown output",
    )
    embed_images: bool = Field(
        default=True,
        description="Whether to embed relative keyframe image links inside generated Markdown",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        """Handle alternate naming conventions (e.g., model_vision -> vision_model)."""
        if isinstance(data, dict):
            if "model_vision" in data and "vision_model" not in data:
                data["vision_model"] = data.pop("model_vision")
            if "model_text" in data and "text_model" not in data:
                data["text_model"] = data.pop("model_text")
            if "max_frames" in data and "max_keyframes" not in data:
                data["max_keyframes"] = data.pop("max_frames")
        return data

    @field_validator("output_dir", mode="before")
    @classmethod
    def parse_output_dir(cls, value: Any) -> Path:
        """Ensure output_dir is parsed into a Path instance."""
        if isinstance(value, str):
            return Path(value)
        return value

    @property
    def model_vision(self) -> str:
        """Alias property matching alternate survey naming."""
        return self.vision_model

    @property
    def model_text(self) -> str:
        """Alias property matching alternate survey naming."""
        return self.text_model

    @property
    def max_frames(self) -> int:
        """Alias property matching alternate survey naming."""
        return self.max_keyframes


@lru_cache(maxsize=1)
def _get_cached_default_settings() -> Settings:
    """Return a cached default Settings instance."""
    return Settings()


def get_settings(
    env_file: str | Path | None = None,
    env_prefix: str | None = None,
    **overrides: Any,
) -> Settings:
    """Retrieve application settings.

    If called without arguments, returns a cached singleton instance.
    If custom env_file, env_prefix, or overrides are provided, instantiates
    and returns a fresh Settings instance.
    """
    if env_file is None and env_prefix is None and not overrides:
        return _get_cached_default_settings()

    init_kwargs: dict[str, Any] = dict(overrides)
    if env_file is not None:
        init_kwargs["_env_file"] = str(env_file)
    if env_prefix is not None:
        init_kwargs["_env_prefix"] = env_prefix

    return Settings(**init_kwargs)


def clear_settings_cache() -> None:
    """Clear the cached default settings instance."""
    _get_cached_default_settings.cache_clear()
