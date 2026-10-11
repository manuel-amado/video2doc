"""Custom exceptions hierarchy for Video2Doc."""

from __future__ import annotations

from typing import Any


class Video2DocError(Exception):
    """Base exception for all Video2Doc domain errors."""

    def __init__(self, message: str, *args: Any, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, *args)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


class MissingDependencyError(Video2DocError):
    """Raised when a required system or binary dependency (e.g., FFmpeg, ffprobe) is missing."""

    def __init__(
        self,
        dependency_name: str,
        message: str | None = None,
        remediation: str | None = None,
    ) -> None:
        self.dependency_name = dependency_name
        self.remediation = remediation
        msg = message or f"Missing required dependency: '{dependency_name}'."
        if remediation:
            msg += f" {remediation}"
        super().__init__(msg, details={"dependency": dependency_name, "remediation": remediation})


class OllamaConnectionError(Video2DocError):
    """Raised when unable to establish a connection to the Ollama server."""

    def __init__(
        self,
        url: str,
        message: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.url = url
        self.reason = reason
        msg = message or f"Failed to connect to Ollama server at '{url}'."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg, details={"url": url, "reason": reason})


class ModelNotFoundError(Video2DocError):
    """Raised when a requested model is not pulled or available in Ollama."""

    def __init__(
        self,
        model_name: str,
        available_models: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.available_models = available_models or []
        msg = message or f"Model '{model_name}' not found on Ollama."
        if self.available_models:
            msg += f" Available models: {', '.join(self.available_models)}"
        super().__init__(
            msg,
            details={"model_name": model_name, "available_models": self.available_models},
        )


class MediaCorruptError(Video2DocError):
    """Raised when a video or audio file is corrupted, empty, or unreadable."""

    def __init__(
        self,
        media_path: str,
        message: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.media_path = media_path
        self.reason = reason
        msg = message or f"Media file is corrupt or unreadable: '{media_path}'."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg, details={"media_path": media_path, "reason": reason})


class TemplateNotFoundError(Video2DocError):
    """Raised when a specified documentation template is not found."""

    def __init__(
        self,
        template_name: str,
        searched_paths: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        self.template_name = template_name
        self.searched_paths = searched_paths or []
        msg = message or f"Template '{template_name}' not found."
        if self.searched_paths:
            msg += f" Searched paths: {', '.join(self.searched_paths)}"
        super().__init__(
            msg,
            details={"template_name": template_name, "searched_paths": self.searched_paths},
        )
