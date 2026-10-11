"""Unit tests for Video2Doc exceptions hierarchy."""

import pytest

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
    """Isolate unit tests from E2E integration fixtures."""
    yield


class TestExceptionsHierarchy:
    """Tests verifying behavior, inheritance, and attributes of custom exceptions."""

    def test_base_video2doc_error(self):
        err = Video2DocError("Something went wrong", details={"code": 500})
        assert isinstance(err, Exception)
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.details == {"code": 500}

    def test_missing_dependency_error_default(self):
        err = MissingDependencyError("ffmpeg")
        assert isinstance(err, Video2DocError)
        assert err.dependency_name == "ffmpeg"
        assert "Missing required dependency: 'ffmpeg'." in str(err)
        assert err.details["dependency"] == "ffmpeg"
        assert err.remediation is None

    def test_missing_dependency_error_with_remediation(self):
        remediation = "Install via 'winget install ffmpeg'."
        err = MissingDependencyError("ffmpeg", remediation=remediation)
        assert err.dependency_name == "ffmpeg"
        assert err.remediation == remediation
        assert remediation in str(err)
        assert err.details["remediation"] == remediation

    def test_ollama_connection_error_default(self):
        url = "http://localhost:11434"
        err = OllamaConnectionError(url)
        assert isinstance(err, Video2DocError)
        assert err.url == url
        assert f"Failed to connect to Ollama server at '{url}'." in str(err)
        assert err.details["url"] == url

    def test_ollama_connection_error_with_reason(self):
        url = "http://localhost:11434"
        reason = "Connection refused (10061)"
        err = OllamaConnectionError(url, reason=reason)
        assert err.url == url
        assert err.reason == reason
        assert reason in str(err)

    def test_model_not_found_error_default(self):
        model = "llama3.2-vision"
        err = ModelNotFoundError(model)
        assert isinstance(err, Video2DocError)
        assert err.model_name == model
        assert f"Model '{model}' not found on Ollama." in str(err)
        assert err.available_models == []

    def test_model_not_found_error_with_available_models(self):
        model = "llama3.2-vision"
        available = ["qwen2.5:14b", "llava:latest"]
        err = ModelNotFoundError(model, available_models=available)
        assert err.model_name == model
        assert err.available_models == available
        assert "Available models: qwen2.5:14b, llava:latest" in str(err)

    def test_media_corrupt_error_default(self):
        media = "damaged_video.mp4"
        err = MediaCorruptError(media)
        assert isinstance(err, Video2DocError)
        assert err.media_path == media
        assert f"Media file is corrupt or unreadable: '{media}'." in str(err)

    def test_media_corrupt_error_with_reason(self):
        media = "damaged_video.mp4"
        reason = "Invalid NAL unit header"
        err = MediaCorruptError(media, reason=reason)
        assert err.media_path == media
        assert err.reason == reason
        assert reason in str(err)

    def test_template_not_found_error_default(self):
        name = "nonexistent_template"
        err = TemplateNotFoundError(name)
        assert isinstance(err, Video2DocError)
        assert err.template_name == name
        assert f"Template '{name}' not found." in str(err)
        assert err.searched_paths == []

    def test_template_not_found_error_with_paths(self):
        name = "custom_lecture"
        paths = ["/templates/custom_lecture.yaml", "~/.video2doc/templates/custom_lecture.yaml"]
        err = TemplateNotFoundError(name, searched_paths=paths)
        assert err.template_name == name
        assert err.searched_paths == paths
        assert "Searched paths:" in str(err)

    def test_catch_all_subclasses_via_base_exception(self):
        exceptions = [
            MissingDependencyError("ffprobe"),
            OllamaConnectionError("http://127.0.0.1:11434"),
            ModelNotFoundError("missing-model"),
            MediaCorruptError("bad.mp4"),
            TemplateNotFoundError("missing_template"),
        ]
        for exc in exceptions:
            with pytest.raises(Video2DocError):
                raise exc

    def test_exception_chaining(self):
        try:
            try:
                raise ConnectionResetError("Socket closed")
            except ConnectionResetError as orig:
                raise OllamaConnectionError("http://localhost:11434", reason=str(orig)) from orig
        except OllamaConnectionError as caught:
            assert isinstance(caught.__cause__, ConnectionResetError)
            assert caught.reason == "Socket closed"
