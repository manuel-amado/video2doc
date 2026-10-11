"""Unit tests for VisionAnalyzer (multimodal keyframe analysis via Ollama)."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from video2doc.config import Settings
from video2doc.exceptions import (
    MediaCorruptError,
    ModelNotFoundError,
    OllamaConnectionError,
)
from video2doc.vision.analyzer import VisionAnalyzer
from video2doc.vision.keyframes import Keyframe


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a minimal valid JPEG image on disk."""
    img_path = tmp_path / "test_frame.jpg"
    img_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    )
    return img_path


@pytest.fixture
def empty_image(tmp_path: Path) -> Path:
    """Create a 0-byte corrupt image on disk."""
    empty_path = tmp_path / "corrupt_frame.jpg"
    empty_path.write_bytes(b"")
    return empty_path


class TestVisionAnalyzerInit:
    """Tests for VisionAnalyzer constructor and configuration defaults."""

    def test_default_initialization(self):
        analyzer = VisionAnalyzer()
        assert analyzer.model_name == "llama3.2-vision"
        assert analyzer.timeout == 300.0

    def test_custom_settings_initialization(self):
        settings = Settings(
            vision_model="llava:13b",
            timeout_seconds=45.0,
            ollama_url="http://remote-ollama:11434",
        )
        analyzer = VisionAnalyzer(settings=settings)
        assert analyzer.model_name == "llava:13b"
        assert analyzer.timeout == 45.0
        assert analyzer.settings.ollama_url == "http://remote-ollama:11434"

    def test_explicit_overrides_in_init(self):
        settings = Settings(vision_model="default-vision")
        analyzer = VisionAnalyzer(
            settings=settings,
            model_name="explicit-model",
            timeout_seconds=60.0,
        )
        assert analyzer.model_name == "explicit-model"
        assert analyzer.timeout == 60.0


class TestImageEncoding:
    """Tests for Base64 image encoding and corrupt media detection."""

    def test_encode_valid_image(self, sample_image: Path):
        analyzer = VisionAnalyzer()
        encoded = analyzer._encode_image(sample_image)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Should be valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == sample_image.read_bytes()

    def test_encode_missing_file_raises_media_corrupt(self, tmp_path: Path):
        analyzer = VisionAnalyzer()
        missing = tmp_path / "nonexistent.jpg"
        with pytest.raises(MediaCorruptError) as exc_info:
            analyzer._encode_image(missing)
        assert "not exist" in str(exc_info.value).lower()

    def test_encode_empty_file_raises_media_corrupt(self, empty_image: Path):
        analyzer = VisionAnalyzer()
        with pytest.raises(MediaCorruptError) as exc_info:
            analyzer._encode_image(empty_image)
        assert "empty" in str(exc_info.value).lower() or "no data" in str(exc_info.value).lower()


class TestAnalyzeFrame:
    """Tests for analyze_frame API requests and error handling."""

    def test_analyze_frame_with_keyframe_object(self, sample_image: Path):
        kf = Keyframe(timestamp=12.5, frame_path=sample_image, score=0.45)
        analyzer = VisionAnalyzer()

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Dashboard showing active analytics tab."}
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            desc = analyzer.analyze_frame(kf, prompt="Describe UI")
            assert desc == "Dashboard showing active analytics tab."
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert "messages" in call_kwargs["json"]
            assert call_kwargs["json"]["model"] == "llama3.2-vision"
            assert len(call_kwargs["json"]["messages"][0]["images"]) == 1

    def test_analyze_frame_with_path_and_str(self, sample_image: Path):
        analyzer = VisionAnalyzer()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Code editor showing Python script."}

        with patch("requests.post", return_value=mock_resp):
            desc_path = analyzer.analyze_frame(sample_image)
            assert desc_path == "Code editor showing Python script."

            desc_str = analyzer.analyze_frame(str(sample_image))
            assert desc_str == "Code editor showing Python script."

    def test_analyze_frame_connection_error_raises_ollama_connection_error(self, sample_image: Path):
        analyzer = VisionAnalyzer()
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            with pytest.raises(OllamaConnectionError) as exc_info:
                analyzer.analyze_frame(sample_image)
            assert "failed" in str(exc_info.value).lower() or "connection" in str(exc_info.value).lower()

    def test_analyze_frame_timeout_raises_ollama_connection_error(self, sample_image: Path):
        analyzer = VisionAnalyzer()
        with patch("requests.post", side_effect=requests.exceptions.Timeout("Request timed out")):
            with pytest.raises(OllamaConnectionError):
                analyzer.analyze_frame(sample_image)

    def test_analyze_frame_404_raises_model_not_found(self, sample_image: Path):
        analyzer = VisionAnalyzer(model_name="missing-vision-model")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"error": "model 'missing-vision-model' not found"}
        mock_resp.text = "model 'missing-vision-model' not found"

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ModelNotFoundError) as exc_info:
                analyzer.analyze_frame(sample_image)
            assert "missing-vision-model" in exc_info.value.model_name

    def test_analyze_frame_error_in_json_body_raises_model_not_found(self, sample_image: Path):
        analyzer = VisionAnalyzer(model_name="bad-tag")
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"error": "model bad-tag not found, please pull"}

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ModelNotFoundError):
                analyzer.analyze_frame(sample_image)

    def test_analyze_frame_server_500_raises_ollama_connection_error(self, sample_image: Path):
        analyzer = VisionAnalyzer()
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(OllamaConnectionError):
                analyzer.analyze_frame(sample_image)


class TestAnalyzeBatchAndKeyframes:
    """Tests for analyze_all and analyze_keyframes."""

    def test_analyze_all_multiple_keyframes(self, tmp_path: Path):
        kfs = []
        for i in range(3):
            p = tmp_path / f"frame_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
            kfs.append(Keyframe(timestamp=float(i * 5), frame_path=p))

        analyzer = VisionAnalyzer()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.side_effect = [
            {"message": {"content": "Step 1 UI"}},
            {"message": {"content": "Step 2 UI"}},
            {"message": {"content": "Step 3 UI"}},
        ]

        with patch("requests.post", return_value=mock_resp):
            results = analyzer.analyze_all(kfs, prompt="Describe step")
            assert results == ["Step 1 UI", "Step 2 UI", "Step 3 UI"]

    def test_analyze_all_empty_list(self):
        analyzer = VisionAnalyzer()
        assert analyzer.analyze_all([]) == []

    def test_analyze_keyframes_with_template_prompt(self, sample_image: Path):
        kf = Keyframe(timestamp=4.0, frame_path=sample_image, score=0.6)
        analyzer = VisionAnalyzer()

        class DummyTemplate:
            vision_prompt = "Identify buttons, navigation bars, and inputs."

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "Top navbar and search bar present."}}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            results = analyzer.analyze_keyframes([kf], template=DummyTemplate())
            assert len(results) == 1
            entry = results[0]
            assert entry["timestamp"] == 4.0
            assert entry["frame_path"] == sample_image
            assert entry["description"] == "Top navbar and search bar present."
            assert entry["score"] == 0.6

            call_payload = mock_post.call_args[1]["json"]
            assert call_payload["messages"][0]["content"] == "Identify buttons, navigation bars, and inputs."
