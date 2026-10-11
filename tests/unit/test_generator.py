"""Unit tests for DocumentSynthesizer (text documentation synthesis via Ollama)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from video2doc.audio.transcriber import TranscriptionResult, TranscriptSegment
from video2doc.config import Settings
from video2doc.exceptions import ModelNotFoundError, OllamaConnectionError
from video2doc.synthesis.generator import DocumentSynthesizer


@pytest.fixture
def sample_transcript() -> TranscriptionResult:
    """Fixture providing a standard TranscriptionResult with segments."""
    return TranscriptionResult(
        language="en",
        duration=45.0,
        segments=[
            TranscriptSegment(start=0.0, end=5.0, text="Welcome to the tutorial."),
            TranscriptSegment(start=5.0, end=20.0, text="Open your terminal and install dependencies."),
            TranscriptSegment(start=20.0, end=45.0, text="Run the main execution script to start."),
        ],
        full_text="Welcome to the tutorial. Open your terminal and install dependencies. Run the main execution script to start.",
    )


class TestDocumentSynthesizerInit:
    """Tests for constructor and configuration binding."""

    def test_default_initialization(self):
        synth = DocumentSynthesizer()
        assert synth.model_name == "qwen2.5-coder:14b"
        assert synth.timeout == 300.0

    def test_custom_settings(self):
        settings = Settings(
            text_model="llama3.1:8b",
            timeout_seconds=90.0,
            model_temperature=0.2,
            model_num_ctx=8192,
        )
        synth = DocumentSynthesizer(settings=settings)
        assert synth.model_name == "llama3.1:8b"
        assert synth.timeout == 90.0

    def test_explicit_overrides(self):
        synth = DocumentSynthesizer(model_name="custom-llm", timeout_seconds=120.0)
        assert synth.model_name == "custom-llm"
        assert synth.timeout == 120.0


class TestFormatHelpers:
    """Tests for transcript and visual descriptions formatting methods."""

    def test_format_transcript_from_transcription_result(self, sample_transcript: TranscriptionResult):
        synth = DocumentSynthesizer()
        formatted = synth._format_transcript(sample_transcript)
        assert "[00:00 -> 00:05] Welcome to the tutorial." in formatted
        assert "[00:05 -> 00:20] Open your terminal and install dependencies." in formatted

    def test_format_transcript_from_string(self):
        synth = DocumentSynthesizer()
        formatted = synth._format_transcript("Raw spoken words in a video.")
        assert formatted == "Raw spoken words in a video."

    def test_format_transcript_none_fallback(self):
        synth = DocumentSynthesizer()
        formatted = synth._format_transcript(None)
        assert "No audio transcript available" in formatted

    def test_format_visuals_dict_list(self):
        synth = DocumentSynthesizer()
        visuals = [
            {"timestamp": 12.0, "description": "Terminal window showing pip install."},
            {"timestamp": 75.0, "description": "Browser window showing localhost:8000."},
        ]
        formatted = synth._format_visuals(visuals)
        assert "- [00:12] Terminal window showing pip install." in formatted
        assert "- [01:15] Browser window showing localhost:8000." in formatted

    def test_format_visuals_str_list(self):
        synth = DocumentSynthesizer()
        visuals = ["Slide 1: Title", "Slide 2: Architecture diagram"]
        formatted = synth._format_visuals(visuals)
        assert "- Slide 1: Title" in formatted
        assert "- Slide 2: Architecture diagram" in formatted

    def test_format_visuals_empty_or_none(self):
        synth = DocumentSynthesizer()
        assert "No keyframe visual" in synth._format_visuals([])
        assert "No keyframe visual" in synth._format_visuals(None)


class TestGenerateDocument:
    """Tests for generate_document execution, prompt formulation, and error handling."""

    def test_generate_document_happy_path(self, sample_transcript: TranscriptionResult):
        synth = DocumentSynthesizer()
        visuals = ["Terminal displaying setup command"]

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_doc = "# Setup Guide\n\n## Instructions\n1. Run setup."
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": mock_doc}
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = synth.generate_document(
                transcript=sample_transcript,
                visual_descriptions=visuals,
                synthesis_prompt="Create SOP based on inputs",
            )
            assert result == mock_doc
            mock_post.assert_called_once()
            call_json = mock_post.call_args[1]["json"]
            assert call_json["model"] == "qwen2.5-coder:14b"
            assert "messages" in call_json
            assert any("Create SOP" in m["content"] for m in call_json["messages"])
            assert call_json["options"]["temperature"] == 0.1
            assert call_json["options"]["num_ctx"] == 32768

    def test_generate_document_jinja_interpolation(self, sample_transcript: TranscriptionResult):
        synth = DocumentSynthesizer()
        prompt_template = "# {{ title }}\n\nSummary:\n{{ transcript }}\n\nVisuals:\n{{ visuals }}"

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "# Custom Rendered"}}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            res = synth.generate_document(
                transcript=sample_transcript,
                visual_descriptions=["Screen 1"],
                synthesis_prompt=prompt_template,
                title="Interactive Guide",
            )
            assert res == "# Custom Rendered"
            user_msg = mock_post.call_args[1]["json"]["messages"][1]["content"]
            assert "# Interactive Guide" in user_msg
            assert "Welcome to the tutorial" in user_msg

    def test_generate_document_connection_error_raises_ollama_connection_error(self, sample_transcript: TranscriptionResult):
        synth = DocumentSynthesizer()
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Failed to reach Ollama")):
            with pytest.raises(OllamaConnectionError):
                synth.generate_document(sample_transcript, [], "Prompt")

    def test_generate_document_timeout_raises_ollama_connection_error(self, sample_transcript: TranscriptionResult):
        synth = DocumentSynthesizer()
        with patch("requests.post", side_effect=requests.exceptions.Timeout("Synthesis timed out")):
            with pytest.raises(OllamaConnectionError):
                synth.generate_document(sample_transcript, [], "Prompt")

    def test_generate_document_missing_model_404_raises_model_not_found(self, sample_transcript: TranscriptionResult):
        synth = DocumentSynthesizer(model_name="nonexistent-llm")
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"error": "model 'nonexistent-llm' not found"}
        mock_resp.text = "model 'nonexistent-llm' not found"

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ModelNotFoundError) as exc_info:
                synth.generate_document(sample_transcript, [], "Prompt")
            assert "nonexistent-llm" in exc_info.value.model_name

    def test_generate_document_server_500_raises_ollama_connection_error(self, sample_transcript: TranscriptionResult):
        synth = DocumentSynthesizer()
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(OllamaConnectionError):
                synth.generate_document(sample_transcript, [], "Prompt")
