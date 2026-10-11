"""Unit tests for MemoryManager (VRAM model unloading and CUDA cache management)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from video2doc.config import Settings
from video2doc.synthesis.memory import MemoryManager


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization and URL resolution."""

    def test_default_initialization(self):
        mgr = MemoryManager()
        assert mgr.ollama_url == "http://localhost:11434"

    def test_custom_settings(self):
        settings = Settings(ollama_url="http://remote-gpu:11434/")
        mgr = MemoryManager(settings=settings)
        assert mgr.ollama_url == "http://remote-gpu:11434"

    def test_explicit_url_override(self):
        mgr = MemoryManager(ollama_url="http://special-node:11434/")
        assert mgr.ollama_url == "http://special-node:11434"


class TestUnloadOllamaModel:
    """Tests for unload_ollama_model network signals and keep_alive payloads."""

    def test_unload_model_success(self):
        mgr = MemoryManager()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = mgr.unload_ollama_model("llama3.2-vision")
            assert result is True
            assert mock_post.call_count >= 1

            # Check that keep_alive: 0 was sent
            first_call_json = mock_post.call_args_list[0][1]["json"]
            assert first_call_json["model"] == "llama3.2-vision"
            assert first_call_json["keep_alive"] == 0

    def test_unload_model_connection_error_returns_false_safely(self):
        mgr = MemoryManager()
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            result = mgr.unload_ollama_model("qwen2.5-coder:14b")
            assert result is False

    def test_unload_model_custom_url_override(self):
        mgr = MemoryManager(ollama_url="http://default-host:11434")
        mock_resp = MagicMock()
        mock_resp.ok = True

        with patch("requests.post", return_value=mock_resp) as mock_post:
            mgr.unload_ollama_model("llava:latest", ollama_url="http://custom-host:11434")
            first_url = mock_post.call_args_list[0][0][0]
            assert "custom-host:11434" in first_url


class TestCudaCacheFlush:
    """Tests for clear_cuda_cache and PyTorch CUDA cache purging."""

    def test_clear_cuda_cache_when_cuda_available(self):
        mgr = MemoryManager()
        empty_cache_mock = MagicMock()
        ipc_collect_mock = MagicMock()

        with patch("gc.collect") as mock_gc, patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.empty_cache", empty_cache_mock), patch(
            "torch.cuda.ipc_collect", ipc_collect_mock, create=True
        ):
            mgr.clear_cuda_cache()
            mock_gc.assert_called_once()
            empty_cache_mock.assert_called_once()
            ipc_collect_mock.assert_called_once()

    def test_clear_cuda_cache_when_cuda_unavailable(self):
        mgr = MemoryManager()
        empty_cache_mock = MagicMock()

        with patch("gc.collect") as mock_gc, patch(
            "torch.cuda.is_available", return_value=False
        ), patch("torch.cuda.empty_cache", empty_cache_mock):
            mgr.clear_cuda_cache()
            mock_gc.assert_called_once()
            empty_cache_mock.assert_not_called()

    def test_flush_vram_invokes_clear_cuda_cache(self):
        mgr = MemoryManager()
        with patch.object(mgr, "clear_cuda_cache") as mock_clear:
            mgr.flush_vram()
            mock_clear.assert_called_once()


class TestWhisperAndStageCleanup:
    """Tests for unload_whisper, stage_cleanup, and cleanup_all."""

    def test_unload_whisper_with_model_instance(self):
        mgr = MemoryManager()
        mock_whisper = MagicMock()

        with patch.object(mgr, "clear_cuda_cache") as mock_clear:
            mgr.unload_whisper(mock_whisper)
            mock_whisper.unload.assert_called_once()
            mock_clear.assert_called_once()

    def test_unload_whisper_without_unload_method(self):
        mgr = MemoryManager()
        dummy_object = object()

        with patch.object(mgr, "clear_cuda_cache") as mock_clear:
            mgr.unload_whisper(dummy_object)
            mock_clear.assert_called_once()

    def test_stage_cleanup_with_model(self):
        mgr = MemoryManager()
        with patch.object(mgr, "unload_ollama_model") as mock_unload, patch.object(
            mgr, "clear_cuda_cache"
        ) as mock_clear:
            mgr.stage_cleanup("vision", model_name="llama3.2-vision")
            mock_unload.assert_called_once_with("llama3.2-vision")
            mock_clear.assert_called_once()

    def test_stage_cleanup_without_model(self):
        mgr = MemoryManager()
        with patch.object(mgr, "unload_ollama_model") as mock_unload, patch.object(
            mgr, "clear_cuda_cache"
        ) as mock_clear:
            mgr.stage_cleanup("audio")
            mock_unload.assert_not_called()
            mock_clear.assert_called_once()

    def test_cleanup_all_defaults(self):
        settings = Settings(vision_model="llama3.2-vision", text_model="qwen2.5-coder:14b")
        mgr = MemoryManager(settings=settings)

        with patch.object(mgr, "unload_ollama_model") as mock_unload, patch.object(
            mgr, "clear_cuda_cache"
        ) as mock_clear:
            mgr.cleanup_all()
            assert mock_unload.call_count == 2
            mock_unload.assert_any_call("llama3.2-vision")
            mock_unload.assert_any_call("qwen2.5-coder:14b")
            mock_clear.assert_called_once()

    def test_cleanup_all_explicit_models(self):
        mgr = MemoryManager()
        with patch.object(mgr, "unload_ollama_model") as mock_unload, patch.object(
            mgr, "clear_cuda_cache"
        ) as mock_clear:
            mgr.cleanup_all(models=["model_a", "model_b", "model_c"])
            assert mock_unload.call_count == 3
            mock_clear.assert_called_once()
