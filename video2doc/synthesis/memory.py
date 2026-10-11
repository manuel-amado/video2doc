"""VRAM and memory management subsystem for staged model unloading and CUDA cache flushing."""

from __future__ import annotations

import gc
import logging
from typing import Any

import requests

from video2doc.config import Settings, get_settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates staged memory and VRAM management across video processing steps."""

    def __init__(
        self,
        settings: Settings | None = None,
        ollama_url: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ollama_url = (ollama_url or self.settings.ollama_url).rstrip("/")

    def unload_ollama_model(
        self,
        model_name: str,
        ollama_url: str | None = None,
    ) -> bool:
        """Explicitly instruct Ollama to evict model weights from VRAM by setting keep_alive to 0.

        Args:
            model_name: Identifier of the Ollama model to unload.
            ollama_url: Optional override URL for the Ollama server.

        Returns:
            True if unload request succeeded, False otherwise.
        """
        base_url = (ollama_url or self.ollama_url).rstrip("/")
        payload = {"model": model_name, "keep_alive": 0}
        success = False

        # Attempt unload via /api/generate
        generate_url = f"{base_url}/api/generate"
        try:
            resp = requests.post(generate_url, json=payload, timeout=10.0)
            if resp.ok:
                success = True
                logger.info("Successfully signaled unload for model '%s' via %s", model_name, generate_url)
        except requests.exceptions.RequestException as err:
            logger.warning("Unload request failed on %s for model '%s': %s", generate_url, model_name, err)

        # Attempt unload via /api/chat as well to ensure full eviction
        chat_url = f"{base_url}/api/chat"
        try:
            chat_payload = {"model": model_name, "messages": [], "keep_alive": 0}
            resp = requests.post(chat_url, json=chat_payload, timeout=10.0)
            if resp.ok:
                success = True
        except requests.exceptions.RequestException as err:
            logger.debug("Chat unload check on %s for model '%s': %s", chat_url, model_name, err)

        return success

    def clear_cuda_cache(self) -> None:
        """Trigger Python garbage collection and release cached PyTorch CUDA allocator blocks."""
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                logger.debug("Flushed PyTorch CUDA cache and collected IPC memory.")
        except (ImportError, Exception) as err:
            logger.debug("CUDA memory flush skipped: %s", err)

    def flush_vram(self) -> None:
        """Convenience alias for clear_cuda_cache."""
        self.clear_cuda_cache()

    def unload_whisper(self, whisper_model: Any = None) -> None:
        """Explicitly unload Whisper model instance and flush CUDA cache."""
        if whisper_model is not None:
            if hasattr(whisper_model, "unload") and callable(whisper_model.unload):
                try:
                    whisper_model.unload()
                except Exception:
                    pass
            del whisper_model

        self.clear_cuda_cache()

    def stage_cleanup(self, stage_name: str, model_name: str | None = None) -> None:
        """Execute staged memory cleanup between pipeline stages.

        Args:
            stage_name: Name of the pipeline stage completed (e.g. 'audio', 'vision', 'synthesis').
            model_name: Optional Ollama model name to unload from VRAM.
        """
        logger.info("Executing staged memory cleanup after stage: '%s'", stage_name)

        if model_name:
            self.unload_ollama_model(model_name)

        self.clear_cuda_cache()

    def cleanup_all(self, models: list[str] | None = None) -> None:
        """Safely unload all active pipeline models and flush CUDA VRAM."""
        target_models = models or [self.settings.vision_model, self.settings.text_model]
        for m in target_models:
            if m:
                self.unload_ollama_model(m)

        self.clear_cuda_cache()
