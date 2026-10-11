"""Video2Doc: Multimodal Video Documentation Framework."""

from video2doc.config import Settings, clear_settings_cache, get_settings
from video2doc.exceptions import (
    MediaCorruptError,
    MissingDependencyError,
    ModelNotFoundError,
    OllamaConnectionError,
    TemplateNotFoundError,
    Video2DocError,
)
from video2doc.pipeline import DocumentPipeline
from video2doc.synthesis import DocumentSynthesizer, MemoryManager
from video2doc.vision import VisionAnalyzer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Settings",
    "get_settings",
    "clear_settings_cache",
    "Video2DocError",
    "MissingDependencyError",
    "OllamaConnectionError",
    "ModelNotFoundError",
    "MediaCorruptError",
    "TemplateNotFoundError",
    "DocumentPipeline",
    "VisionAnalyzer",
    "DocumentSynthesizer",
    "MemoryManager",
]
