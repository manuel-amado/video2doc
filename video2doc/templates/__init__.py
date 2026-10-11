"""Domain-agnostic template engine for Video2Doc."""

from __future__ import annotations

from video2doc.exceptions import TemplateNotFoundError
from video2doc.templates.engine import TemplateEngine
from video2doc.templates.loader import TemplateDefinition, TemplateLoader

__all__ = [
    "TemplateDefinition",
    "TemplateLoader",
    "TemplateEngine",
    "TemplateNotFoundError",
]
