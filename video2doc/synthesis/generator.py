"""Document synthesis engine formulating structured prompts and querying local LLMs via Ollama."""

from __future__ import annotations

import logging
from typing import Any

import jinja2
import requests

from video2doc.config import Settings, get_settings
from video2doc.exceptions import ModelNotFoundError, OllamaConnectionError

logger = logging.getLogger(__name__)


class DocumentSynthesizer:
    """Synthesizes structured documentation from multimodal transcript and vision inputs."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.text_model
        self.timeout = (
            timeout_seconds if timeout_seconds is not None else self.settings.timeout_seconds
        )
        self._jinja_env = jinja2.Environment(
            autoescape=False,
            undefined=jinja2.ChainableUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _format_transcript(self, transcript: Any) -> str:
        """Extract and format textual transcript from TranscriptionResult or raw string."""
        if transcript is None:
            return "No audio transcript available."

        if hasattr(transcript, "to_formatted_timestamp_text") and callable(
            transcript.to_formatted_timestamp_text
        ):
            formatted = transcript.to_formatted_timestamp_text().strip()
            if formatted:
                return formatted

        if hasattr(transcript, "full_text"):
            full = str(transcript.full_text).strip()
            if full:
                return full

        return str(transcript).strip() or "No audio transcript available."

    def _format_visuals(self, visual_descriptions: list[str | dict[str, Any]] | None) -> str:
        """Format list of visual descriptions into a readable bulleted summary."""
        if not visual_descriptions:
            return "No keyframe visual observations available."

        lines: list[str] = []
        for item in visual_descriptions:
            if isinstance(item, dict):
                ts = item.get("timestamp", 0.0)
                desc = item.get("description", "")
                m, s = divmod(int(ts), 60)
                lines.append(f"- [{m:02d}:{s:02d}] {desc}".strip())
            else:
                lines.append(f"- {item}".strip())

        return "\n".join(lines).strip() or "No keyframe visual observations available."

    def generate_document(
        self,
        transcript: Any,
        visual_descriptions: list[str | dict[str, Any]] | None = None,
        synthesis_prompt: str | None = None,
        template: Any = None,
        **kwargs: Any,
    ) -> str:
        """Synthesize multimodal video inputs into publication-grade Markdown documentation.

        Args:
            transcript: TranscriptionResult or raw transcript text.
            visual_descriptions: List of keyframe descriptions or metadata dicts.
            synthesis_prompt: Template synthesis instructions or prompt template.
            template: Optional TemplateDefinition instance.
            **kwargs: Extra template context variables (e.g. title).

        Returns:
            Generated Markdown documentation string.
        """
        transcript_text = self._format_transcript(transcript)
        visuals_text = self._format_visuals(visual_descriptions)

        raw_prompt = synthesis_prompt or (
            template.synthesis_prompt if template and hasattr(template, "synthesis_prompt") else ""
        ) or "Synthesize comprehensive procedural documentation based on the provided inputs."

        # Interpolate Jinja2 template placeholders if present
        if "{{" in raw_prompt or "{%" in raw_prompt:
            context: dict[str, Any] = {
                "transcript": transcript_text,
                "transcription_text": transcript_text,
                "full_text": getattr(transcript, "full_text", transcript_text),
                "visuals": visuals_text,
                "visual_descriptions": visuals_text,
                "visual_summary": visuals_text,
                "title": kwargs.get("title", "Video Documentation"),
                "sections": getattr(template, "sections", []) if template else [],
                "template": template,
            }
            context.update(kwargs)
            try:
                jinja_tmpl = self._jinja_env.from_string(raw_prompt)
                user_prompt = jinja_tmpl.render(**context)
            except Exception as err:
                logger.warning("Jinja2 rendering of synthesis prompt failed: %s; using raw prompt", err)
                user_prompt = f"{raw_prompt}\n\n## Transcript\n{transcript_text}\n\n## Visuals\n{visuals_text}"
        else:
            user_prompt = (
                f"{raw_prompt}\n\n"
                f"## Speech Transcript\n{transcript_text}\n\n"
                f"## Visual Keyframe Observations\n{visuals_text}"
            )

        url = f"{self.settings.ollama_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior technical writer and documentation engineer. "
                        "Produce comprehensive, publication-grade Markdown documentation. "
                        "Include clear section headers, concise overviews, step-by-step instructions, "
                        "and relevant technical details based strictly on the provided transcript and visual observations."
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": self.settings.model_temperature,
                "num_ctx": self.settings.model_num_ctx,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as err:
            raise OllamaConnectionError(
                self.settings.ollama_url,
                reason=f"Failed to connect to Ollama synthesis endpoint: {err}",
            ) from err

        if resp.status_code == 404:
            try:
                err_body = resp.json()
                err_msg = err_body.get("error", resp.text)
            except Exception:
                err_msg = resp.text
            raise ModelNotFoundError(
                self.model_name,
                message=f"Text synthesis model '{self.model_name}' not found on Ollama: {err_msg}",
            )

        if not resp.ok:
            raise OllamaConnectionError(
                self.settings.ollama_url,
                reason=f"HTTP {resp.status_code}: {resp.text}",
            )

        try:
            data = resp.json()
        except Exception as err:
            raise OllamaConnectionError(
                self.settings.ollama_url,
                reason=f"Ollama synthesis response is malformed JSON: {err}",
            ) from err

        if "error" in data and "not found" in str(data["error"]).lower():
            raise ModelNotFoundError(self.model_name, message=str(data["error"]))

        msg = data.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = ""

        if not content:
            content = data.get("response", "")

        return str(content).strip()
