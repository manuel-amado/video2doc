"""Orchestration pipeline executing the sequential multi-stage Video2Doc workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from video2doc.audio.transcriber import AudioTranscriber
from video2doc.config import Settings, get_settings
from video2doc.exceptions import (
    MediaCorruptError,
    ModelNotFoundError,
    OllamaConnectionError,
    Video2DocError,
)
from video2doc.synthesis.generator import DocumentSynthesizer
from video2doc.synthesis.memory import MemoryManager
from video2doc.templates.engine import TemplateEngine
from video2doc.vision.analyzer import VisionAnalyzer
from video2doc.vision.keyframes import KeyframeExtractor

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


class DocumentPipeline:
    """Orchestrates audio transcription, keyframe extraction, vision analysis,

    document synthesis, and template rendering into a unified workflow.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.audio_transcriber = AudioTranscriber(settings=self.settings)
        self.keyframe_extractor = KeyframeExtractor(settings=self.settings)
        self.vision_analyzer = VisionAnalyzer(settings=self.settings)
        self.synthesizer = DocumentSynthesizer(settings=self.settings)
        self.memory_manager = MemoryManager(settings=self.settings)
        self.template_engine = TemplateEngine()

    def check_ollama_health(self) -> bool:
        """Verify that the Ollama server is responsive and healthy."""
        url = f"{self.settings.ollama_url.rstrip('/')}/api/tags"
        try:
            resp = requests.get(url, timeout=10.0)
            if not resp.ok:
                raise OllamaConnectionError(
                    self.settings.ollama_url,
                    reason=f"Ollama server responded with HTTP {resp.status_code}",
                )
            return True
        except requests.exceptions.RequestException as err:
            raise OllamaConnectionError(
                self.settings.ollama_url,
                reason=f"Failed to connect to Ollama server: {err}",
            ) from err

    def validate_models_available(self) -> bool:
        """Verify that the required vision and text models exist in the Ollama instance."""
        url = f"{self.settings.ollama_url.rstrip('/')}/api/tags"
        try:
            resp = requests.get(url, timeout=10.0)
            if not resp.ok:
                raise OllamaConnectionError(
                    self.settings.ollama_url,
                    reason=f"Ollama tags endpoint returned HTTP {resp.status_code}",
                )
            data = resp.json()
        except requests.exceptions.RequestException as err:
            raise OllamaConnectionError(
                self.settings.ollama_url,
                reason=f"Failed to fetch model tags from Ollama: {err}",
            ) from err
        except Exception as err:
            raise OllamaConnectionError(
                self.settings.ollama_url,
                reason=f"Malformed response from Ollama tags: {err}",
            ) from err

        models_list = data.get("models", [])
        available_names: list[str] = [
            m.get("name", "") for m in models_list if isinstance(m, dict) and "name" in m
        ]

        def _is_model_present(required_model: str) -> bool:
            req_clean = required_model.lower().strip()
            for av in available_names:
                av_clean = av.lower().strip()
                if (
                    av_clean == req_clean
                    or av_clean.split(":")[0] == req_clean.split(":")[0]
                    or av_clean.startswith(req_clean)
                    or req_clean.startswith(av_clean)
                ):
                    return True
            return False

        if not _is_model_present(self.settings.vision_model):
            raise ModelNotFoundError(
                self.settings.vision_model,
                available_models=available_names,
                message=(
                    f"Required vision model '{self.settings.vision_model}' is not pulled in Ollama. "
                    f"Run `ollama pull {self.settings.vision_model}`."
                ),
            )

        if not _is_model_present(self.settings.text_model):
            raise ModelNotFoundError(
                self.settings.text_model,
                available_models=available_names,
                message=(
                    f"Required text model '{self.settings.text_model}' is not pulled in Ollama. "
                    f"Run `ollama pull {self.settings.text_model}`."
                ),
            )

        return True

    def process_video(
        self,
        video_path: Path | str,
        template: str = "sop",
        output_dir: Path | str | None = None,
    ) -> Path:
        """Process a single video through all 5 sequential stages.

        Args:
            video_path: Path to the input video file.
            template: Identifier or filepath for documentation template (default: 'sop').
            output_dir: Destination directory for Markdown and extracted assets.

        Returns:
            Path to the generated Markdown document file.
        """
        video = Path(video_path).resolve()
        if not video.exists():
            raise MediaCorruptError(str(video), reason="Video file does not exist.")
        if video.stat().st_size == 0:
            raise MediaCorruptError(str(video), reason="Video file is empty (0 bytes).")

        out_path = Path(output_dir) if output_dir is not None else self.settings.output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        template_def = self.template_engine.load_template(template)

        # Stage 1: Audio Transcription
        logger.info("Executing Stage 1: Audio ASR Transcription for %s", video.name)
        transcript = self.audio_transcriber.transcribe(video)
        self.audio_transcriber.unload()
        self.memory_manager.stage_cleanup("audio")

        # Stage 2: Smart Keyframe Extraction
        logger.info("Executing Stage 2: Smart Keyframe Extraction for %s", video.name)
        frames_dir = (
            out_path / f"{video.stem}_frames"
            if self.settings.keep_frames
            else out_path / "frames"
        )
        keyframes = self.keyframe_extractor.extract(
            video,
            frames_dir,
            max_frames=self.settings.max_keyframes,
            threshold=self.settings.scene_threshold,
            debounce_seconds=self.settings.debounce_seconds,
        )

        # Stage 3: Vision Analysis
        logger.info("Executing Stage 3: Vision Analysis on %d keyframes", len(keyframes))
        visual_descriptions = self.vision_analyzer.analyze_keyframes(keyframes, template_def)
        self.memory_manager.stage_cleanup("vision", self.settings.vision_model)

        # Stage 4: Text Synthesis
        logger.info("Executing Stage 4: Text Documentation Synthesis")
        title = video.stem.replace("_", " ").replace("-", " ").title()
        synthesized_text = self.synthesizer.generate_document(
            transcript=transcript,
            visual_descriptions=visual_descriptions,
            synthesis_prompt=template_def.synthesis_prompt,
            template=template_def,
            title=title,
        )
        self.memory_manager.stage_cleanup("synthesis", self.settings.text_model)

        # Stage 5: Template Engine Rendering
        logger.info("Executing Stage 5: Template Engine Rendering")
        transcription_formatted = (
            transcript.to_formatted_timestamp_text()
            if hasattr(transcript, "to_formatted_timestamp_text")
            else getattr(transcript, "full_text", str(transcript))
        )
        visual_summary_lines = [
            v.get("description", "") if isinstance(v, dict) else str(v)
            for v in visual_descriptions
        ]
        visual_summary_str = "\n".join(visual_summary_lines)

        context: dict[str, Any] = {
            "title": title,
            "video_name": video.name,
            "video_stem": video.stem,
            "duration": getattr(transcript, "duration", 0.0),
            "language": getattr(transcript, "language", "en"),
            "transcript": transcript,
            "transcription_text": transcription_formatted,
            "full_text": getattr(transcript, "full_text", transcription_formatted),
            "segments": getattr(transcript, "segments", []),
            "keyframes": keyframes,
            "visual_descriptions": visual_descriptions,
            "visual_summary": visual_summary_str,
            "synthesized_content": synthesized_text,
            "synthesized_text": synthesized_text,
            "template": template_def,
            "sections": getattr(template_def, "sections", []),
        }

        template_body = (
            template_def.layout_template.strip()
            if template_def.layout_template and template_def.layout_template.strip()
            else (
                template_def.synthesis_prompt.strip()
                if ("{{" in template_def.synthesis_prompt or "{%" in template_def.synthesis_prompt)
                else ""
            )
        )
        if template_body:
            final_markdown = self.template_engine.render_string(
                template_body, context
            )
        else:
            final_markdown = synthesized_text

        dest_doc = out_path / f"{video.stem}.md"
        dest_doc.write_text(final_markdown, encoding="utf-8")
        logger.info("Generated documentation successfully written to %s", dest_doc)

        return dest_doc

    def process_batch(
        self,
        input_dir: Path | str,
        template: str = "sop",
        output_dir: Path | str | None = None,
    ) -> list[Path]:
        """Batch process all video files in a directory tree with error isolation.

        Args:
            input_dir: Source folder containing video files.
            template: Template identifier or filepath.
            output_dir: Destination directory for Markdown files.

        Returns:
            List of generated Markdown document paths.
        """
        in_path = Path(input_dir).resolve()
        if not in_path.exists() or not in_path.is_dir():
            raise Video2DocError(
                f"Input batch directory does not exist or is not a directory: '{in_path}'."
            )

        video_files = sorted([
            p
            for p in in_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ])

        if not video_files:
            logger.warning("No supported video files found in '%s'", in_path)
            return []

        logger.info("Found %d video file(s) for batch processing in '%s'", len(video_files), in_path)

        generated_paths: list[Path] = []
        for vpath in video_files:
            try:
                doc_path = self.process_video(
                    vpath,
                    template=template,
                    output_dir=output_dir,
                )
                generated_paths.append(doc_path)
            except Exception as err:
                logger.error(
                    "Error processing video file '%s': %s. Continuing with remaining batch.",
                    vpath,
                    err,
                    exc_info=True,
                )

        return generated_paths
