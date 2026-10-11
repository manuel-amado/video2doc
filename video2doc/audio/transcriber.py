"""Audio extraction and speech-to-text transcription pipeline using Whisper backends."""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from video2doc.config import Settings, get_settings
from video2doc.exceptions import MediaCorruptError, MissingDependencyError

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a discrete segment of transcribed speech with timing metadata."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Represents the complete result of an audio transcription operation."""

    language: str
    duration: float
    segments: list[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""

    def to_formatted_timestamp_text(self) -> str:
        """Format transcript segments into standard readable timestamped text blocks."""
        lines: list[str] = []
        for s in self.segments:
            m_s, s_s = divmod(int(s.start), 60)
            m_e, s_e = divmod(int(s.end), 60)
            lines.append(f"[{m_s:02d}:{s_s:02d} -> {m_e:02d}:{s_e:02d}] {s.text.strip()}")
        return "\n".join(lines)


class AudioTranscriber:
    """Speech-to-text transcriber supporting faster-whisper with openai-whisper fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.whisper_model
        raw_device = device or self.settings.whisper_device
        self.device = self._negotiate_device(raw_device)
        self._model: Any = None
        self._backend: str | None = None

    @staticmethod
    def _negotiate_device(requested_device: str) -> str:
        """Resolve execution device ('cuda', 'cpu', 'auto') using torch.cuda availability."""
        norm = requested_device.lower().strip()
        if norm == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except (ImportError, Exception):
                pass
            return "cpu"
        elif norm == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except (ImportError, Exception):
                pass
            return "cpu"
        return "cpu"

    def _ensure_model(self) -> Any:
        """Load and return the Whisper model instance using faster-whisper or openai-whisper."""
        if self._model is not None:
            return self._model

        # 1. Attempt faster-whisper backend
        try:
            import faster_whisper

            if faster_whisper is None or not hasattr(faster_whisper, "WhisperModel"):
                raise ImportError("faster-whisper is not available")

            compute_type = "float16" if self.device == "cuda" else "int8"
            try:
                self._model = faster_whisper.WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=compute_type,
                )
            except Exception:
                self._model = faster_whisper.WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type="default",
                )
            self._backend = "faster-whisper"
            logger.info(
                "Loaded faster-whisper backend with model '%s' on device '%s'",
                self.model_name,
                self.device,
            )
            return self._model
        except (ImportError, Exception):
            pass

        # 2. Fallback to openai-whisper backend
        try:
            import whisper

            if whisper is None or not hasattr(whisper, "load_model"):
                raise ImportError("openai-whisper is not available")

            self._model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            logger.info(
                "Loaded openai-whisper fallback with model '%s' on device '%s'",
                self.model_name,
                self.device,
            )
            return self._model
        except (ImportError, Exception) as err:
            raise MissingDependencyError(
                "whisper",
                message="Neither 'faster-whisper' nor 'openai-whisper' could be loaded.",
                remediation="Install faster-whisper (`pip install faster-whisper`) or openai-whisper (`pip install openai-whisper`).",
            ) from err

    def _extract_audio(self, media_path: Path, temp_wav_path: Path) -> None:
        """Extract a 16kHz mono PCM WAV audio stream from media container using FFmpeg."""
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise MissingDependencyError(
                "ffmpeg",
                message="FFmpeg executable was not found in system PATH.",
                remediation="Please install FFmpeg and ensure it is available in system PATH.",
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(temp_wav_path),
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.SubprocessError) as err:
            err_msg = (
                err.stderr.decode("utf-8", errors="replace").strip()
                if hasattr(err, "stderr") and isinstance(err.stderr, bytes)
                else str(err)
            )
            raise MediaCorruptError(
                str(media_path),
                reason=f"FFmpeg failed to extract audio track: {err_msg}",
            ) from err

    def transcribe(
        self,
        video_path: Path | str,
        language: str | None = None,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        """Transcribe speech from a media file into a TranscriptionResult.

        Args:
            video_path: Path to the input video or audio file.
            language: Optional ISO language code (e.g., 'en', 'es'). Auto-detects if None.
            vad_filter: Whether to apply Voice Activity Detection (VAD) on faster-whisper.

        Returns:
            TranscriptionResult containing language, duration, segments, and full text.
        """
        path = Path(video_path)
        if not path.exists():
            raise MediaCorruptError(str(path), reason="Media file does not exist.")
        if path.stat().st_size == 0:
            raise MediaCorruptError(str(path), reason="Media file is empty (0 bytes).")

        if not shutil.which("ffmpeg"):
            raise MissingDependencyError(
                "ffmpeg",
                message="FFmpeg executable was not found in system PATH.",
                remediation="Please install FFmpeg and ensure it is available in system PATH.",
            )

        model = self._ensure_model()
        target_lang = language or self.settings.whisper_language

        temp_wav: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav = Path(f.name)

            self._extract_audio(path, temp_wav)

            if self._backend == "faster-whisper":
                kwargs: dict[str, Any] = {"vad_filter": vad_filter}
                if target_lang:
                    kwargs["language"] = target_lang

                try:
                    segments_iter, info = model.transcribe(str(temp_wav), **kwargs)
                except Exception as err:
                    raise MediaCorruptError(
                        str(path),
                        reason=f"Whisper transcription failed: {err}",
                    ) from err

                segments: list[TranscriptSegment] = []
                full_text_parts: list[str] = []
                for s in segments_iter:
                    seg_text = s.text.strip()
                    segments.append(
                        TranscriptSegment(
                            start=float(s.start),
                            end=float(s.end),
                            text=seg_text,
                        )
                    )
                    if seg_text:
                        full_text_parts.append(seg_text)

                duration = getattr(info, "duration", 0.0)
                if duration <= 0.0 and segments:
                    duration = segments[-1].end
                detected_lang = getattr(info, "language", target_lang or "en")
                full_text = " ".join(full_text_parts).strip()

                return TranscriptionResult(
                    language=detected_lang,
                    duration=duration,
                    segments=segments,
                    full_text=full_text,
                )
            else:
                # openai-whisper fallback
                kwargs = {}
                if target_lang:
                    kwargs["language"] = target_lang

                try:
                    raw_result = model.transcribe(str(temp_wav), **kwargs)
                except Exception as err:
                    raise MediaCorruptError(
                        str(path),
                        reason=f"Whisper transcription failed: {err}",
                    ) from err

                segments = []
                full_text_parts = []
                raw_segments = raw_result.get("segments", []) if isinstance(raw_result, dict) else []
                for s in raw_segments:
                    start_val = float(s["start"] if isinstance(s, dict) else s.start)
                    end_val = float(s["end"] if isinstance(s, dict) else s.end)
                    txt_val = str(s["text"] if isinstance(s, dict) else s.text).strip()
                    segments.append(
                        TranscriptSegment(
                            start=start_val,
                            end=end_val,
                            text=txt_val,
                        )
                    )
                    if txt_val:
                        full_text_parts.append(txt_val)

                raw_text = raw_result.get("text", "") if isinstance(raw_result, dict) else ""
                full_text = raw_text.strip() or " ".join(full_text_parts).strip()
                detected_lang = (
                    raw_result.get("language", target_lang or "en")
                    if isinstance(raw_result, dict)
                    else (target_lang or "en")
                )
                duration = segments[-1].end if segments else 0.0

                return TranscriptionResult(
                    language=detected_lang,
                    duration=duration,
                    segments=segments,
                    full_text=full_text,
                )
        finally:
            if temp_wav is not None and temp_wav.exists():
                try:
                    temp_wav.unlink()
                except OSError:
                    pass

    def unload(self) -> None:
        """Unload model from RAM/VRAM and trigger garbage collection and CUDA cache clear."""
        if self._model is not None:
            del self._model
            self._model = None
            self._backend = None

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, Exception):
            pass
