import os
import tempfile
import subprocess
import logging
from langchain_core.documents import Document
from rag import prompts

log = logging.getLogger("rag.media")

class MediaEngine:
    def __init__(self, whisper_model: str, doc_engine):
        self.whisper_model = whisper_model
        self.doc_engine = doc_engine
        self._whisper = None

    def _get_whisper(self):
        if self._whisper is None:
            log.info("Loading Whisper model (%s)...", self.whisper_model)
            try:
                import whisper
                self._whisper = whisper.load_model(self.whisper_model)
                log.info("Whisper loaded.")
            except Exception as e:
                log.error("Whisper load failed: %s", e)
                raise
        return self._whisper

    def _segments_to_docs(self, segments: list, source: str, kind: str) -> list:
        docs = []
        for seg in segments:
            ts = seg.get("start", 0)
            label = f"[{int(ts)//60:02d}:{int(ts)%60:02d}]"
            docs.append(Document(
                page_content=f"{label} {seg['text'].strip()}",
                metadata={"source": source, "type": kind, "timestamp": ts},
            ))
        return docs

    def ingest_audio(self, path: str) -> int:
        log.info("Transcribing audio: %s", path)
        model = self._get_whisper()
        result = model.transcribe(path, verbose=False, initial_prompt=prompts.WHISPER_INITIAL_PROMPT)
        docs = self._segments_to_docs(result["segments"], path, "audio")
        log.info("Audio transcribed: %d segment(s).", len(docs))
        return self.doc_engine.index_docs(docs)

    def ingest_video(self, path: str) -> int:
        log.info("Extracting audio from video: %s", path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-i", path, "-ar", "16000", "-ac", "1", "-vn", tmp_wav, "-y"],
                check=True, capture_output=True,
            )
            log.info("ffmpeg audio extraction done.")
            model = self._get_whisper()
            result = model.transcribe(tmp_wav, verbose=False, initial_prompt=prompts.WHISPER_INITIAL_PROMPT)
            docs = self._segments_to_docs(result["segments"], path, "video")
            log.info("Video transcribed: %d segment(s).", len(docs))
            return self.doc_engine.index_docs(docs)
        except subprocess.CalledProcessError as e:
            log.error("ffmpeg failed: %s", e.stderr.decode())
            raise
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    def extract_clip(self, video_path: str, start_time: float, end_time: float) -> str:
        """Extracts a short video clip (or audio + keyframe if needed) using ffmpeg."""
        log.info(f"Extracting video clip from {video_path} ({start_time}s to {end_time}s)")
        
        # Ensure start_time is not negative
        start_time = max(0.0, start_time)
        duration = end_time - start_time
        
        # Create a temporary file for the extracted clip
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_audio_path = tmp_audio.name
            
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            tmp_img_path = tmp_img.name

        try:
            # We extract both audio and a single keyframe. 
            # litert_lm easily supports audio and images as attachments.
            # Extract audio snippet
            subprocess.run(
                ["ffmpeg", "-ss", str(start_time), "-t", str(duration), "-i", video_path, "-q:a", "0", "-map", "a", tmp_audio_path, "-y"],
                check=True, capture_output=True,
            )
            # Extract keyframe in the middle of the clip
            mid_time = start_time + (duration / 2)
            subprocess.run(
                ["ffmpeg", "-ss", str(mid_time), "-i", video_path, "-frames:v", "1", "-q:v", "2", tmp_img_path, "-y"],
                check=True, capture_output=True,
            )
            log.info("Successfully extracted audio snippet and keyframe.")
            return [tmp_audio_path, tmp_img_path]
        except subprocess.CalledProcessError as e:
            log.error("ffmpeg clip extraction failed: %s", e.stderr.decode())
            return []
