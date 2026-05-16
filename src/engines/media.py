"""
media.py - Audio and Video Processing Engine

This module contains the MediaEngine class, which handles the transcription of
audio and video files using the Whisper model. It also provides utilities to 
dynamically extract audio and keyframe clips from video files.
"""
import os
import tempfile
import subprocess
import logging
import imageio_ffmpeg
from langchain_core.documents import Document
from rag import prompts

log = logging.getLogger("rag.media")

class MediaEngine:
    """
    Handles transcription of media files and dynamic clip extraction.
    """
    def __init__(self, whisper_model: str, doc_engine):
        """
        Initializes the MediaEngine.
        
        Args:
            whisper_model: The name of the Whisper model to load (e.g., 'base').
            doc_engine: The DocumentEngine instance used to chunk and index the resulting transcripts.
        """
        self.whisper_model = whisper_model
        self.doc_engine = doc_engine
        self._whisper = None
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def _get_whisper(self):
        if self._whisper is None:
            log.info("Loading Faster-Whisper model (%s)...", self.whisper_model)
            try:
                from faster_whisper import WhisperModel
                # Using CPU and int8 for ultra-light inference
                self._whisper = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
                log.info("Faster-Whisper loaded.")
            except Exception as e:
                log.error("Faster-Whisper load failed: %s", e)
                raise
        return self._whisper

    def _segments_to_docs(self, segments, source: str, kind: str) -> list:
        docs = []
        for seg in segments:
            ts = seg.start
            label = f"[{int(ts)//60:02d}:{int(ts)%60:02d}]"
            docs.append(Document(
                page_content=f"{label} {seg.text.strip()}",
                metadata={"source": source, "type": kind, "timestamp": ts},
            ))
        return docs

    def ingest_audio(self, path: str) -> int:
        """
        Transcribes an audio file and sends the segments to the DocumentEngine for indexing.
        
        Args:
            path: Absolute path to the audio file.
            
        Returns:
            int: The number of transcript chunks indexed.
        """
        log.info("Transcribing audio: %s", path)
        model = self._get_whisper()
        segments, info = model.transcribe(path, initial_prompt=prompts.WHISPER_INITIAL_PROMPT)
        docs = self._segments_to_docs(segments, path, "audio")
        log.info("Audio transcribed: %d segment(s).", len(docs))
        return self.doc_engine.index_docs(docs)

    def ingest_video(self, path: str) -> int:
        """
        Extracts the audio track from a video using ffmpeg, transcribes it, 
        and sends the segments to the DocumentEngine for indexing.
        
        Args:
            path: Absolute path to the video file.
            
        Returns:
            int: The number of transcript chunks indexed.
        """
        log.info("Extracting audio from video: %s", path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            subprocess.run(
                [self.ffmpeg_exe, "-i", path, "-ar", "16000", "-ac", "1", "-vn", tmp_wav, "-y"],
                check=True, capture_output=True,
            )
            log.info("ffmpeg audio extraction done.")
            model = self._get_whisper()
            segments, info = model.transcribe(tmp_wav, initial_prompt=prompts.WHISPER_INITIAL_PROMPT)
            docs = self._segments_to_docs(segments, path, "video")
            log.info("Video transcribed: %d segment(s).", len(docs))
            return self.doc_engine.index_docs(docs)
        except subprocess.CalledProcessError as e:
            log.error("ffmpeg failed: %s", e.stderr.decode())
            raise
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    def extract_clip(self, video_path: str, start_time: float, end_time: float) -> list:
        """
        Extracts an audio snippet and a single keyframe image from a specific 
        timestamp range within a video file using ffmpeg.
        
        Args:
            video_path: Absolute path to the source video.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            
        Returns:
            list: A list containing two absolute paths (one for the temporary 
                  audio snippet, one for the keyframe image).
        """
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
                [self.ffmpeg_exe, "-ss", str(start_time), "-t", str(duration), "-i", video_path, "-q:a", "0", "-map", "a", tmp_audio_path, "-y"],
                check=True, capture_output=True,
            )
            # Extract keyframe in the middle of the clip
            mid_time = start_time + (duration / 2)
            subprocess.run(
                [self.ffmpeg_exe, "-ss", str(mid_time), "-i", video_path, "-frames:v", "1", "-q:v", "2", tmp_img_path, "-y"],
                check=True, capture_output=True,
            )
            log.info("Successfully extracted audio snippet and keyframe.")
            return [tmp_audio_path, tmp_img_path]
        except subprocess.CalledProcessError as e:
            log.error("ffmpeg clip extraction failed: %s", e.stderr.decode())
            return []
