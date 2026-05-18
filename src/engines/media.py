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
import re
from statistics import mean
import imageio_ffmpeg
from langchain_core.documents import Document

log = logging.getLogger("rag.media")
FFMPEG_TIMEOUT_SECONDS = 30

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
        self._active_processes = set()

    def kill_active_processes(self):
        """Forcefully kills any running ffmpeg tasks to free resources immediately."""
        count = 0
        for p in list(self._active_processes):
            try:
                p.kill()
                count += 1
            except Exception:
                pass
        self._active_processes.clear()
        if count > 0:
            log.warning("Killed %d orphaned ffmpeg processes.", count)

    def _run_ffmpeg(self, args: list, text: bool = False):
        process = None
        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=text,
            )
            self._active_processes.add(process)
            stdout, stderr = process.communicate(timeout=FFMPEG_TIMEOUT_SECONDS)
            return process, stdout, stderr
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
                stdout, stderr = process.communicate()
                log.error("ffmpeg timed out after %ss: %s", FFMPEG_TIMEOUT_SECONDS, " ".join(args))
                raise subprocess.CalledProcessError(-1, args, output=stdout, stderr=stderr)
            raise
        finally:
            if process is not None:
                self._active_processes.discard(process)

    def _get_whisper(self):
        if self._whisper is None:
            import sys
            if getattr(sys, "frozen", False):
                whisper_path = os.path.join(sys._MEIPASS, "model", "whisper-base")
            else:
                whisper_path = self.whisper_model

            log.info("Loading Faster-Whisper model (%s) from path=%s...", self.whisper_model, whisper_path)
            try:
                from faster_whisper import WhisperModel
                # Using CPU and int8 for ultra-light inference
                self._whisper = WhisperModel(whisper_path, device="cpu", compute_type="int8")
                log.info("Faster-Whisper loaded.")
            except Exception as e:
                log.error("Faster-Whisper load failed: %s", e)
                raise
        return self._whisper

    def get_media_duration(self, path: str) -> float:
        try:
            _, stdout, stderr = self._run_ffmpeg(
                [self.ffmpeg_exe, "-nostdin", "-i", path],
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as e:
            log.warning("ffmpeg duration probe failed for %s: %s", path, e)
            return 0.0

        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
        if not match:
            log.warning("Could not parse media duration for %s.", path)
            return 0.0

        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

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

    def _is_reliable_transcript(self, segments: list, info) -> bool:
        if not segments:
            return False

        texts = [" ".join(seg.text.lower().split()) for seg in segments if seg.text.strip()]
        if not texts:
            return False

        avg_no_speech = mean(seg.no_speech_prob for seg in segments)
        avg_logprob = mean(seg.avg_logprob for seg in segments)
        repeated_ratio = 1 - (len(set(texts)) / len(texts)) if len(texts) > 1 else 0.0
        words = " ".join(texts).split()
        unique_word_ratio = (len(set(words)) / len(words)) if words else 0.0
        speech_ratio = (info.duration_after_vad / info.duration) if info.duration else 0.0

        if avg_no_speech >= 0.7:
            return False
        if speech_ratio <= 0.1:
            return False
        if repeated_ratio >= 0.6 and unique_word_ratio <= 0.4:
            return False
        if avg_logprob <= -1.0 and avg_no_speech >= 0.5:
            return False
        return True

    def _transcribe(self, path: str):
        model = self._get_whisper()
        segments, info = model.transcribe(
            path,
            vad_filter=True,
            initial_prompt=None,
        )
        segments = list(segments)
        if not self._is_reliable_transcript(segments, info):
            log.warning(
                "Rejected unreliable transcript for %s (segments=%d, speech_ratio=%.2f).",
                path,
                len(segments),
                (info.duration_after_vad / info.duration) if info.duration else 0.0,
            )
            return [], info
        return segments, info

    def ingest_audio(self, path: str) -> int:
        """
        Transcribes an audio file and sends the segments to the DocumentEngine for indexing.
        
        Args:
            path: Absolute path to the audio file.
            
        Returns:
            int: The number of transcript chunks indexed.
        """
        log.info("Transcribing audio: %s", path)
        segments, _ = self._transcribe(path)
        if not segments:
            log.info("No reliable speech transcript detected for audio: %s", path)
            return 0
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
            process, stdout, stderr = self._run_ffmpeg(
                [self.ffmpeg_exe, "-nostdin", "-i", path, "-ar", "16000", "-ac", "1", "-vn", tmp_wav, "-y"]
            )
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args, output=stdout, stderr=stderr)
            log.info("ffmpeg audio extraction done.")
            segments, _ = self._transcribe(tmp_wav)
            if not segments:
                log.info("No reliable speech transcript detected for video: %s", path)
                return 0
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
            # litert_lm easily supports audio and images as attachments.
            # Extract audio snippet
            p1, out1, err1 = self._run_ffmpeg(
                [self.ffmpeg_exe, "-nostdin", "-ss", str(start_time), "-t", str(duration), "-i", video_path, "-q:a", "0", "-map", "a", tmp_audio_path, "-y"]
            )
            if p1.returncode != 0:
                raise subprocess.CalledProcessError(p1.returncode, p1.args, output=out1, stderr=err1)

            # Extract single keyframe image
            p2, out2, err2 = self._run_ffmpeg(
                [
                    self.ffmpeg_exe,
                    "-nostdin",
                    "-ss", str(start_time),
                    "-i", video_path,
                    "-t", str(duration),
                    "-q:v", "2",
                    "-vframes", "1",
                    "-an",
                    tmp_img_path,
                    "-y"
                ]
            )
            if p2.returncode != 0:
                raise subprocess.CalledProcessError(p2.returncode, p2.args, output=out2, stderr=err2)
            log.info("Successfully extracted audio snippet and keyframe.")
            return [tmp_audio_path, tmp_img_path]
        except subprocess.CalledProcessError as e:
            log.error("ffmpeg clip extraction failed: %s", e.stderr.decode())
            return []
