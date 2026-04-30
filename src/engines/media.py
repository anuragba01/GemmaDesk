import os
import tempfile
import subprocess
import logging
from langchain_core.documents import Document

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
        result = model.transcribe(path, verbose=False)
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
            result = model.transcribe(tmp_wav, verbose=False)
            docs = self._segments_to_docs(result["segments"], path, "video")
            log.info("Video transcribed: %d segment(s).", len(docs))
            return self.doc_engine.index_docs(docs)
        except subprocess.CalledProcessError as e:
            log.error("ffmpeg failed: %s", e.stderr.decode())
            raise
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
