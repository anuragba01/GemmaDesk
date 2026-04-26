""

import os
import json
import base64
import tempfile
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import httpx
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

CHROMA_DIR     = "./chroma_db"
IMAGE_DIR      = "./uploaded_images"
IMAGE_MANIFEST = "./image_manifest.json"
EMBED_MODEL    = "nomic-embed-text"
CHAT_MODEL     = "gemma:2b"
WHISPER_MODEL  = "base"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50


class MultimodalRAG:
    ""

    def __init__(self):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="text_docs",
        )
        self._whisper = None  # lazy-loaded on first audio/video ingest
        self.image_paths = self._load_manifest()


    def _load_manifest(self) -> list:
        if os.path.exists(IMAGE_MANIFEST):
            with open(IMAGE_MANIFEST) as f:
                return json.load(f)
        return []

    def _save_manifest(self):
        with open(IMAGE_MANIFEST, "w") as f:
            json.dump(self.image_paths, f, indent=2)


    def _get_whisper(self):
        if self._whisper is None:
            import whisper
            self._whisper = whisper.load_model(WHISPER_MODEL)
        return self._whisper


    def _index(self, docs: list) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        splits = splitter.split_documents(docs)
        if splits:
            self.vectorstore.add_documents(splits)
        return len(splits)

    # INGEST

    def ingest_pdf(self, path: str) -> int:
        ""
        return self._index(PyPDFLoader(path).load())

    def ingest_text(self, path: str) -> int:
        ""
        return self._index(TextLoader(path, encoding="utf-8").load())

    def ingest_audio(self, path: str) -> int:
        ""
        model = self._get_whisper()
        result = model.transcribe(path, verbose=False)
        docs = self._segments_to_docs(result["segments"], path, "audio")
        return self._index(docs)

    def ingest_video(self, path: str) -> int:
        ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-i", path, "-ar", "16000", "-ac", "1", "-vn", tmp_wav, "-y"],
                check=True, capture_output=True,
            )
            model = self._get_whisper()
            result = model.transcribe(tmp_wav, verbose=False)
            docs = self._segments_to_docs(result["segments"], path, "video")
            return self._index(docs)
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    def ingest_image(self, src_path: str) -> bool:
        ""
        import shutil
        dest = os.path.join(IMAGE_DIR, os.path.basename(src_path))
        if dest in self.image_paths:
            return False
        shutil.copy2(src_path, dest)
        self.image_paths.append(dest)
        self._save_manifest()
        return True

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

    # QUERY

    def query_text(self, question: str) -> dict:
        ""
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(question)
        except Exception:
            docs = []

        context = "\n\n".join(d.page_content for d in docs) if docs else ""
        sources = list({d.metadata.get("source", "unknown") for d in docs}) if docs else []

        if docs:
            sys_prompt = (
                "You are a friendly study assistant. "
                "Try to answer the user's question using the context below. "
                "If the context doesn't contain the answer (e.g., they are just saying hi), just answer them naturally.\n\n"
                f"Context:\n{context}"
            )
        else:
            sys_prompt = "You are a friendly study assistant. Tell the user you don't have any documents indexed yet, but answer their question anyway."

        payload = {
            "model": "gemma-4",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ]
        }
        
        resp = httpx.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=120.0)
        resp.raise_for_status()
        
        return {
            "answer": resp.json()["choices"][0]["message"]["content"],
            "sources": [os.path.basename(s) for s in sources],
        }

    def query_image(self, question: str) -> dict:
        ""
        valid = [p for p in self.image_paths if os.path.exists(p)]
        if not valid:
            return {
                "answer": "No images indexed. Please upload some images first.",
                "sources": [],
            }

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": f"I am sharing {len(valid)} image(s) with you: {filenames}.\n\nQuestion: {question}"
                }
            ]
        }]
        
        for path in valid:
            messages[0]["content"].append({"type": "image", "path": path})

        payload = {
            "model": "gemma-4",
            "messages": messages
        }
        
        resp = httpx.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=120.0)
        resp.raise_for_status()
        
        return {
            "answer": resp.json()["choices"][0]["message"]["content"],
            "sources": [os.path.basename(p) for p in valid],
        }

    def analyze_frame(self, video_path: str, timestamp: float) -> str:
        ""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_frame = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-ss", str(timestamp), "-i", video_path, "-vframes", "1", tmp_frame, "-y"],
                check=True, capture_output=True,
            )
            with open(tmp_frame, "rb") as f:
                pass # Base64 no longer needed for local file path passing

            ts_label = f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}"
            
            payload = {
                "model": "gemma-4",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Describe this video frame from timestamp {ts_label}."},
                        {"type": "image", "path": tmp_frame}
                    ]
                }]
            }
            
            resp = httpx.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=120.0)
            resp.raise_for_status()
            
            return resp.json()["choices"][0]["message"]["content"]
        finally:
            if os.path.exists(tmp_frame):
                os.unlink(tmp_frame)

    # UTILITIES

    def get_stats(self) -> dict:
        try:
            count = self.vectorstore._collection.count()
        except Exception:
            count = 0
        return {"text_chunks": count, "images": len(self.image_paths)}

    def clear_all(self):
        ""
        import shutil
        for d in [CHROMA_DIR, IMAGE_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
        if os.path.exists(IMAGE_MANIFEST):
            os.unlink(IMAGE_MANIFEST)
        os.makedirs(IMAGE_DIR, exist_ok=True)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="text_docs",
        )
        self.image_paths = []
