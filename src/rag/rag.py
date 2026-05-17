"""
rag.py - The Orchestrator for GemmaDesk

This module contains the MultimodalRAG class, which acts as the central brain
for the application. It coordinates the interactions between the Vector DB (Chroma),
the Multimodal LLM (Gemma), and the various data engines (Document, Media, Vision).
"""
import os
import logging
import re
from statistics import mean

from engines.document import DocumentEngine
from engines.vectorstore import VectorStoreEngine
from rag.gemma import GemmaEngine
from engines.vision import VisionEngine
from rag import prompts
from rag.gateway import IntentGateway

log = logging.getLogger("rag.orchestrator")

CHROMA_DIR = "./chroma_db"
IMAGE_DIR = "./uploaded_images"
IMAGE_MANIFEST = "./image_manifest.json"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_PATH = "./model/gemma-4-E4B-it.litertlm"

def extract_seconds(text: str) -> list[int]:
    """Parses timestamps like '1 min 50 sec', '1:50', '30s' from text."""
    pattern = r'(?:(\d+)\s*(?:min(?:ute)?s?|m))?\s*(?:and\s*)?(?:(\d+)\s*(?:sec(?:ond)?s?|s))?'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    times = []
    for match in matches:
        m, s = match.groups()
        if m or s:
            total = 0
            if m: total += int(m) * 60
            if s: total += int(s)
            times.append(total)
            
    pattern2 = r'(\d{1,2}):(\d{2})'
    for match in re.finditer(pattern2, text):
        m, s = match.groups()
        times.append(int(m) * 60 + int(s))
        
    return list(set(times))


class MultimodalRAG:
    """
    The central coordinator for the RAG pipeline.
    Handles intent detection, database retrieval, prompt assembly, and inference routing.
    """
    def __init__(self, doc_engine: DocumentEngine, vector_store: VectorStoreEngine, vision_engine: VisionEngine, media_engine, gemma_engine: GemmaEngine):
        """
        Initializes the Orchestrator with all necessary engines and the IntentGateway.
        
        Args:
            doc_engine: Handles text/PDF chunking and parsing.
            vector_store: Handles ChromaDB interactions (semantic search).
            vision_engine: Handles image tracking (manifest).
            media_engine: Handles audio/video transcription and clipping.
            gemma_engine: Handles LiteRT LLM inference.
        """
        self.doc_engine = doc_engine
        self.vector_store = vector_store
        self.vision_engine = vision_engine
        self.media_engine = media_engine
        self.gemma_engine = gemma_engine
        self.gateway = IntentGateway(self.vector_store.embeddings)

    def _build_media_durations(self, media_paths: list) -> dict:
        durations = {}
        for path in media_paths:
            dur = self.media_engine.get_media_duration(path)
            if dur > 0:
                durations[os.path.basename(path)] = dur
        return durations

    def _is_media_path(self, path: str) -> bool:
        return path.endswith((".mp4", ".mov", ".avi", ".mp3", ".wav"))

    def _audio_transcript_fallback(self, target_media: list, docs: list) -> str | None:
        audio_paths = [p for p in target_media if p.endswith((".mp3", ".wav"))]
        if not audio_paths:
            return None

        audio_docs = [doc for doc in docs if doc.metadata.get("source") in audio_paths]
        if not audio_docs:
            return (
                "I could not extract a reliable speech transcript from the selected audio. "
                "It may be music, noise, or otherwise non-speech content."
            )

        texts = []
        for doc in audio_docs:
            text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", doc.page_content).strip().lower()
            if text:
                texts.append(" ".join(text.split()))

        if not texts:
            return (
                "I could not extract a reliable speech transcript from the selected audio. "
                "It may be music, noise, or otherwise non-speech content."
            )

        repeated_ratio = 1 - (len(set(texts)) / len(texts)) if len(texts) > 1 else 0.0
        words = " ".join(texts).split()
        unique_word_ratio = (len(set(words)) / len(words)) if words else 0.0
        avg_chars = mean(len(text) for text in texts)
        if repeated_ratio >= 0.3 and unique_word_ratio <= 0.25 and avg_chars < 160:
            return (
                "The stored transcript for this audio does not look reliable. "
                "It may be music, noise, or a hallucinated transcription, so I cannot answer confidently from it."
            )
        return None

    def _detect_media_need(self, question: str) -> str:
        lowered = question.lower()
        visual_hits = any(token in lowered for token in (
            "show", "see", "look", "image", "frame", "screen", "picture", "visual"
        ))
        audio_hits = any(token in lowered for token in (
            "sound", "music", "voice", "hear", "audio", "listen"
        ))
        if visual_hits and audio_hits:
            return "multimodal"
        if visual_hits:
            return "visual"
        if audio_hits:
            return "audio"
        return "text_only"

    def _get_timestamp_context_docs(self, media_paths: list, explicit_times: list[int], limit: int = 200) -> list:
        if not media_paths or not explicit_times:
            return []

        try:
            docs = self.vector_store.get_all_chunks(filter_paths=media_paths, limit=limit)
        except Exception as e:
            log.warning("Timestamp context retrieval failed: %s", e)
            return []

        candidates = []
        for doc in docs:
            timestamp = doc.metadata.get("timestamp")
            source = doc.metadata.get("source")
            if source not in media_paths or timestamp is None:
                continue
            try:
                ts_value = float(timestamp)
            except (TypeError, ValueError):
                continue
            distance = min(abs(ts_value - requested) for requested in explicit_times)
            candidates.append((distance, ts_value, doc))

        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = []
        seen = set()
        for _, ts_value, doc in candidates:
            key = (doc.metadata.get("source"), ts_value, doc.page_content)
            if key in seen:
                continue
            seen.add(key)
            selected.append(doc)
            if len(selected) >= min(4, len(candidates)):
                break
        return selected

    def _merge_docs(self, primary_docs: list, fallback_docs: list) -> list:
        merged = []
        seen = set()
        for doc in primary_docs + fallback_docs:
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("timestamp"),
                doc.page_content,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged

    def _validate_explicit_timestamps(self, explicit_times: list[int], media_durations: dict, media_paths: list) -> str | None:
        if not explicit_times or not media_paths:
            return None

        for path in media_paths:
            duration = media_durations.get(os.path.basename(path), 0)
            if duration <= 0:
                continue
            for ts in explicit_times:
                if ts > duration:
                    return (
                        f"{os.path.basename(path)} is only {duration:.1f} seconds long, "
                        f"so {ts} seconds is out of range."
                    )
        return None

    def _retrieve_text_docs(self, question: str, filter_paths: list = None, k: int = 4) -> list:
        retriever = self.vector_store.get_retriever(filter_paths=filter_paths, k=k)
        return retriever.invoke(question)

    def _split_selected_paths(self, filter_paths: list = None) -> tuple[list, list]:
        """
        Splits UI-selected file paths into text-searchable paths (PDF/audio) 
        and raw visual paths (JPG/PNG).
        """
        valid_images = set(self.vision_engine.get_valid_images())
        image_paths = []
        text_paths = []
        for path in filter_paths or []:
            if path in valid_images:
                image_paths.append(path)
            else:
                text_paths.append(path)
        return text_paths, image_paths

    def _build_text_context(self, docs: list) -> str:
        parts = []
        for doc in docs:
            hardness = doc.metadata.get("hardness", "UNKNOWN")
            parts.append(prompts.CONTEXT_BLOCK_TEMPLATE.format(
                source=os.path.basename(doc.metadata.get("source", "unknown")),
                kind=doc.metadata.get("type", "text"),
                content=f"[Complexity: {hardness}]\n{doc.page_content}"
            ))
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, docs: list, image_paths: list, user_profile: dict = None, media_durations: dict = None, apply_confusion_gateway: bool = True) -> tuple[str, str]:
        """
        Assembles the master prompt sent to the LLM.
        Order of assembly: User Profile -> System Rules -> Intent Modifiers -> Images -> Context -> Question.
        """
        context = self._build_text_context(docs)
        system_text = prompts.CORE_SYSTEM_PROMPT
        
        if media_durations:
            system_text += "\n\n### MEDIA METADATA:\n"
            for v, d in media_durations.items():
                system_text += f"- Media '{v}' has a total duration of {int(d)} seconds.\n"
        
        if user_profile:
            profile_str = "\n### USER PROFILE:\n"
            if "language" in user_profile:
                profile_str += f"- Preferred Language: {user_profile['language']}. ALWAYS reply in this language.\n"
            if "education" in user_profile or "background" in user_profile:
                edu = user_profile.get("education", "")
                bg = user_profile.get("background", "")
                profile_str += f"- Background: {edu} in {bg}. Tailor your explanations to this level and context.\n"
            if "continent" in user_profile and user_profile["continent"] != "None":
                profile_str += f"- Location/Context: {user_profile['continent']}. Use culturally relevant examples when appropriate.\n"
            system_text = profile_str + "\n" + system_text

        # Gateway Check
        if apply_confusion_gateway and self.gateway.is_confused(question):
            system_text += f"\n\n{prompts.GATEWAY_CONFUSION_MODIFIER}"

        image_text = ""
        if image_paths:
            filenames = ", ".join(os.path.basename(p) for p in image_paths)
            image_text = prompts.IMAGE_LIST_TEMPLATE.format(filenames=filenames)

        prompt_text = prompts.QUERY_PROMPT_TEMPLATE.format(
            context_text=context if context else "No relevant text chunks found.",
            image_text=image_text,
            question=question
        )
        return system_text, prompt_text

    def _source_names(self, docs: list, image_paths: list) -> list:
        sources = {os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs}
        sources.update(
            os.path.basename(path)
            for path in image_paths
            if os.path.dirname(os.path.abspath(path)) != "/tmp"
        )
        return sorted(sources)

    def _answer(self, question: str, history: list, docs: list, image_paths: list, user_profile: dict = None, media_durations: dict = None, apply_confusion_gateway: bool = True) -> dict:
        system_text, prompt_text = self._build_prompt(
            question,
            docs,
            image_paths,
            user_profile,
            media_durations,
            apply_confusion_gateway,
        )
        answer = self.gemma_engine.answer({
            "system_text": system_text,
            "prompt_text": prompt_text,
            "history": history or [],
            "image_paths": image_paths,
        })
        return {
            "answer": answer,
            "sources": self._source_names(docs, image_paths),
        }

    def query_stream(self, question: str, filter_paths: list = None, history: list = None, user_profile: dict = None, fetch_full: bool = False, session_id: str = None) -> dict:
        """
        The main entry point for the Streamlit UI to ask questions.
        
        Process:
        1. Checks IntentGateway for "summary" requests to bypass semantic search.
        2. Retrieves text chunks from ChromaDB.
        3. (Addition 2a) Retrieves relevant past chat blocks from ChromaDB using semantic
           search, prefiltered by type=chat and current session_id.
        4. Dynamically extracts video clips via ffmpeg if a video chunk is found.
        5. Builds the final multimodal prompt.
        6. Streams the answer from the Gemma model.
        
        Returns:
            dict: Contains the generator 'stream' and the 'sources' list.
        """
        log.info("query_stream: '%s'", question[:80])
        text_paths, image_paths = self._split_selected_paths(filter_paths)
        explicit_times = extract_seconds(question)
        target_media = [p for p in text_paths if self._is_media_path(p)] if text_paths else []
        target_videos = [p for p in target_media if p.endswith((".mp4", ".mov", ".avi"))]
        media_need = self._detect_media_need(question)
        skip_gateway = bool(explicit_times and target_media)

        # Only trigger full-document fetch for text/PDF files.
        # For video/audio, transcripts are long and fetch_full floods the context → very slow.
        # Semantic search (k=4) works better and faster for media queries.
        if not skip_gateway and not fetch_full and not target_media and self.gateway.is_summary_request(question):
            log.info("Gateway detected summary intent. Auto-enabling fetch_full mode.")
            fetch_full = True

        docs = []
        if text_paths:
            try:
                if skip_gateway and target_media:
                    docs = self._get_timestamp_context_docs(target_media, explicit_times)
                elif fetch_full:
                    docs = self.vector_store.get_all_chunks(filter_paths=text_paths, limit=10)
                if not docs:
                    docs = self._retrieve_text_docs(question, filter_paths=text_paths)
            except Exception as e:
                log.warning("Text retrieval failed: %s", e)

        media_durations = self._build_media_durations(target_media)
        duration_error = self._validate_explicit_timestamps(explicit_times, media_durations, target_media)
        if duration_error:
            def duration_error_stream():
                yield duration_error
            return {
                "stream": duration_error_stream(),
                "sources": [os.path.basename(p) for p in target_media],
            }

        audio_fallback = self._audio_transcript_fallback(target_media, docs)
        if audio_fallback:
            def audio_fallback_stream():
                yield audio_fallback
            return {
                "stream": audio_fallback_stream(),
                "sources": [os.path.basename(p) for p in target_media if p.endswith((".mp3", ".wav"))],
            }
        clips_to_delete = []
        if explicit_times:
            if not target_media:
                def error_stream():
                    yield "Please select a specific media file from the sidebar to answer timestamp questions."
                return {"stream": error_stream(), "sources": []}
            for media_path in target_media:
                if not os.path.exists(media_path):
                    def missing_video_stream():
                        yield "Media file is removed from storage, please upload again."
                    return {"stream": missing_video_stream(), "sources": []}
            timestamp_docs = self._get_timestamp_context_docs(target_media, explicit_times)
            docs = self._merge_docs(timestamp_docs, docs)
            if media_need in {"visual", "audio", "multimodal"} and target_videos:
                for video in target_videos:
                    for ts in explicit_times:
                        clips = self.media_engine.extract_clip(video, ts - 5, ts + 10)
                        if clips:
                            image_paths.extend(clips)
                            clips_to_delete.extend(clips)
                            log.info("Extracted explicit support clips at %ss: %s", ts, clips)
                            break
                    if image_paths:
                        break

        # --- Long-term chat memory via RAG (Addition 2a) ---
        # Prefilter: type == "chat" AND session_id == current session
        # Then semantic search k=4 to find the most relevant past conversation blocks.
        if session_id:
            chat_docs = self.vector_store.search_chat_history(question, session_id, k=4)
            if chat_docs:
                log.info("Prepending %d chat history block(s) from RAG.", len(chat_docs))
                # Prepend so they appear before document chunks in the context
                docs = chat_docs + docs

        system_text, prompt_text = self._build_prompt(
            question,
            docs,
            image_paths,
            user_profile,
            media_durations,
            not skip_gateway,
        )
        stream = self.gemma_engine.answer_stream({
            "system_text": system_text,
            "prompt_text": prompt_text,
            "history": history or [],
            "image_paths": image_paths,
        })
        
        def _cleanup_stream(gen):
            try:
                yield from gen
            finally:
                for path in clips_to_delete:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                            log.info("Cleaned up temp clip: %s", path)
                        except Exception as e:
                            log.error("Failed to delete temp clip %s: %s", path, e)

        return {
            "stream": _cleanup_stream(stream),
            "sources": self._source_names(docs, image_paths),
        }

    def get_stats(self) -> dict:
        return {
            "text_chunks": self.vector_store.get_stats(),
            "images": self.vision_engine.get_stats(),
        }

    def get_source_map(self) -> dict:
        mapping = self.vector_store.get_source_map()
        mapping.update(self.vision_engine.get_source_map())
        return mapping

    def clear_all(self):
        log.info("Clearing all indexed data...")
        self.vector_store.clear()
        self.vision_engine.clear()
