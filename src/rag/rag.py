"""
rag.py - The Orchestrator for GemmaDesk

This module contains the MultimodalRAG class, which acts as the central brain
for the application. It coordinates the interactions between the Vector DB (Chroma),
the Multimodal LLM (Gemma), and the various data engines (Document, Media, Vision).
"""
import os
import logging

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
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
MODEL_PATH = "./model/gemma-4-E4B-it.litertlm"


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
        self.image_paths = self.vision_engine.get_valid_images()

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

    def _build_prompt(self, question: str, docs: list, image_paths: list, user_profile: dict = None) -> tuple[str, str]:
        """
        Assembles the master prompt sent to the LLM.
        Order of assembly: User Profile -> System Rules -> Intent Modifiers -> Images -> Context -> Question.
        """
        context = self._build_text_context(docs)
        system_text = prompts.CORE_SYSTEM_PROMPT
        
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
        if self.gateway.is_confused(question):
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
        sources.update(os.path.basename(path) for path in image_paths)
        return sorted(sources)

    def _answer(self, question: str, history: list, docs: list, image_paths: list, user_profile: dict = None) -> dict:
        system_text, prompt_text = self._build_prompt(question, docs, image_paths, user_profile)
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

    def query(self, question: str, filter_paths: list = None, history: list = None, user_profile: dict = None) -> dict:
        log.info("query: '%s'", question[:80])
        text_paths, image_paths = self._split_selected_paths(filter_paths)

        docs = []
        text_filter = None if not filter_paths else text_paths
        if text_filter is None or text_filter:
            try:
                docs = self._retrieve_text_docs(question, filter_paths=text_filter)
            except Exception as e:
                log.warning("Text retrieval failed: %s", e)

        return self._answer(question, history, docs, image_paths, user_profile)

    def query_stream(self, question: str, filter_paths: list = None, history: list = None, user_profile: dict = None, fetch_full: bool = False) -> dict:
        """
        The main entry point for the Streamlit UI to ask questions.
        
        Process:
        1. Checks IntentGateway for "summary" requests to bypass semantic search.
        2. Retrieves text chunks from ChromaDB.
        3. Dynamically extracts video clips via ffmpeg if a video chunk is found.
        4. Builds the final multimodal prompt.
        5. Streams the answer from the Gemma model.
        
        Returns:
            dict: Contains the generator 'stream' and the 'sources' list.
        """
        log.info("query_stream: '%s'", question[:80])
        text_paths, image_paths = self._split_selected_paths(filter_paths)

        if not fetch_full and self.gateway.is_summary_request(question):
            log.info("Gateway detected summary intent. Auto-enabling fetch_full mode.")
            fetch_full = True

        docs = []
        text_filter = None if not filter_paths else text_paths
        
        if text_filter is None or text_filter:
            try:
                if fetch_full:
                    docs = self.vector_store.get_all_chunks(filter_paths=text_filter, limit=30)
                else:
                    docs = self._retrieve_text_docs(question, filter_paths=text_filter)
            except Exception as e:
                log.warning("Text retrieval failed: %s", e)

        # Video auto-clipping feature
        for doc in docs:
            if doc.metadata.get("type") == "video" and "timestamp" in doc.metadata:
                ts = float(doc.metadata["timestamp"])
                video_path = doc.metadata.get("source")
                if video_path and os.path.exists(video_path):
                    clips = self.media_engine.extract_clip(video_path, ts - 10, ts + 20)
                    if clips:
                        image_paths.extend(clips)
                        log.info(f"Appended extracted video clips to visual context: {clips}")
                    break # Only extract one clip per query to save compute

        system_text, prompt_text = self._build_prompt(question, docs, image_paths, user_profile)
        stream = self.gemma_engine.answer_stream({
            "system_text": system_text,
            "prompt_text": prompt_text,
            "history": history or [],
            "image_paths": image_paths,
        })
        
        return {
            "stream": stream,
            "sources": self._source_names(docs, image_paths),
        }

    def get_stats(self) -> dict:
        self.image_paths = self.vision_engine.get_valid_images()
        return {
            "text_chunks": self.vector_store.get_stats(),
            "images": self.vision_engine.get_stats(),
        }

    def get_source_map(self) -> dict:
        mapping = self.vector_store.get_source_map()
        mapping.update(self.vision_engine.get_source_map())
        self.image_paths = self.vision_engine.get_valid_images()
        return mapping

    def clear_all(self):
        log.info("Clearing all indexed data...")
        self.vector_store.clear()
        self.vision_engine.clear()
        self.image_paths = []
