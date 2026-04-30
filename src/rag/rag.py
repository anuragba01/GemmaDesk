import os
import logging

from engines.document import DocumentEngine
from engines.gemma import GemmaEngine
from engines.vision import VisionEngine

log = logging.getLogger("rag.orchestrator")

CHROMA_DIR = "./chroma_db"
IMAGE_DIR = "./uploaded_images"
IMAGE_MANIFEST = "./image_manifest.json"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
MODEL_PATH = "./model/gemma-4-E4B-it.litertlm"


class MultimodalRAG:
    def __init__(self, doc_engine: DocumentEngine, vision_engine: VisionEngine, gemma_engine: GemmaEngine):
        self.doc_engine = doc_engine
        self.vision_engine = vision_engine
        self.gemma_engine = gemma_engine
        self.image_paths = self.vision_engine.get_valid_images()

    def _retrieve_text_docs(self, question: str, filter_paths: list = None, k: int = 4) -> list:
        retriever = self.doc_engine.get_retriever(filter_paths=filter_paths, k=k)
        return retriever.invoke(question)

    def _build_text_context(self, docs: list) -> str:
        parts = []
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            kind = doc.metadata.get("type", "text")
            parts.append(f"[From {kind} file: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, docs: list, image_paths: list) -> tuple[str, str]:
        context = self._build_text_context(docs)
        if docs or image_paths:
            system_text = (
                "You are a friendly study assistant. Use the retrieved context and attached files when they are relevant. "
                "If the retrieved material is not enough, answer naturally and say what is missing."
            )
        else:
            system_text = "You are a friendly study assistant. No indexed material matched, so answer naturally."

        prompt_parts = []
        if context:
            prompt_parts.append(f"Retrieved text context:\n{context}")
        if image_paths:
            filenames = ", ".join(os.path.basename(path) for path in image_paths)
            prompt_parts.append(f"Retrieved image files: {filenames}")
        prompt_parts.append(f"Question: {question}")
        return system_text, "\n\n".join(prompt_parts)

    def _source_names(self, docs: list, image_paths: list) -> list:
        sources = {os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs}
        sources.update(os.path.basename(path) for path in image_paths)
        return sorted(sources)

    def _answer(self, question: str, history: list, docs: list, image_paths: list) -> dict:
        system_text, prompt_text = self._build_prompt(question, docs, image_paths)
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

    def query_text(self, question: str, filter_paths: list = None, history: list = None) -> dict:
        log.info("query_text: '%s'", question[:80])
        try:
            docs = self._retrieve_text_docs(question, filter_paths=filter_paths)
        except Exception as e:
            log.warning("Text retrieval failed: %s", e)
            docs = []
        return self._answer(question, history, docs, [])

    def query_image(self, question: str, filter_paths: list = None, history: list = None) -> dict:
        log.info("query_image: '%s'", question[:80])
        image_paths = self.vision_engine.get_valid_images(filter_paths=filter_paths)
        if not image_paths:
            return {
                "answer": "Select at least one image to ask about.",
                "sources": [],
            }
        return self._answer(question, history, [], image_paths)

    def query_multimodal(self, question: str, filter_paths: list = None, history: list = None) -> dict:
        log.info("query_multimodal: '%s'", question[:80])
        try:
            docs = self._retrieve_text_docs(question, filter_paths=filter_paths)
        except Exception as e:
            log.warning("Text retrieval failed: %s", e)
            docs = []
        image_paths = self.vision_engine.get_valid_images(filter_paths=filter_paths)
        return self._answer(question, history, docs, image_paths)

    def get_stats(self) -> dict:
        self.image_paths = self.vision_engine.get_valid_images()
        return {
            "text_chunks": self.doc_engine.get_stats(),
            "images": self.vision_engine.get_stats(),
        }

    def get_source_map(self) -> dict:
        mapping = self.doc_engine.get_source_map()
        mapping.update(self.vision_engine.get_source_map())
        self.image_paths = self.vision_engine.get_valid_images()
        return mapping

    def clear_all(self):
        log.info("Clearing all indexed data...")
        self.doc_engine.clear()
        self.vision_engine.clear()
        self.image_paths = []
