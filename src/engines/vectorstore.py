import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag import prompts

log = logging.getLogger("rag.vectorstore")

class VectorStoreEngine:
    def __init__(self, chroma_dir: str, embed_model: str):
        self.chroma_dir = chroma_dir
        
        log.info("Loading local embeddings (%s)...", embed_model)
        try:
            import torch
            _device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        except ImportError:
            _device = "cpu"
        log.info("Nomic embeddings using device: %s", _device)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embed_model,
                model_kwargs={'device': _device},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            log.error("Failed to load local embeddings: %s", e)
            raise

        log.info("Opening ChromaDB at %s...", chroma_dir)
        self._init_db()

    def _init_db(self):
        try:
            self.vectorstore = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings,
                collection_name="text_docs",
            )
        except Exception as e:
            log.error("ChromaDB init failed: %s", e)
            raise

    def add_documents(self, splits: list):
        log.info("Indexing %d chunk(s) into ChromaDB...", len(splits))
        try:
            if splits:
                self.vectorstore.add_documents(splits)
        except Exception as e:
            log.error("ChromaDB add_documents failed: %s", e)
            raise

    def get_retriever(self, filter_paths: list = None, k: int = 4):
        # We wrap the search to ensure Nomic's required prefix is added to the query
        search_kwargs = {"k": k}
        if filter_paths:
            if len(filter_paths) == 1:
                search_kwargs["filter"] = {"source": filter_paths[0]}
            else:
                search_kwargs["filter"] = {"source": {"$in": filter_paths}}
        
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # Override the invoke method to add the prefix
        original_invoke = retriever.invoke
        def prefixed_invoke(query: str, **kwargs):
            return original_invoke(prompts.EMBED_QUERY_PREFIX + query, **kwargs)
        retriever.invoke = prefixed_invoke
        
        return retriever

    def get_all_chunks(self, filter_paths: list = None, limit: int = 30) -> list:
        """Directly fetches up to `limit` chunks from ChromaDB for any source type
        (PDF, audio transcript, video transcript). Bypasses semantic search entirely."""
        from langchain.schema import Document
        try:
            if filter_paths:
                where = {"source": filter_paths[0]} if len(filter_paths) == 1 else {"source": {"$in": filter_paths}}
                data = self.vectorstore.get(where=where, limit=limit, include=["documents", "metadatas"])
            else:
                data = self.vectorstore.get(limit=limit, include=["documents", "metadatas"])

            docs = []
            if data and data.get("documents"):
                for i, text in enumerate(data["documents"]):
                    docs.append(Document(
                        page_content=text,
                        metadata=data["metadatas"][i] if data.get("metadatas") else {}
                    ))
            log.info("Full-content bypass: retrieved %d chunks from vectorstore.", len(docs))
            return docs
        except Exception as e:
            log.error("get_all_chunks failed: %s", e)
            return []

    def get_stats(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def get_source_map(self) -> dict:
        mapping = {}
        try:
            data = self.vectorstore.get(include=["metadatas"])
            if data and "metadatas" in data:
                for meta in data["metadatas"]:
                    if meta and "source" in meta:
                        mapping[os.path.basename(meta["source"])] = meta["source"]
        except Exception as e:
            log.error("Failed to get sources from ChromaDB: %s", e)
        return mapping

    def clear(self):
        try:
            if hasattr(self, 'vectorstore') and self.vectorstore is not None:
                self.vectorstore.delete_collection()
        except Exception as e:
            log.warning("Failed to delete collection: %s", e)
        
        # We reinitialize to ensure a fresh collection is ready for new docs
        self._init_db()
