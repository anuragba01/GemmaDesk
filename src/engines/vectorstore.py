"""
vectorstore.py - Vector Database Engine

This module contains the VectorStoreEngine class which manages all interactions 
with the local ChromaDB instance. It handles embedding generation, document 
indexing, semantic retrieval, and full-content bypass retrieval.
"""
import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from rag import prompts

log = logging.getLogger("rag.vectorstore")

class VectorStoreEngine:
    """
    Manages the ChromaDB vector database and HuggingFace local embeddings.
    """
    def __init__(self, chroma_dir: str, embed_model: str):
        """
        Initializes the embedding model and connects to the Chroma database.
        
        Args:
            chroma_dir: The local directory path where ChromaDB saves its SQLite files.
            embed_model: The HuggingFace model identifier (e.g., 'nomic-ai/nomic-embed-text-v1.5').
        """
        self.chroma_dir = chroma_dir
        
        log.info("Loading FastEmbed embeddings (%s)...", embed_model)
        try:
            self.embeddings = FastEmbedEmbeddings(
                model_name=embed_model,
                cache_dir=os.path.expanduser("~/.cache/fastembed_models"),
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
        """
        Embeds and indexes a list of text chunks into ChromaDB.
        
        Args:
            splits: A list of LangChain Document objects. Each Document must contain
                    `page_content` (the text) and `metadata` (source, hardness, etc.).
        """
        log.info("Indexing %d chunk(s) into ChromaDB...", len(splits))
        try:
            if splits:
                self.vectorstore.add_documents(splits)
        except Exception as e:
            log.error("ChromaDB add_documents failed: %s", e)
            raise

    def get_retriever(self, filter_paths: list = None, k: int = 4):
        """
        Creates a custom LangChain retriever for semantic search.
        
        Features:
        1. Applies metadata filtering if `filter_paths` are provided.
        2. Automatically prepends the required prefix ('search_query: ') to 
           user queries before calculating the embedding, which is a specific 
           requirement for Nomic embedding models to function optimally.
           
        Args:
            filter_paths: Optional list of file paths to restrict the search to.
            k: The number of chunks to retrieve (default is 4).
            
        Returns:
            A custom LangChain retriever object.
        """
        # We wrap the search to ensure Nomic's required prefix is added to the query
        search_kwargs = {"k": k}
        if filter_paths:
            if len(filter_paths) == 1:
                search_kwargs["filter"] = {"source": filter_paths[0]}
            else:
                search_kwargs["filter"] = {"source": {"$in": filter_paths}}
        
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # Override the invoke method to add the prefix (safely bypassing Pydantic)
        original_invoke = retriever.invoke
        def prefixed_invoke(query: str, **kwargs):
            return original_invoke(prompts.EMBED_QUERY_PREFIX + query, **kwargs)
        object.__setattr__(retriever, 'invoke', prefixed_invoke)
        
        return retriever

    def get_all_chunks(self, filter_paths: list = None, limit: int = 30) -> list:
        """Directly fetches up to `limit` chunks from ChromaDB for any source type
        (PDF, audio transcript, video transcript). Bypasses semantic search entirely."""
        from langchain_core.documents import Document
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

    def search_chat_history(self, query: str, session_id: str, k: int = 4) -> list:
        """
        Semantic search restricted to chat history blocks for a specific session.

        Prefilter: metadata["type"] == "chat" AND metadata["session_id"] == session_id
        This implements the RAG long-term chat memory (Addition 2a).

        Args:
            query:      The user's current question / message.
            session_id: UUID of the current chat session to scope the search.
            k:          Number of chat blocks to retrieve (default 4).

        Returns:
            A list of LangChain Document objects representing past chat blocks.
        """
        from langchain_core.documents import Document
        try:
            where_filter = {
                "$and": [
                    {"type": {"$eq": "chat"}},
                    {"session_id": {"$eq": session_id}},
                ]
            }
            results = self.vectorstore.similarity_search(
                prompts.EMBED_QUERY_PREFIX + query,
                k=k,
                filter=where_filter,
            )
            log.info(
                "Chat RAG: retrieved %d block(s) for session %s", len(results), session_id
            )
            return results
        except Exception as e:
            log.warning("search_chat_history failed: %s", e)
            return []

    def get_stats(self) -> int:
        """Returns the total number of text chunks currently indexed in ChromaDB."""
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
        """
        Deletes the entire ChromaDB collection, effectively wiping all indexed 
        text and transcript data, then reinitializes a fresh, empty database.
        """
        try:
            if hasattr(self, 'vectorstore') and self.vectorstore is not None:
                self.vectorstore.delete_collection()
        except Exception as e:
            log.warning("Failed to delete collection: %s", e)
        
        # We reinitialize to ensure a fresh collection is ready for new docs
        self._init_db()
