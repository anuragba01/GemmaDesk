import os
import shutil
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

log = logging.getLogger("rag.document")

class DocumentEngine:
    def __init__(self, chroma_dir: str, embed_model: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chroma_dir = chroma_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        log.info("Loading local embeddings (%s)...", embed_model)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embed_model,
                model_kwargs={'device': 'cpu'}, # Use CPU for embeddings to save GPU for Gemma
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

    def index_docs(self, docs: list) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        splits = splitter.split_documents(docs)
        log.info("Indexing %d chunk(s) into ChromaDB...", len(splits))
        try:
            if splits:
                self.vectorstore.add_documents(splits)
        except Exception as e:
            log.error("ChromaDB add_documents failed: %s", e)
            raise
        return len(splits)

    def ingest_pdf(self, path: str) -> int:
        log.info("Ingesting PDF: %s", path)
        return self.index_docs(PyPDFLoader(path).load())

    def ingest_text(self, path: str) -> int:
        log.info("Ingesting text: %s", path)
        return self.index_docs(TextLoader(path, encoding="utf-8").load())

    def get_retriever(self, filter_paths: list = None, k: int = 4):
        search_kwargs = {"k": k}
        if filter_paths:
            if len(filter_paths) == 1:
                search_kwargs["filter"] = {"source": filter_paths[0]}
            else:
                search_kwargs["filter"] = {"source": {"$in": filter_paths}}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

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
        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)
        self._init_db()
