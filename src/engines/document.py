import os
import shutil
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag import prompts

log = logging.getLogger("rag.document")

class DocumentEngine:
    def __init__(self, chroma_dir: str, embed_model: str, chunk_size: int = 500, chunk_overlap: int = 50, gemma_engine=None):
        self.chroma_dir = chroma_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.gemma_engine = gemma_engine
        
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
        if not docs:
            return 0
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        splits = splitter.split_documents(docs)
        
        # Hardness Classification
        hardness = "UNKNOWN"
        if self.gemma_engine and splits:
            try:
                from rag import prompts
                # Sample from the middle to avoid title pages/TOCs
                mid_index = len(splits) // 2
                sample_text = splits[mid_index].page_content[:2000]
                prompt = prompts.HARDNESS_CLASSIFICATION_PROMPT + "\n\nText:\n" + sample_text
                hardness = self.gemma_engine.answer({"prompt_text": prompt}).strip().upper()
                if "HARD" in hardness: hardness = "HARD"
                elif "EASY" in hardness: hardness = "EASY"
                else: hardness = "MEDIUM"
                log.info(f"Classified material as {hardness}")
            except Exception as e:
                log.warning(f"Failed to classify hardness: {e}")
                
        for split in splits:
            split.metadata["hardness"] = hardness

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
