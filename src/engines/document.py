"""
document.py - Document Processing Engine

This module contains the DocumentEngine class, which is responsible for loading 
raw text and PDF files, splitting them into manageable chunks, and using the 
Gemma model to classify the difficulty ("hardness") of the content before 
sending it to the VectorStore for indexing.
"""
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger("rag.document")

class DocumentEngine:
    """
    Handles parsing, chunking, and AI-assisted classification of text documents.
    """
    def __init__(self, vector_store, chunk_size: int = 500, chunk_overlap: int = 50, gemma_engine=None):
        """
        Initializes the DocumentEngine.
        
        Args:
            vector_store: The VectorStoreEngine instance to save chunks into.
            chunk_size: The maximum character length of a single text chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            gemma_engine: The GemmaEngine instance used for hardness classification.
        """
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.gemma_engine = gemma_engine

    def index_docs(self, docs: list) -> int:
        """
        Processes a list of raw LangChain Documents by splitting them, 
        evaluating their complexity, and saving them to the vector store.
        
        Args:
            docs: A list of loaded LangChain Document objects.
            
        Returns:
            int: The number of chunks successfully indexed.
        """
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

        self.vector_store.add_documents(splits)
        return len(splits)

    def ingest_pdf(self, path: str) -> int:
        """Loads a PDF file from the given path and triggers indexing."""
        log.info("Ingesting PDF: %s", path)
        return self.index_docs(PyPDFLoader(path).load())

    def ingest_text(self, path: str) -> int:
        """Loads a plain text file from the given path and triggers indexing."""
        log.info("Ingesting text: %s", path)
        return self.index_docs(TextLoader(path, encoding="utf-8").load())
