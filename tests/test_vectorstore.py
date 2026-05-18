import pytest
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from engines.vectorstore import VectorStoreEngine

@pytest.fixture
def mock_vectorstore():
    # Bypass __init__ to prevent loading heavy ChromaDB and Embedding models
    engine = object.__new__(VectorStoreEngine)
    engine.vectorstore = MagicMock()
    return engine

def test_get_stats(mock_vectorstore):
    # Mock the count method inside the collection object
    mock_vectorstore.vectorstore._collection.count.return_value = 42
    
    stats = mock_vectorstore.get_stats()
    assert stats == 42
    mock_vectorstore.vectorstore._collection.count.assert_called_once()

def test_get_source_map(mock_vectorstore):
    # Mock the vectorstore get method to return list of metadatas
    mock_vectorstore.vectorstore.get.return_value = {
        "metadatas": [
            {"source": "/path/to/doc1.pdf"},
            {"source": "/path/to/doc2.pdf"},
            None, # Handles null metadata
            {"source": "/path/to/doc1.pdf"} # Handles duplicate sources
        ]
    }
    
    source_map = mock_vectorstore.get_source_map()
    assert source_map == {
        "doc1.pdf": "/path/to/doc1.pdf",
        "doc2.pdf": "/path/to/doc2.pdf"
    }
    mock_vectorstore.vectorstore.get.assert_called_once_with(include=["metadatas"])

def test_get_all_chunks(mock_vectorstore):
    # Mock database responses for chunks
    mock_vectorstore.vectorstore.get.return_value = {
        "documents": ["chunk 1", "chunk 2"],
        "metadatas": [{"source": "doc.pdf"}, {"source": "doc.pdf"}]
    }
    
    docs = mock_vectorstore.get_all_chunks(filter_paths=["doc.pdf"], limit=5)
    assert len(docs) == 2
    assert docs[0].page_content == "chunk 1"
    assert docs[0].metadata["source"] == "doc.pdf"
    mock_vectorstore.vectorstore.get.assert_called_once_with(
        where={"source": "doc.pdf"},
        limit=5,
        include=["documents", "metadatas"]
    )
