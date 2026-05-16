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

def test_get_video_duration(mock_vectorstore):
    # Mock the ChromaDB 'get' response to simulate multiple transcript chunks
    mock_vectorstore.vectorstore.get.return_value = {
        "metadatas": [
            {"timestamp": 12.5, "type": "video", "source": "vid.mp4"},
            {"timestamp": 41.2, "type": "video", "source": "vid.mp4"},
            {"timestamp": 3.0, "type": "video", "source": "vid.mp4"},
            # Some chunks might not have timestamps
            {"type": "video", "source": "vid.mp4"}
        ]
    }
    
    # It should correctly identify 41.2 as the max timestamp
    duration = mock_vectorstore.get_video_duration("vid.mp4")
    assert duration == 41.2
    
    # Verify the vectorstore was queried correctly
    mock_vectorstore.vectorstore.get.assert_called_once_with(
        where={"source": "vid.mp4"},
        limit=10000,
        include=["metadatas"]
    )

def test_get_video_duration_empty(mock_vectorstore):
    # If the database returns no chunks
    mock_vectorstore.vectorstore.get.return_value = {"metadatas": []}
    
    duration = mock_vectorstore.get_video_duration("vid.mp4")
    assert duration == 0.0

def test_get_video_duration_error(mock_vectorstore):
    # If the database throws an error
    mock_vectorstore.vectorstore.get.side_effect = Exception("DB Error")
    
    duration = mock_vectorstore.get_video_duration("vid.mp4")
    assert duration == 0.0
