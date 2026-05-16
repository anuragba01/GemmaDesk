import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rag.rag import extract_seconds, MultimodalRAG

def test_extract_seconds():
    # Test cases that should return expected seconds
    assert set(extract_seconds("tell me about 15 sec")) == {15}
    assert set(extract_seconds("what happens at 1 min 50 sec")) == {110}
    assert set(extract_seconds("show me 30s")) == {30}
    assert set(extract_seconds("tell me about 1m 50s and 2 minutes 10 seconds")) == {110, 130}
    assert set(extract_seconds("look at 1:30")) == {90}
    assert set(extract_seconds("no time here")) == set()

@pytest.fixture
def mock_rag():
    # Create a completely mocked MultimodalRAG instance
    doc_engine = MagicMock()
    vector_store = MagicMock()
    vision_engine = MagicMock()
    media_engine = MagicMock()
    gemma_engine = MagicMock()
    
    # We must mock get_valid_images so initialization doesn't fail
    vision_engine.get_valid_images.return_value = []
    
    rag = MultimodalRAG(doc_engine, vector_store, vision_engine, media_engine, gemma_engine)
    return rag

def test_query_stream_explicit_time_no_video(mock_rag):
    # If the user asks for a time but has no videos selected
    result = mock_rag.query_stream(
        question="what is at 10 sec?",
        filter_paths=["/some/path/document.pdf"] # PDF selected, not a video
    )
    
    # Since it's an error stream, let's consume the generator to see the message
    stream = result["stream"]
    output = "".join(list(stream))
    assert "Please select a specific video" in output
    assert result["sources"] == []
    
    # The media engine should NOT have been called
    mock_rag.media_engine.extract_clip.assert_not_called()

def test_query_stream_explicit_time_with_video(mock_rag):
    # Setup mock video duration
    mock_rag.vector_store.get_video_duration.return_value = 60.0
    
    # Setup mock clip extraction
    mock_rag.media_engine.extract_clip.return_value = ["/tmp/clip_audio.wav", "/tmp/clip_img.jpg"]
    
    # Setup mock building prompt to avoid errors
    mock_rag._build_prompt = MagicMock(return_value=("sys", "prompt"))
    mock_rag.gemma_engine.answer_stream.return_value = (x for x in ["mock", "answer"])
    
    with patch("os.path.exists", return_value=True):
        result = mock_rag.query_stream(
            question="show me 20s",
            filter_paths=["/path/to/my_video.mp4"]
        )
    
    # The media engine SHOULD have been called specifically for timestamp 20
    mock_rag.media_engine.extract_clip.assert_called_once_with("/path/to/my_video.mp4", 15, 30)
    
    # The prompt builder should have been passed the duration
    mock_rag._build_prompt.assert_called_once()
    args, kwargs = mock_rag._build_prompt.call_args
    assert "my_video.mp4" in args[4] # video_durations dict is the 5th argument
    assert args[4]["my_video.mp4"] == 60.0
