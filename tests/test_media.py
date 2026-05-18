import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from engines.media import MediaEngine

@pytest.fixture
def mock_media_engine():
    # Mock the doc_engine
    doc_engine = MagicMock()
    # Mock imageio_ffmpeg.get_ffmpeg_exe to avoid actual path lookup during init if needed
    # but here we actually want to test if it finds it.
    engine = MediaEngine(whisper_model="base", doc_engine=doc_engine)
    return engine

def test_media_engine_init(mock_media_engine):
    # Verify that ffmpeg_exe is populated and points to a file that exists or is a string
    assert hasattr(mock_media_engine, "ffmpeg_exe")
    assert isinstance(mock_media_engine.ffmpeg_exe, str)
    assert len(mock_media_engine.ffmpeg_exe) > 0
    # In a real environment with imageio-ffmpeg installed, this should be a path
    assert "ffmpeg" in mock_media_engine.ffmpeg_exe.lower()

def test_extract_clip_calls_correct_exe(mock_media_engine):
    # Mock subprocess.Popen
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        # We need to mock tempfile to return stable paths for testing
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/fake.path"
            
            mock_media_engine.extract_clip("video.mp4", 10.0, 20.0)
            
            # Verify subprocess.Popen was called with the portable ffmpeg path, not the string "ffmpeg"
            assert mock_popen.call_count == 2
            first_call_args = mock_popen.call_args_list[0][0][0]
            assert first_call_args[0] == mock_media_engine.ffmpeg_exe
            assert first_call_args[0] != "ffmpeg"
