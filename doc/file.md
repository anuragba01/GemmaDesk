# GemmaDesk File Structure & Functions

This document describes the role of each file and directory in the GemmaDesk project.

## Core Application
- **`app/app.py`**: The main entry point. A Streamlit-based web interface that handles the UI, file uploads, and user chat interactions.
- **`app/setup.py`**: The **Dependency Manager**. Handles pre-flight checks and the automated model downloader UI.
- **`app/__init__.py`**: Makes the `app` folder a Python package.

## Source Code (`src/`)
The core logic is modularized into specialized engines and utilities.

### RAG Orchestration
- **`src/rag/rag.py`**: The orchestrator (Facade pattern). Routes tasks to specialized engines and manages the "Temporal Bypass" logic for video clipping.
- **`src/rag/gemma.py`**: The LiteRT integration engine. Loads the Gemma model directly into the app process.
- **`src/rag/__init__.py`**: Python package initializer.

### Processing Engines
- **`src/engines/document.py`**: Handles text and PDF files. Manages chunking and ChromaDB operations.
- **`src/engines/media.py`**: Handles audio and video. Uses Whisper for transcription and **portable `imageio-ffmpeg`** for frame extraction.
- **`src/engines/vision.py`**: Manages image file tracking and registry.
- **`src/engines/__init__.py`**: Python package initializer.

### Utilities
- **`src/utilities/chat_storage.py`**: Manages persistent chat sessions via JSON files.
- **`src/utilities/__init__.py`**: Python package initializer.

## Web & Deployment
- **`web/`**: Contains the premium, off-white landing page (`index.html`, `style.css`, `script.js`).
- **`script/launcher.py`**: The **Native Desktop Launcher**. Wraps the app in a `pywebview` window.
- **`script/download_model.py`**: Legacy CLI utility for downloading models.

## Testing & Docs
- **`tests/`**: Contains the **Pytest suite** (`test_rag.py`, `test_media.py`, `test_vectorstore.py`).
- **`doc/guide.md`**: The setup guide and running instructions.
- **`doc/detailed_system_design.md`**: In-depth architectural documentation.
- **`requirements.txt`**: List of all Python libraries, including portable FFmpeg.

## Data & Assets
- **`model/`**: Stores the heavy LiteRT Gemma model files.
- **`chroma_db/`**: Persistent vector database.
- **`chat_sessions/`**: Persistent chat history.
- **`uploaded_media/`**: Local storage for images and video transcripts.
