# GemmaDesk File Structure & Functions

This document describes the role of each file and directory in the GemmaDesk project.

## Core Application
- **`app/app.py`**: The main entry point. A Streamlit-based web interface that handles the UI, file uploads, and user chat interactions.
- **`app/__init__.py`**: Makes the `app` folder a Python package.

## Source Code (`src/`)
The core logic is modularized into specialized engines and utilities.

### RAG Orchestration
- **`src/rag/rag.py`**: The "Brain" of the operation. It acts as an orchestrator (Facade pattern), routing tasks to the specific engines below and managing the final communication with the LiteRT AI server.
- **`src/rag/__init__.py`**: Python package initializer.

### Processing Engines
- **`src/engines/document.py`**: Handles text and PDF files. Manages chunking and all vector database (ChromaDB) operations for text data.
- **`src/engines/media.py`**: Handles audio and video. Uses Whisper for transcription and `ffmpeg` for extracting audio or video frames.
- **`src/engines/vision.py`**: Manages image file tracking and registry. (Future home for CLIP semantic search).
- **`src/engines/__init__.py`**: Python package initializer.

### Utilities
- **`src/utilities/litert_server.py`**: A FastAPI wrapper that runs the LiteRT model locally and provides an OpenAI-compatible API on port 8000.
- **`src/utilities/chat_storage.py`**: Manages persistent chat sessions, saving and loading conversation history as JSON files.
- **`src/utilities/__init__.py`**: Python package initializer.

## Documentation & Configuration
- **`doc/guide.md`**: The developer setup guide and running instructions.
- **`doc/h`**: Original architecture notes and project overview.
- **`requirements.txt`**: List of all Python libraries required to run the project.
- **`.gitignore`**: Tells Git which files/folders (like models and databases) to ignore.

## Data & Assets
- **`model/`**: Stores the heavy LiteRT Gemma model files.
- **`chroma_db/`**: The persistent vector database where all your document "memories" are stored.
- **`chat_sessions/`**: Stores your saved chat history.
- **`uploaded_images/`**: A local storage folder for images you have indexed.
- **`script/download_model.py`**: A utility script for downloading the required AI models.
