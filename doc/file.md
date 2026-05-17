# GemmaDesk File Structure & Functions

This document summarizes the current role of the main files and directories in GemmaDesk.

## Core Application
- **`app/app.py`**: Main Streamlit entry point. Handles setup gating, profile collection, file uploads, source filters, chat UI, and streaming responses.
- **`app/setup.py`**: Preflight dependency and model setup screen. Checks for Gemma 4, BGE embeddings, and Whisper, then offers one-click downloads.
- **`app/__init__.py`**: Makes `app` importable as a package.

## Source Code (`src/`)

### RAG Orchestration
- **`src/rag/rag.py`**: Main orchestration layer. Handles retrieval, source filtering, transcript-first media QA, timestamp validation, summary bypass, chat-memory retrieval, and prompt assembly.
- **`src/rag/gemma.py`**: LiteRT integration layer. Loads the Gemma model locally and provides synchronous and streaming multimodal inference.
- **`src/rag/gateway.py`**: Intent gateway for summary/confusion routing using keyword-first checks with embedding fallback.
- **`src/rag/prompts.py`**: Central prompt and template definitions.
- **`src/rag/__init__.py`**: Package initializer.

### Processing Engines
- **`src/engines/document.py`**: Loads PDFs/TXT files, splits text into chunks, performs hardness classification, and forwards chunks for indexing.
- **`src/engines/media.py`**: Handles audio/video ingestion, exact media-duration probing, transcript reliability checks, and targeted support clip extraction through `imageio-ffmpeg`.
- **`src/engines/vectorstore.py`**: Wraps local ChromaDB access, semantic retrieval, full-content bypass retrieval, chat-memory retrieval, and source-map generation.
- **`src/engines/vision.py`**: Tracks uploaded images in a manifest and serves them as direct multimodal inputs.
- **`src/engines/chat_ingestion.py`**: Converts every completed 8-message chat block into long-term vector memory in ChromaDB.
- **`src/engines/__init__.py`**: Package initializer.

### Utilities
- **`src/utilities/chat_storage.py`**: Persists conversations as JSONL files and manages session metadata.
- **`src/utilities/profile.py`**: Stores and loads the local learner profile.
- **`src/utilities/__init__.py`**: Package initializer.

## Launch & Supporting Scripts
- **`script/launcher.py`**: Native desktop launcher using `pywebview`.
- **`script/download_model.py`**: Older standalone helper for model download workflows.

## Documentation
- **`doc/guide.md`**: Developer setup and run guide.
- **`doc/detailed_system_design.md`**: Current system architecture and design notes.
- **`doc/file.md`**: This file map.
- **`whiteups.txt`**: High-level product and engineering writeup.

## Data & Runtime Directories
- **`model/`**: Stores the LiteRT Gemma model artifact.
- **`chroma_db/`**: Local persistent ChromaDB storage.
- **`chat_sessions/`**: JSONL conversation history files.
- **`uploaded_media/`**: Local storage for uploaded PDFs, text files, audio, and videos.
- **`uploaded_images/`**: Local storage for indexed images.
- **`image_manifest.json`**: Registry for indexed images.
- **`user_profile.json`**: Local learner profile preferences.

## Dependency Definition
- **`requirements.txt`**: Python dependency list for Streamlit, LiteRT, ChromaDB, FastEmbed, Faster-Whisper, `imageio-ffmpeg`, and desktop launcher support.
