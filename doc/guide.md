# GemmaDesk Developer Guide

Welcome to the GemmaDesk project! This guide will help you set up and run the local Offline Multimodal RAG study tool.

## Architecture Overview
GemmaDesk is a **unified, local-first application**. Unlike traditional RAG pipelines that rely on cloud APIs or separate heavy servers, GemmaDesk runs entirely in a single process:

1. **Frontend / UI:** Built with Streamlit, handling file uploads and conversation history.
2. **Unified RAG Engine:** A single Python backend that coordinates:
   - **LiteRT (Gemma 4):** Our lightweight LLM loaded directly into the app memory.
   - **ChromaDB:** Local vector storage for documents, videos, and chat history.
   - **Multi-Engine Pipeline:** Dedicated handlers for PDF parsing, video clipping (FFmpeg), and image vision.

## 1. Setup & Installation

GemmaDesk manages almost all dependencies automatically via Python. Ensure you have `uv` installed, then install all project dependencies:

```bash
# 1. Create a virtual environment
uv venv

# 2. Activate the environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install dependencies
uv pip install -r requirements.txt
```

> [!NOTE]  
> **No System FFmpeg Required:** We use `imageio-ffmpeg`, which automatically downloads a portable FFmpeg binary into your virtual environment. You do NOT need to run `sudo apt install`.

## 2. Automated Model Setup

The first time you run GemmaDesk, it will detect if you are missing the required AI models (Gemma 4, Nomic Embeddings, and Whisper). 

1. Launch the application (see below).
2. The UI will redirect you to a **Setup Screen**.
3. Click **"Download Missing Models"**.
4. The app will automatically pull ~3.2GB of models from HuggingFace and cache them locally. 
5. Once finished, the app will auto-refresh into the Chat Interface.

## 3. Running the Application

There are two ways to run GemmaDesk:

### A. Desktop Mode (Recommended)
To run GemmaDesk as a standalone native desktop window:
```bash
python3 script/launcher.py
```

### B. Developer Mode
To run it in your default web browser:
```bash
uv run streamlit run app/app.py
```

## Troubleshooting
- **Slow First Load:** Loading the 2.5GB model into memory takes ~15-20 seconds on the first run of a session. Subsequent chats are near-instant.
- **GPU Acceleration:** By default, GemmaDesk tries to use your GPU for vision tasks. If it fails, it will automatically fallback to CPU mode (check terminal logs for "CPU fallback").
- **Missing Directory:** Ensure you have write permissions in the project root so the app can create the `model/`, `chroma_db/`, and `uploaded_media/` folders.
