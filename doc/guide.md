# GemmaDesk Developer Guide

Welcome to the GemmaDesk project! This guide will help you set up and run the local Offline Multimodal RAG study tool.

## Architecture Overview
This application is split into two main components:
1. **Frontend / RAG Engine (`app/app.py` & `src/rag/rag.py`):** A Streamlit interface that handles file uploads, chunking, and ChromaDB vector storage.
2. **Backend AI Engine (`src/utilities/litert_server.py`):** A local FastAPI server that wraps the lightweight `gemma-4-E4B-it.litertlm` (LiteRT) model and serves an OpenAI-compatible `/v1/chat/completions` endpoint.

## 1. Setup & Installation

Ensure you have `uv` installed, then install all project dependencies:

```bash
uv pip install -r requirements.txt
```

## 2. Download the Model

The application requires the 4-bit compressed LiteRT Gemma model (~3.6 GB). Because Gemma is gated by Google, you must provide a HuggingFace Token.

1. Go to HuggingFace, accept the Gemma 4 terms.
2. Get your Access Token from your HuggingFace settings.
3. Run the downloader script:

```bash
HF_TOKEN="hf_YOUR_TOKEN_HERE" uv run python download_model.py
```

*This will download `gemma-4-E4B-it.litertlm` into the root directory.*

## 3. Running the Application

GemmaDesk now runs entirely in a **single terminal**. The AI model is loaded directly into the Streamlit process.

### Start the Application

```bash
uv run streamlit run app/app.py
```

> [!NOTE]  
> The first time you run this, or whenever you make a code change, it will take ~20-30 seconds to load the 3.6GB model into memory. Once loaded, the chat responses will be instant.

## Troubleshooting
- **Model Not Found Error:** Ensure the `.litertlm` file downloaded successfully and is in the root directory.
- **Connection Refused:** Ensure Terminal 1 (Uvicorn) is running properly on port 8000 before sending chats in Streamlit.
- **Port 8501 / 8000 in use:** Kill existing zombie processes or specify different ports (`--port 8001` for uvicorn, `--server.port 8502` for streamlit) and update `rag.py` to match.
