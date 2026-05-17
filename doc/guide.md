# GemmaDesk Developer Guide

This guide explains how to set up and run GemmaDesk locally.

## Architecture Overview
GemmaDesk is a local-first multimodal RAG desktop app built around a single Streamlit process with cached local models.

Core runtime pieces:
1. **Streamlit UI**: File upload, source selection, learner profile, session history, and streaming chat output.
2. **Local retrieval stack**:
   - **Gemma 4 via LiteRT** for answer generation
   - **ChromaDB** for local vector storage
   - **FastEmbed / BGE small** for embeddings
   - **Faster-Whisper** for audio/video transcription
   - **imageio-ffmpeg** for bundled ffmpeg-based media processing
3. **Orchestration layer**:
   - transcript-first media QA
   - exact media-duration validation
   - summary bypass for whole-document requests
   - long-term chat-memory retrieval
   - multimodal attachment only when needed

## 1. Setup & Installation

Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Windows activation:

```bash
.venv\Scripts\activate
```


## 2. Automated Model Setup

On first launch, GemmaDesk checks for three local model dependencies:
1. Gemma 4 LiteRT model
2. BGE embedding model cache
3. Faster-Whisper base model

If any are missing:
1. Launch the app.
2. You will see the setup screen.
3. Click **Download Missing Models**.
4. The app downloads and caches the required models locally.
5. When setup finishes, the UI reloads into the main chat interface.

## 3. Running the Application

### A. Desktop Mode

```bash
python3 script/launcher.py
```

### B. Developer Mode

```bash
uv run streamlit run app/app.py
```


## 5. Chat Memory Behavior

GemmaDesk uses two layers of memory:
1. **Short-term memory**: recent conversation passed directly in the active prompt.
2. **Long-term memory**: every 8 messages, the latest chat block is embedded into ChromaDB and can be retrieved later by `session_id`.

## Troubleshooting

- **Slow first load:** Initial model loading can take time because Gemma 4, embeddings, and Whisper are loaded locally.
- **GPU fallback:** GemmaDesk attempts GPU-backed vision inference first. If it fails, LiteRT falls back to CPU automatically.
- **Unreliable audio answers:** If an audio file contains music, noise, or weak speech, GemmaDesk may refuse to answer from it rather than hallucinate a transcript.
- **Missing directory errors:** Ensure the project root is writable so the app can create `model/`, `chroma_db/`, `uploaded_media/`, `uploaded_images/`, and `chat_sessions/`.
