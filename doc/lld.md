# GemmaDesk Low-Level Design (LLD)

This document provides a detailed technical breakdown of the GemmaDesk multimodal RAG architecture.

---

## 1. Ingestion Pipeline LLD
Responsible for transforming raw files into searchable vector chunks.

### Chunking Strategy
- **Text/PDF:** Uses `RecursiveCharacterTextSplitter`. 
  - `chunk_size`: 500 characters.
  - `chunk_overlap`: 50 characters.
  - *Rationale:* Small chunks are better for local 4B-parameter models to prevent context dilution.
- **Audio/Video:** Uses `openai-whisper`.
  - Transcribes into time-stamped segments.
  - Each segment is treated as a chunk with `[MM:SS]` prefix added to the text.

### Metadata Schema
Every chunk in ChromaDB contains:
- `source`: Absolute path to the original file (used for filtering).
- `type`: `pdf`, `text`, `audio`, or `video`.
- `timestamp`: (Optional) Float value for media files to allow "jump-to-time" features.

### File Type Detection
- Handled in `app/app.py` via file extension mapping (`.suffix.lower()`).
- Routes to `ingest_pdf`, `ingest_text`, `ingest_audio`, or `ingest_video` in the `rag.py` orchestrator.

---

## 2. Embedding LLD
The bridge between raw text and mathematical vectors.

### Text Embeddings
- **Model:** `nomic-ai/nomic-embed-text-v1.5`.
- **Implementation:** `sentence-transformers` via `langchain-huggingface`.
- **Flow:** `Text Chunk` → `Mean Pooling` → `L2 Normalization` → `768-D Vector`.
- **Distance Metric:** **Cosine Similarity** (handled by ChromaDB).

### Vision Embeddings (Planned)
- **Model:** `CLIP (ViT-B/32)`.
- **Implementation:** Images are processed through the CLIP Image Encoder.
- **Storage:** Stored in a separate ChromaDB collection (`image_docs`) or as metadata-linked vectors.
- **Dimensions:** 512-D.

---

## 3. RAG / Retrieval LLD
How the system finds relevant information.

### Query Flow
1. User question is embedded using the same `nomic-embed-text` model.
2. **Search Params:**
   - `top_k`: 4 chunks (balanced for speed vs. context).
   - `filter`: Metadata-based filter using the `source` field if the user selected specific files in the UI.
3. **Session Scoping:** Retrieval is performed against the entire database unless file-level filters are active.

---

## 4. Prompt Construction LLD
How we "talk" to Gemma 4.

### Template Structure
The prompt is assembled as a list of messages:
1. **System Prompt:** Sets the persona ("Friendly Study Assistant") and injects the retrieved **Context**.
2. **Conversation History:** Previous `N` messages from the current session.
3. **User Query:** The current question.

### Example Construction
```python
messages = [
  {"role": "system", "content": "Context: [Chunks from ChromaDB]\nAnswer based on this..."},
  {"role": "user", "content": "Previous Q"},
  {"role": "assistant", "content": "Previous A"},
  {"role": "user", "content": "Current Question"}
]
```

---

## 5. Context Window Management LLD
Gemma 4 has a finite memory (context window).

- **Max Token Limit:** 8192 tokens (approx).
- **Trimming Strategy:** 
  - If total tokens exceed limit, the **oldest chat history** messages are dropped first.
  - **System Prompt & Retrieved Context** are always preserved as priority.
  - **Logic:** `Current Query` > `Retrieved Context` > `System Prompt` > `Recent History`.

---

## 6. Session & Storage LLD
Currently transitioning from JSON files to a structured format.

### Storage Logic
- **Current:** Single JSON file per session in `./chat_sessions`.
- **Proposed SQLite Schema:**
  - `sessions` Table: `id` (UUID), `title`, `created_at`, `updated_at`.
  - `messages` Table: `id`, `session_id`, `role`, `content`, `sources`, `timestamp`.

---

## 7. LiteRT-LM Wrapper LLD
The interface between the Python code and the AI model.

### Direct Wrapper Design (Internal)
Instead of a network server, a `BaseLLM` style class would:
1. **Init:** Load `.litertlm` model into memory (using `st.cache_resource`).
2. **Invoke:** Take `List[Message]`, convert to LiteRT's `Conversation` object.
3. **Stream:** Use a generator to yield `token_text` as the model produces it.

---

## 8. Streamlit UI LLD
The interaction layer.

- **State Management:** Uses `st.session_state` to track:
  - `messages`: Current chat bubble data.
  - `session_id`: Unique ID for the current chat session.
- **File Handling:** Files are temporarily saved to `/tmp/gemmadesk_uploads/` during processing to ensure clean paths for metadata indexing.
- **Rendering:** Custom CSS for "Chat Bubbles" to ensure a premium, modern feel.
