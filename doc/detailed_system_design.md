# GemmaDesk: Detailed System Design

This document details the architecture and implementation of the GemmaDesk system, mapped directly against the LLD checklist provided.

## 1. Requirements
*   **[x] Functional requirements (user facing)**: A multimodal RAG study assistant allowing users to upload documents (PDF, TXT) and media (Audio, Video, Images). Users can chat with the AI, ask questions, request summaries, and filter context by specific files. Includes a persistent chat history and a user profile personalization system.
*   **[x] Functional requirements (system/internal)**: The system must ingest various file types, chunk text, transcribe audio/video, classify document difficulty, and route queries between standard semantic search and full-document bypass based on user intent.
*   **[x] Non-functional requirements**: Local-only execution (offline), responsive UI (Streamlit streaming), persistent state across reboots.
*   **[x] AI-specific requirements**: High hallucination tolerance (strict grounding instructions in prompt), offline execution using local models.
*   **[x] Constraints**: Must run on consumer hardware. Uses `LiteRT` for the LLM to manage memory and performance, falling back to CPU if GPU is unavailable.
*   **[x] Single user or multi-user**: Single-user desktop application.
*   **[x] Real-time or batch processing**: Real-time chat inference; synchronous/sequential batch processing for document ingestion.

## 2. Data Pipeline
*   **[x] Data sources identified**: Local filesystem uploads via Streamlit (`PDF`, `TXT`, `MP3`, `WAV`, `MP4`, `JPG`, `PNG`).
*   **[x] Ingestion strategy per data type**:
    *   **Text/PDF**: Direct load and split via `DocumentEngine`.
    *   **Audio/Video**: Transcribed via `MediaEngine` using Whisper (`base` model). Videos have audio extracted via `ffmpeg` before transcription.
    *   **Images**: Copied to a local `uploaded_images` directory and tracked in `image_manifest.json` by `VisionEngine`.
*   **[x] Preprocessing steps**: Audio extraction from video. Text sampling to determine "hardness" using the Gemma LLM.
*   **[x] Chunking strategy**: `RecursiveCharacterTextSplitter` (Size: 500 characters, Overlap: 50 characters).
*   **[x] Metadata schema**:
    *   `source`: Absolute file path.
    *   `hardness`: EASY/MEDIUM/HARD.
    *   `type`: audio/video (for media).
    *   `timestamp`: Start time in seconds (for media).
    *   `page`: Page number (for PDFs).
*   **[x] File type detection and routing logic**: Managed in `app.py` based on file extensions.
*   **[x] Error handling**: Try/except blocks around ffmpeg execution and ChromaDB insertion.
*   **[x] Pipeline is idempotent**: `VisionEngine` prevents duplicate image copies. ChromaDB ingestion does not currently deduplicate chunks if the same text file is uploaded twice.

## 3. Embedding Design
*   **[x] Embedding model**: `nomic-ai/nomic-embed-text-v1.5` loaded locally via `HuggingFaceEmbeddings`.
*   **[x] Embedding dimensions**: 768 (native to Nomic v1.5).
*   **[x] Distance metric**: L2 (ChromaDB default).
*   **[x] Separate embedding strategy**: Text and Transcripts are embedded. Images are NOT embedded; they are passed directly to the multimodal LLM via file paths.
*   **[x] Re-embedding strategy**: A `Clear All Indexed Data` UI button allows the user to wipe ChromaDB and re-index if the model changes.

## 4. Vector Storage
*   **[x] Vector DB**: ChromaDB (`chroma_db` directory) chosen for its simplicity and local persistence.
*   **[x] Collection structure**: Single collection named `text_docs`.
*   **[x] Metadata fields defined**: Stored inside Chroma's `metadatas` mapping (source, hardness, type, timestamp).
*   **[x] Indexing strategy**: HNSW (ChromaDB default).
*   **[x] Session/user isolation strategy**: N/A (Single user system).
*   **[x] Persistence and backup strategy**: Persistent local directory. No automated cloud backups.
*   **[x] Separate collections for different modalities**: No, all text (documents + transcripts) shares the `text_docs` collection. Images are managed entirely outside the vector DB.

## 5. Retrieval Design
*   **[x] top-k value**: `k=4` chunks for standard semantic search.
*   **[x] Similarity threshold**: Not explicitly defined; relies on top-k sorting.
*   **[x] Metadata filtering strategy**: UI allows filtering by `source` filename, which translates to a `$in` filter in ChromaDB.
*   **[x] Hybrid search**: Not implemented. Pure semantic search.
*   **[x] Query rewriting/expansion**: Prefix `search_query: ` is automatically prepended to user queries to satisfy Nomic embedding requirements.
*   **[x] Retrieval quality measurable**: Logs track the number of chunks retrieved, but quantitative eval metrics (precision/recall) are not implemented.

## 6. Prompt Design
*   **[x] Prompt template defined**: Yes, in `src/rag/prompts.py`.
*   **[x] Component order**: System Profile -> System Prompt -> History -> Context Blocks -> Images -> Query.
*   **[x] System prompt finalized**: `CORE_SYSTEM_PROMPT` enforces strict grounding.
*   **[x] Prompt versioned**: No explicit versioning system.
*   **[x] Output format instructed**: Plain text/markdown expected. Citations forced into `[Source: filename]` format.

## 7. Context Window Management
*   **[x] Max token limit**: Handled opaquely by the underlying `LiteRT` engine.
*   **[x] Trimming priority order**: No active token trimming in Python layer. The entire chat history is passed to the engine.
*   **[x] What gets dropped first**: N/A.
*   **[x] Specific rules**: If images are passed in the current query, older assistant text responses in the history are excluded to prevent visual anchoring issues (`gemma.py` line 35).

## 8. Model / Inference
*   **[x] Model chosen**: Gemma 4 E4B (`gemma-4-E4B-it.litertlm`) for its multimodal capabilities and local efficiency.
*   **[x] Quantization strategy**: Pre-quantized `.litertlm` format used.
*   **[x] Runtime**: `LiteRT` via the Python `litert_lm` bindings.
*   **[x] Hardware target**: CPU/GPU. Attempts to initialize vision backend on GPU; falls back entirely to CPU if GPU is unavailable.
*   **[x] Model loading strategy**: Loaded once and cached in memory using Streamlit's `@st.cache_resource`.
*   **[x] Streaming response supported**: Yes, via `engine.create_conversation().send_message_async()` yielding chunks.

## 9. Memory & Session Management
*   **[x] Session storage design**: JSON files.
*   **[x] Schema defined**: Each session is a JSON file in `./chat_sessions` containing `id`, `title`, `timestamp`, and an array of `messages`.
*   **[x] Chat history retrieval strategy**: Loads the full JSON array and injects it into the LiteRT conversation object before the new query.
*   **[x] Session cleanup/expiry strategy**: No automatic cleanup. User profile preferences (language, education) are stored separately in `user_profile.json`.

## 10. Evaluation & Quality
*   **[x] Intent classification**: `IntentGateway` (uses cosine similarity against predefined phrase clusters) to detect "Summary" or "Confusion" intents.
*   **[x] Hallucination detection strategy**: Mitigated via strict prompt instructions, not algorithmically detected post-generation.
*   (Formal evaluation datasets, baselines, and regression testing are not currently implemented).

## 11. Scalability
*   **[x] Concurrent user handling**: Single-user architecture via Streamlit local server.
*   **[x] Embedding pipeline scalable**: Sequential processing.
*   **[x] Vector DB scalable**: Local SQLite-backed ChromaDB.
*   **[x] Model inference scalable**: Synchronous single-threaded LiteRT execution.

## 12. Reliability
*   **[x] Fallback if retrieval returns nothing**: Prompt template dynamically injects *"No relevant text chunks found"* if retrieval is empty, allowing the LLM to gracefully state it lacks context.
*   **[x] Graceful degradation**: LiteRT automatically falls back to CPU if the GPU fails to initialize.
*   **[x] Data backup strategy**: All data (Chroma, Images, Chats) is written to standard local directories, making manual backup easy.

## 13. Security & Privacy
*   **[x] User data isolation**: Local-first app. Data remains on the user's hard drive.
*   **[x] Uploaded files stored securely**: Files are copied to `uploaded_images` or embedded into Chroma locally.
*   **[x] PII handling**: No external network calls are made for embeddings or inference, inherently protecting PII.

## 14. Monitoring & Observability
*   **[x] Logging**: standard Python `logging` used across all engines (`rag.vectorstore`, `rag.media`, `rag.orchestrator`, etc.).
*   **[x] Retrieval tracked**: Logs indicate when a query bypasses semantic search for full-document retrieval (summary intent).
*   **[x] Fallback logging**: Logs when the GPU backend fails and falls back to CPU.

## 15. Trade-offs Documented
*   **[x] Media Handling**: Chose to transcribe Video/Audio to text and embed the text, rather than using a multimodal embedding space (like ImageBind). This is faster and uses less VRAM locally.
*   **[x] Video Visuals**: Instead of processing every frame of a video, the system extracts a single keyframe and an audio snippet around the exact timestamp of a retrieved text chunk to save inference time.
*   **[x] Session Storage**: Opted for JSON files over a formal SQL database for chat history to keep the architecture portable and file-system based.
*   **[x] Vector Search**: Semantic search is limited to `k=4`. To handle requests for "summaries" that require more context, the system completely bypasses semantic search and dumps up to 30 chunks of a document directly into the prompt (Full-Content Bypass). This trades context window size for guaranteed recall on summarization tasks.
