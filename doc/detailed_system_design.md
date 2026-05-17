# GemmaDesk: Detailed System Design

This document describes the current GemmaDesk architecture as implemented in the codebase.

## 1. Requirements
*   **[x] Functional requirements (user facing)**: Offline multimodal study assistant supporting PDF, TXT, MP3, WAV, MP4, JPG, JPEG, and PNG uploads. Users can chat with the assistant, ask source-filtered questions, request summaries, inspect media timestamps, and maintain persistent chat sessions.
*   **[x] Functional requirements (system/internal)**: The system must ingest and chunk text, transcribe audio/video, store image references, classify document hardness, route summary requests to a full-content retrieval path, support timestamp-aware media questions, and preserve long-term chat memory.
*   **[x] Non-functional requirements**: Entirely local execution, persistent storage on disk, streaming UI responses, and operation on consumer hardware.
*   **[x] AI-specific requirements**: Responses must stay grounded in retrieved local context; multimodal inference must only attach media when necessary; non-speech audio should not be treated as reliable transcript text.
*   **[x] Constraints**: Single-user desktop app, offline, using LiteRT for inference and local embedding/vector storage.
*   **[x] Single user or multi-user**: Single-user Streamlit desktop workflow.
*   **[x] Real-time or batch processing**: Real-time query answering; synchronous file ingestion; background chat-memory indexing every 8 messages.

## 2. Data Pipeline
*   **[x] Data sources identified**: Local filesystem uploads through Streamlit.
*   **[x] Ingestion strategy per data type**:
    *   **PDF/TXT**: Loaded by `DocumentEngine`, split into chunks, hardness-labeled once, then indexed into ChromaDB.
    *   **Audio**: Transcribed by `MediaEngine` with `faster-whisper` (`base`, CPU, int8) using `vad_filter=True` and no domain-specific initial prompt. Transcript segments become timestamped documents.
    *   **Video**: Converted to mono 16 kHz WAV via bundled `ffmpeg`, then transcribed exactly like audio. Timestamped transcript chunks are indexed as `type=video`.
    *   **Images**: Copied into `uploaded_images` and tracked in `image_manifest.json` by `VisionEngine`. Images are not embedded.
    *   **Chat history**: Every 8 messages, `ChatHistoryIngestion` formats the latest block and embeds it into ChromaDB as `type=chat` with `session_id` metadata.
*   **[x] Preprocessing steps**:
    *   Recursive text chunking for documents.
    *   Hardness classification (`EASY`, `MEDIUM`, `HARD`) on a sampled document chunk.
    *   Audio extraction for video.
    *   Transcript reliability validation for audio/video before indexing.
*   **[x] Chunking strategy**: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` for document-like text. Whisper transcript segments are stored directly as individual timestamped units before later chunking/indexing.
*   **[x] Metadata schema**:
    *   `source`: Original file path or chat session path.
    *   `hardness`: `EASY` / `MEDIUM` / `HARD` for indexed text/transcript chunks.
    *   `type`: `audio`, `video`, `chat`, or document-derived types.
    *   `timestamp`: Start time in seconds for media transcript chunks.
    *   `session_id`: Chat-session UUID for long-term memory blocks.
    *   `block`: Chat block number.
    *   `id_range`: Message ID range covered by a chat memory block.
*   **[x] File type detection and routing logic**: Handled in `app/app.py` from extension-based routing.
*   **[x] Error handling**: ffmpeg failures, Chroma failures, and transcription failures are wrapped with logging and surfaced to the UI.
*   **[x] Pipeline idempotency**:
    *   `VisionEngine` prevents duplicate image registration.
    *   Chroma text ingestion does not fully deduplicate repeated uploads.
    *   Invalid media transcripts are rejected before indexing.

## 3. Embedding Design
*   **[x] Embedding model**: `BAAI/bge-small-en-v1.5` via `langchain_community.embeddings.fastembed.FastEmbedEmbeddings`.
*   **[x] Embedding dimensions**: Model-native dimensions for BGE small (handled by FastEmbed runtime).
*   **[x] Distance metric**: Chroma default similarity strategy.
*   **[x] Separate embedding strategy**: Documents, media transcripts, and chat memory blocks are embedded. Images are passed directly to Gemma 4 and are never embedded.
*   **[x] Query prefixing**: Retrieval prepends `search_query: ` to semantic queries to align with embedding model expectations.
*   **[x] Re-embedding strategy**: `Clear All Indexed Data` clears Chroma and the image manifest so data can be reprocessed from scratch.

## 4. Vector Storage
*   **[x] Vector DB**: ChromaDB persisted in `./chroma_db`.
*   **[x] Collection structure**: Single collection named `text_docs`.
*   **[x] Metadata fields defined**: `source`, `hardness`, `type`, `timestamp`, `session_id`, `block`, `id_range`.
*   **[x] Indexing strategy**: Default Chroma local indexing.
*   **[x] Session/user isolation strategy**: Single-user application; chat memory retrieval is isolated by `session_id`.
*   **[x] Persistence and backup strategy**: Local directories and JSONL files on disk; no automatic remote backup.
*   **[x] Separate collections for modalities**: No. Searchable text for documents, transcripts, and chat memory shares one collection. Images remain external.

## 5. Retrieval Design
*   **[x] top-k value**: `k=4` for normal semantic retrieval.
*   **[x] Similarity threshold**: None explicitly configured; ranked top-k retrieval is used.
*   **[x] Metadata filtering strategy**: User-selected sources are converted to `source` filters (`=` or `$in`) in Chroma.
*   **[x] Hybrid search**: Not implemented. Retrieval is semantic plus explicit metadata filters.
*   **[x] Query rewriting/expansion**: No multi-step rewrite pipeline. Only embedding prefixing is applied.
*   **[x] Retrieval quality measurable**: Retrieval counts are logged, but there is no formal offline eval suite yet.
*   **[x] Timestamp-aware retrieval**: For explicit media timestamps, GemmaDesk first fetches transcript chunks for the selected media and chooses the nearest timestamped segments before considering multimodal support clips.

## 6. Prompt Design
*   **[x] Prompt template defined**: `src/rag/prompts.py`.
*   **[x] Component order**: Optional user profile -> core system prompt -> optional confusion modifier -> retrieved context blocks -> optional image list -> user question.
*   **[x] System prompt finalized**: `CORE_SYSTEM_PROMPT` instructs strict grounding and source citation.
*   **[x] Prompt versioned**: No formal prompt versioning system.
*   **[x] Output format instructed**: Plain text/Markdown with `[Source: filename]` citations.
*   **[x] Media metadata injection**: When media files are selected, exact media durations are injected into the system prompt as factual metadata.

## 7. Context Window Management
*   **[x] Max token limit**: Managed by LiteRT; Python code does not do token counting.
*   **[x] Trimming priority order**: No explicit token trimming. The app limits live history passed to the model to the last 8 prior messages in Streamlit and supplements older context via retrieved chat-memory blocks.
*   **[x] What gets dropped first**: Older direct chat turns are omitted from prompt history outside the short-term window; long-term memory retrieval reintroduces only relevant prior blocks.
*   **[x] Specific rules**: If media is attached on the current turn, older assistant messages are skipped when replaying history to avoid stale multimodal anchoring in LiteRT.

## 8. Model / Inference
*   **[x] Model chosen**: `gemma-4-E4B-it.litertlm`.
*   **[x] Quantization strategy**: Pre-quantized LiteRT model artifact.
*   **[x] Runtime**: `litert_lm` Python bindings.
*   **[x] Hardware target**: CPU for audio backend; GPU preferred for vision backend with full CPU fallback if unavailable.
*   **[x] Model loading strategy**: Cached once with Streamlit `@st.cache_resource` in `load_components()`.
*   **[x] Streaming response supported**: Yes, via `send_message_async()` and Streamlit `write_stream()`.
*   **[x] Multimodal invocation strategy**: Most queries remain text-first. Audio/image support clips are only attached when explicitly needed, especially for visual/audio-focused timestamp questions.

## 9. Memory & Session Management
*   **[x] Session storage design**: JSONL files in `./chat_sessions`.
*   **[x] Schema defined**: First line contains metadata (`type`, `id`, `title`, `timestamp`); following lines contain individual messages with sequential `id` fields.
*   **[x] Chat history retrieval strategy**: The UI keeps the current session in `st.session_state`. For inference, the app passes only the most recent message window directly and retrieves older relevant blocks from Chroma by `session_id`.
*   **[x] Session cleanup/expiry strategy**: No automatic deletion. New conversation resets the active `session_id`.
*   **[x] User profile persistence**: Stored locally in `user_profile.json`.

## 10. Evaluation & Quality
*   **[x] Intent classification**: `IntentGateway` uses keyword-first rules for summary/confusion detection, with embedding similarity fallback for ambiguous cases.
*   **[x] Hallucination mitigation**:
    *   Strict grounding prompt.
    *   Source filtering by selected files.
    *   Transcript-first media QA.
    *   Exact media duration validation.
    *   Rejection of unreliable audio/video transcripts.
    *   Refusal to answer from suspicious already-indexed audio transcripts.
*   **[x] Formal evaluation**: No benchmark dataset or regression suite is implemented yet.

## 11. Scalability
*   **[x] Concurrent user handling**: Not a goal; single-user local architecture.
*   **[x] Embedding pipeline scalable**: Sequential and local; optimized for small personal corpora.
*   **[x] Vector DB scalable**: Limited by local Chroma/SQLite constraints.
*   **[x] Model inference scalable**: Single-process local inference tuned for responsiveness over throughput.

## 12. Reliability
*   **[x] Fallback if retrieval returns nothing**: Prompt injects `No relevant text chunks found.` when no context exists.
*   **[x] Graceful degradation**:
    *   LiteRT falls back to CPU if GPU vision init fails.
    *   Out-of-range timestamp questions are rejected before expensive media extraction.
    *   Non-speech or unreliable audio is reported honestly instead of being hallucinated into transcript text.
*   **[x] Data backup strategy**: All persistent state is local and file-based (`chroma_db`, `uploaded_images`, `uploaded_media`, `chat_sessions`, `user_profile.json`).

## 13. Security & Privacy
*   **[x] User data isolation**: Local-first design; no inference or embedding network calls in normal operation after setup models are downloaded.
*   **[x] Uploaded files stored securely**: Uploaded media is written to local directories under the workspace.
*   **[x] PII handling**: Since all retrieval, embedding, and generation stay on-device, sensitive study materials never need to leave the machine.

## 14. Monitoring & Observability
*   **[x] Logging**: Python `logging` used across ingestion, retrieval, and inference modules.
*   **[x] Retrieval tracked**: Logs cover retrieval failures, full-content bypass usage, chat-memory retrieval, and timestamp context selection.
*   **[x] Fallback logging**: Logs include GPU fallback, invalid media extraction, rejected unreliable transcripts, and forced cleanup of orphaned active streams.

## 15. Trade-offs Documented
*   **[x] Text-first media QA**: GemmaDesk prefers transcript-grounded answers and only escalates to multimodal clips when the question explicitly demands visual/audio evidence. This is much faster and more stable than default clip extraction on every video query.
*   **[x] No multimodal embedding space**: Media is converted into transcript text plus optional support clips, rather than building a unified multimodal retrieval index.
*   **[x] Full-content bypass for summaries**: Summary-style queries trade prompt size for recall by loading up to 30 chunks directly.
*   **[x] File-system persistence over service infrastructure**: JSONL, manifests, and local Chroma make the app portable and inspectable, at the cost of stronger concurrency guarantees.
*   **[x] Transcript rejection over aggressive recall**: The system would rather refuse to answer from unreliable audio than index and retrieve likely hallucinations.
