AI System Design Checklist

1. Requirements
[ ] Functional requirements (user facing)
[ ] Functional requirements (system/internal)
[ ] Non-functional requirements (latency, availability, scale)
[ ] AI-specific requirements (accuracy, hallucination tolerance, offline/online)
[ ] Constraints (hardware, model size, cost, privacy)
[ ] Single user or multi-user defined
[ ] Real-time or batch processing defined

2. Data Pipeline
[ ] Data sources identified (files, APIs, DBs, streams)
[ ] Ingestion strategy per data type defined
[ ] Preprocessing steps defined (cleaning, normalization)
[ ] Chunking strategy defined (size, overlap, method)
[ ] Metadata schema defined per chunk
[ ] File type detection and routing logic
[ ] Error handling for corrupt/unsupported files
[ ] Pipeline is idempotent (re-running doesn't duplicate data)

3. Embedding Design
[ ] Embedding model chosen and justified
[ ] Embedding dimensions defined
[ ] Distance metric chosen (cosine, L2, dot product)
[ ] Separate embedding strategy for different modalities (text, image, audio)
[ ] Embedding model versioned
[ ] Re-embedding strategy if model changes

4. Vector Storage
[ ] Vector DB chosen and justified
[ ] Collection structure defined
[ ] Metadata fields defined per vector
[ ] Indexing strategy defined (HNSW, flat, IVF)
[ ] Session/user isolation strategy
[ ] Persistence and backup strategy
[ ] Separate collections for different modalities

5. Retrieval Design
[ ] top-k value decided
[ ] Similarity threshold defined
[ ] Metadata filtering strategy (by session, file, type)
[ ] Hybrid search considered (semantic + keyword)
[ ] Query rewriting/expansion considered
[ ] Retrieval quality measurable (precision, recall)

6. Prompt Design
[ ] Prompt template defined
[ ] Component order defined (system → history → context → query)
[ ] System prompt finalized
[ ] Prompt versioned
[ ] Prompt tested against edge cases (empty context, long query)
[ ] Output format instructed if needed (JSON, markdown, plain)

7. Context Window Management
[ ] Max token limit defined
[ ] Token counting method defined
[ ] Trimming priority order defined
[ ] What is always preserved (system prompt, current query)
[ ] What gets dropped first (oldest history)
[ ] Tested at limit boundary

8. Model / Inference
[ ] Model chosen and justified
[ ] Quantization strategy defined
[ ] Runtime chosen (LiteRT, llama.cpp, vLLM, Ollama)
[ ] Hardware target defined (CPU/GPU/NPU, RAM limit)
[ ] Model loading strategy (cold start, caching)
[ ] Streaming response supported
[ ] Model swap/upgrade strategy defined
[ ] Fallback if model fails

9. Memory & Session Management
[ ] Session storage design defined (SQLite, JSON, Redis)
[ ] Schema defined (sessions, messages tables)
[ ] Session isolation per user defined
[ ] Chat history retrieval strategy
[ ] Session cleanup/expiry strategy

10. Evaluation & Quality
[ ] Eval dataset defined
[ ] Retrieval quality metrics defined (precision@k, recall@k)
[ ] Response quality metrics defined (relevance, faithfulness)
[ ] Hallucination detection strategy
[ ] Baseline established
[ ] Regression testing on prompt changes

11. Scalability
[ ] Concurrent user handling strategy
[ ] Embedding pipeline scalable (batch processing)
[ ] Vector DB scalable (sharding, partitioning)
[ ] Model inference scalable (batching, replicas)
[ ] Queue for async ingestion if needed

12. Reliability
[ ] Retry logic for model inference
[ ] Fallback response if retrieval returns nothing
[ ] Fallback if embedding fails
[ ] Graceful degradation (answer without context if retrieval fails)
[ ] Data backup strategy

13. Security & Privacy
[ ] User data isolation
[ ] Uploaded files stored securely
[ ] Model doesn't leak data across sessions
[ ] PII handling defined
[ ] Access control if multi-user

14. Monitoring & Observability
[ ] Inference latency tracked
[ ] Retrieval latency tracked
[ ] Token usage tracked
[ ] Retrieval score logged per query
[ ] Hallucination/bad response flagging mechanism
[ ] Alerts on model failure or high latency

15. Trade-offs Documented
[ ] Every major decision has a rationale
[ ] Alternatives considered and rejected with reason
[ ] Known bottlenecks identified
[ ] Known limitations documented (e.g. CLIP vs nomic dimension mismatch)

