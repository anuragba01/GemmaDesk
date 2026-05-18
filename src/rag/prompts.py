# GemmaDesk Systematic Prompts

CORE_SYSTEM_PROMPT = """You are a GemmaDesk Multimodal Learning Assistant. 
Your goal is to provide accurate and detailed answers using provided text context, images, video frames, and audio snippets.

### GROUNDING RULES:
1. **Multimodal Integration**: Use all provided context (text, visual frames, and audio) to form a complete answer.
2. **Handle Conflicts**: If a provided document or media contradicts your internal knowledge, trust the provided context.
3. **Honest Admissions**: If the answer cannot be found in the provided context or media, state that clearly. Do not hallucinate.
4. **Professional Tone**: Maintain a helpful, educational, and professional tone.

### CITATION STYLE:
1. **Minimalist Citations**: Do not cite individual sentences or paragraphs.
2. **Source Summary**: At the very end of your entire response, provide a single list of sources used in the format: [Sources: filename1, filename2].
"""

CONTEXT_BLOCK_TEMPLATE = """---
[SOURCE: {source}]
[TYPE: {kind}]
CONTENT:
{content}
---"""

IMAGE_LIST_TEMPLATE = "The following images have been provided as visual context: {filenames}."

QUERY_PROMPT_TEMPLATE = """Task: Answer the user's question using the retrieved context blocks and images above.

Retrieved Context:
{context_text}

{image_text}

User Question: {question}"""

# --- Technical Engine Prompts ---

# Whisper helps the model recognize specific technical terms in your project.
WHISPER_INITIAL_PROMPT = "The following is a technical discussion about GemmaDesk, a local RAG tool using LiteRT, ChromaDB, and multimodal Gemma 4 models."

# Nomic Embed v1.5 requires a specific prefix to distinguish queries from documents.
EMBED_QUERY_PREFIX = "search_query: "

# Ingestion Modifiers
HARDNESS_CLASSIFICATION_PROMPT = "Analyze this text and return ONLY one word describing its complexity: EASY, MEDIUM, or HARD."

# UI Templates
TEMPLATE_SUMMARIZE = "Provide a comprehensive summary of the provided materials."
TEMPLATE_PRACTICAL = "Provide 3 specific, real-world practical use cases for the concepts discussed."
TEMPLATE_PREREQUISITES = "What do I need to study before studying this material? What are the prerequisites?"
TEMPLATE_MEDIA_ANALYSIS = "Carefully analyze the provided video frames and audio clips. Summarize the visual events and the spoken content."
TEMPLATE_QUIZ = "Generate a 5-question multiple-choice quiz based on these materials to test my understanding."
TEMPLATE_TUTORIAL = "Transform this information into a clear, step-by-step 'How-To' tutorial."
TEMPLATE_GLOSSARY = "Identify the key technical terms and concepts in these materials and provide a brief glossary."
