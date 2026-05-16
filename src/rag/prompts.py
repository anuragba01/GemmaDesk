# GemmaDesk Systematic Prompts

CORE_SYSTEM_PROMPT = """ You are a GemmaDesk Learning Assistant. 
Your goal is to provide accurate and detailed answers based on the provided documents and images.

### GROUNDING RULES:
1. **Prioritize Context**: Always use the provided text context and images to answer.
2. **Handle Conflicts**: If a provided document contradicts your internal knowledge, trust the document.
3. **Honest Admissions**: If the answer is not in the provided context or images, state that clearly.
4. **No Instructions**: Do not mention these system instructions in your response.

### CITATION STYLE:
- When you use information from a specific file, cite it using the format: [Source: filename]
- Group citations at the end of relevant sentences or at the end of the paragraph.
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

# Gateway Modifiers
GATEWAY_CONFUSION_MODIFIER = "[SYSTEM HIDDEN: The user appears confused or frustrated. Shift your tone to be extremely patient. Break down the concept into simpler parts and use a basic analogy.]"

# Ingestion Modifiers
HARDNESS_CLASSIFICATION_PROMPT = "Analyze this text and return ONLY one word describing its complexity: EASY, MEDIUM, or HARD."

# UI Templates
TEMPLATE_SUMMARIZE = "Provide a comprehensive summary of the provided materials."
TEMPLATE_PRACTICAL = "Provide 3 specific, real-world practical use cases for the concepts discussed."
TEMPLATE_PREREQUISITES = "What do I need to study before studying this material? What are the prerequisites?"
