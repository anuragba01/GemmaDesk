# GemmaDesk: Offline Multimodal Study Assistant

GemmaDesk is a local-first, privacy-focused Retrieval-Augmented Generation (RAG) system designed to transform personal documents and media into an interactive, AI-powered knowledge base. By utilizing Google's **Gemma 4 (LiteRT)** and specialized local processing engines, it allows users to chat with their files—including PDFs, Text, Audio, Video, and Images—without ever sending data to the cloud.

## The Vision: Democratizing AI for Everyone

AI is often an exclusive tool, requiring high-speed internet, expensive subscriptions, and massive hardware. This creates a digital divide, leaving students in remote areas or with limited resources behind. 

**GemmaDesk destroys that barrier.** Our mission is to provide high-end AI capabilities to anyone, anywhere, regardless of their internet connectivity. By running state-of-the-art models directly on standard consumer hardware, we ensure that a world-class personal tutor is always available—100% offline, 100% private, and 100% accessible. 

## Core Capabilities

- **Personalized Polyglot Tutor:** GemmaDesk breaks down language barriers. It can read dense academic jargon in one language and instantly explain it in simple terms in another, supporting over 140 languages.
- **Visual Intelligence:** It doesn't just read text—it natively *sees* diagrams, charts, and math formulas inside textbooks, providing clear, grounded explanations.
- **Temporal Video Search:** Large video lectures are no longer a black box. Users can ask about specific moments (e.g., *"What was on the blackboard at 14:30?"*) and GemmaDesk will instantly find the exact timestamp.
- **Total Data Sovereignty:** Your study materials, private notes, and recorded lectures never leave your machine.

## High-Level Design (HLD)

GemmaDesk follows a modular "Unified Engine" architecture, designed to run as a single process for maximum efficiency.

```mermaid
graph TD
    User([User Query]) --> Gateway[Intent Gateway]
    Gateway -->|Semantic Search| Chroma[(ChromaDB)]
    Gateway -->|Temporal Bypass| Media[Media Engine]
    Gateway -->|Full Content Bypass| Doc[Document Engine]
    
    Chroma --> Orchestrator[RAG Orchestrator]
    Media --> Orchestrator
    Doc --> Orchestrator
    
    Orchestrator --> Prompt[Multimodal Prompt Builder]
    Prompt --> Gemma[Gemma 4 LiteRT Engine]
    Gemma --> Response([Streaming Response])
```

- **Intent Gateway:** Routes queries based on user intent (e.g., bypassing search for explicit summary requests or timestamp queries).
- **RAG Orchestrator:** The central brain coordinating context retrieval across Vision, Media, and Document engines.
- **LiteRT Engine:** An optimized local inference layer for the 4-bit quantized Gemma model.

## Getting Started

> **⚠️ IMPORTANT — Clone to your Home Directory**
> The AI model uses memory-mapping to load efficiently. Always clone and run GemmaDesk from your **home directory** (`~/`), not from an external drive or USB stick, to avoid load errors.

### Prerequisites
Install the required system libraries (one-time):
```bash
sudo apt install libxcb-cursor0 libgl1 python3-pip
```

### 1. Clone the Repository
```bash
git clone https://github.com/anuragba01/gemmadesk.git ~/gemmadesk
cd ~/gemmadesk
```

### 2. Install `uv` (Fast Python Package Manager)
```bash
pip install uv
```
> `uv` will automatically use the correct Python version. No manual Python install needed.

### 3. Create Environment & Install Dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 4. Launch the App
```bash
uv run streamlit run app/app.py
```

### 5. First-Time Setup (Automatic)
On first launch, the app will show a **Setup Manager** screen.
Click **"Download Missing Models"** and wait for the AI models to download (~3.2 GB total):

| Model | Size | Purpose |
|:---|:---|:---|
| Gemma 4 LiteRT | ~2.5 GB | Core language model |
| BGE Embeddings | ~130 MB | Semantic search |
| Whisper Base | ~140 MB | Audio/video transcription |

> These download **once only**. After that, the app runs completely offline.

Once the download completes, the app automatically opens the main chat interface.

## Live Demo
You can view the project details and access downloads on our landing page:
[**GemmaDesk Official Website**](https://anuragba01.github.io/gemmadesk/)

## Detailed Documentation
- [**Developer Setup Guide**](doc/guide.md): Installation and configuration.
- [**Detailed System Design**](doc/detailed_system_design.md): LLD, data pipelines, and trade-offs.
- [**File Structure Guide**](doc/file.md): Map of the repository's modules.
