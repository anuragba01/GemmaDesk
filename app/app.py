"""
app.py - GemmaDesk Main Entry Point

This is a Streamlit application that provides a multimodal RAG (Retrieval-Augmented Generation) 
interface. It allows users to upload documents (PDF, TXT), audio/video (transcribed via Whisper), 
and images. The application runs entirely offline using Google's LiteRT and Nomic embeddings.
"""
import os
import sys
import tempfile
import warnings
import logging
from pathlib import Path
import streamlit as st

# Suppress transformer verbosity before imports to keep the logs clean
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path so we can import our custom engines and utilities
sys.path.append(str(Path(__file__).parent.parent / "src"))

from engines.document import DocumentEngine
from engines.vectorstore import VectorStoreEngine
from rag.gemma import GemmaEngine
from engines.media import MediaEngine
from engines.vision import VisionEngine
from rag.rag import CHROMA_DIR, EMBED_MODEL, IMAGE_DIR, IMAGE_MANIFEST, MODEL_PATH, MultimodalRAG
from utilities import chat_storage, profile

# Global configurations and logging setup
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="GemmaDesk", layout="wide")

@st.cache_resource(show_spinner="Loading RAG engine...")
def load_components():
    """
    Initializes all backend engines. Uses st.cache_resource to ensure 
    models are only loaded into memory once across app reruns.
    """
    gemma_engine = GemmaEngine(MODEL_PATH)
    vector_store = VectorStoreEngine(CHROMA_DIR, EMBED_MODEL)
    doc_engine = DocumentEngine(vector_store, gemma_engine=gemma_engine)
    media_engine = MediaEngine("base", doc_engine)
    vision_engine = VisionEngine(IMAGE_DIR, IMAGE_MANIFEST)
    rag = MultimodalRAG(doc_engine, vector_store, vision_engine, media_engine, gemma_engine)
    return {
        "doc_engine": doc_engine,
        "media_engine": media_engine,
        "vision_engine": vision_engine,
        "rag": rag,
    }

# Initialize engines
components = load_components()
doc_engine = components["doc_engine"]
media_engine = components["media_engine"]
vision_engine = components["vision_engine"]
rag = components["rag"]

# --- User Profile Logic ---
if not profile.has_profile():
    @st.dialog("Welcome to GemmaDesk!")
    def setup_dialog():
        """Prompts new users to set up their learning profile (language, background)."""
        st.write("Please set up your profile to personalize your experience.")
        lang = st.text_input("Preferred Language", value="English")
        edu = st.selectbox("Education Level", ["High School", "Undergraduate", "Graduate", "Professional", "Other"])
        bg = st.text_input("Background (e.g., Business, Technology, Arts)")
        continent = st.selectbox("Continent (Optional)", ["None", "North America", "South America", "Europe", "Asia", "Africa", "Australia", "Antarctica"])
        if st.button("Save Profile"):
            profile_data = {"language": lang, "education": edu, "background": bg}
            if continent != "None":
                profile_data["continent"] = continent
            profile.save_profile(profile_data)
            st.rerun()
    setup_dialog()

user_profile = profile.load_profile()

# --- Chat Session Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = chat_storage.generate_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar UI ---
with st.sidebar:
    st.title("GemmaDesk")
    st.caption("Offline Multimodal Study Tool")
    st.divider()

    st.header("Personalization")
    if user_profile:
        new_lang = st.text_input("Language", value=user_profile.get("language", "English"))
        if new_lang and new_lang != user_profile.get("language"):
            user_profile["language"] = new_lang
            profile.save_profile(user_profile)
            st.rerun()
    st.divider()

    # Display database statistics
    stats = rag.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Text Chunks", stats["text_chunks"])
    col2.metric("Images", stats["images"])
    st.divider()

    # File Upload Section
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "PDF · TXT · MP3 · WAV · MP4 · JPG · PNG",
        type=["pdf", "txt", "mp3", "wav", "mp4", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Process Uploaded Files
    if st.button("Process Files", type="primary", disabled=not uploaded_files):
        for uf in uploaded_files:
            ext = Path(uf.name).suffix.lower()
            
            # Save uploaded file to a temporary location for the engines to read
            dest_dir = os.path.join(tempfile.gettempdir(), "gemmadesk_uploads")
            os.makedirs(dest_dir, exist_ok=True)
            tmp_path = os.path.join(dest_dir, uf.name)
            
            with open(tmp_path, "wb") as f:
                f.write(uf.getvalue())

            try:
                # Route file to the appropriate engine based on extension
                if ext == ".pdf":
                    with st.spinner(f"Indexing {uf.name}..."):
                        n = doc_engine.ingest_pdf(tmp_path)
                    st.success(f"{uf.name} → {n} chunks")

                elif ext == ".txt":
                    with st.spinner(f"Indexing {uf.name}..."):
                        n = doc_engine.ingest_text(tmp_path)
                    st.success(f"{uf.name} → {n} chunks")

                elif ext in (".mp3", ".wav"):
                    with st.spinner(f"Transcribing {uf.name}..."):
                        n = media_engine.ingest_audio(tmp_path)
                    st.success(f"{uf.name} → {n} transcript chunks")

                elif ext == ".mp4":
                    with st.spinner(f"Transcribing {uf.name}..."):
                        n = media_engine.ingest_video(tmp_path)
                    st.success(f"{uf.name} → {n} transcript chunks")

                elif ext in (".jpg", ".jpeg", ".png"):
                    with st.spinner(f"Indexing {uf.name}..."):
                        added = vision_engine.ingest_image(tmp_path)
                    if added:
                        st.success(f"{uf.name} registered")
                    else:
                        st.info(f"{uf.name} already indexed")

            except Exception as e:
                st.error(f"{uf.name}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        st.rerun()

    st.divider()

    # Search and Filter Settings
    st.header("Search Filters")
    source_map = rag.get_source_map()
    if source_map:
        selected_basenames = st.multiselect(
            "Limit search to specific files:",
            options=list(source_map.keys()),
            default=[]
        )
        selected_paths = [source_map[b] for b in selected_basenames]
    else:
        st.info("No files indexed yet.")
        selected_paths = []

    # Visual context selection
    indexed_image_paths = set(vision_engine.get_valid_images())
    selected_image_paths = [path for path in selected_paths if path in indexed_image_paths]
    if len(selected_image_paths) > 10:
        st.error("Select up to 10 images at once.")

    st.divider()

    # Conversation Management
    st.header("Chat Actions")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.session_id = chat_storage.generate_session_id()
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Quick Study Templates
    with st.expander("📖 Study Templates", expanded=True):
        from rag import prompts
        
        def trigger_template(prompt):
            st.session_state.template_query = prompt
            
        if st.button("📝 Summarize Material", use_container_width=True, on_click=trigger_template, args=(prompts.TEMPLATE_SUMMARIZE,)):
            pass
        if st.button("🛠️ Practical Use Cases", use_container_width=True, on_click=trigger_template, args=(prompts.TEMPLATE_PRACTICAL,)):
            pass
        if st.button("📚 Prerequisites", use_container_width=True, on_click=trigger_template, args=(prompts.TEMPLATE_PREREQUISITES,)):
            pass

    st.divider()

    # Past Chat Sessions
    with st.expander("🕰️ Chat History", expanded=False):
        sessions = chat_storage.list_sessions()
        if sessions:
            for s in sessions[:10]:  # Show last 10 chats
                title = s["title"]
                if st.session_state.session_id == s["id"]:
                    title = f"👉 {title}"
                if st.button(title, key=s["id"], use_container_width=True):
                    st.session_state.session_id = s["id"]
                    st.session_state.messages = chat_storage.load_session(s["id"])
                    st.rerun()
        else:
            st.caption("No saved chats.")

    st.divider()

    with st.expander("Danger Zone"):
        if st.button("Clear All Indexed Data", type="secondary"):
            rag.clear_all()
            st.session_state.messages = []
            st.success("Cleared!")
            st.rerun()

# --- Main Chat UI ---
st.header("Chat")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

# Handle user input
question = st.chat_input("Ask a question about your documents...")

# Handle template triggers
if st.session_state.get("template_query"):
    question = st.session_state.template_query
    st.session_state.template_query = None

if question:
    # Add user message to history and UI
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if len(selected_image_paths) > 10:
                    raise ValueError("Select up to 10 images at once.")
                
                # Call the RAG orchestrator for a streaming response
                result = rag.query_stream(
                    question,
                    filter_paths=selected_paths,
                    history=st.session_state.messages[:-1],
                    user_profile=user_profile
                )

                # Use Streamlit's native streaming for a typewriter effect
                full_response = st.write_stream(result["stream"])
                
                if result.get("sources"):
                    st.caption("Sources: " + ", ".join(result["sources"]))

                # Save the complete interaction to history and persistent storage
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": result.get("sources", []),
                })
                
                chat_storage.save_session(st.session_state.session_id, st.session_state.messages)

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
