"""
app.py - GemmaDesk Main Entry Point

This is a Streamlit application that provides a multimodal RAG (Retrieval-Augmented Generation) 
interface. It allows users to upload documents (PDF, TXT), audio/video (transcribed via Whisper), 
and images. The application runs entirely offline using Google's LiteRT and Nomic embeddings.
"""
import os
import sys
import warnings
import logging
from pathlib import Path
import streamlit as st

def resolve_asset_path(filename: str) -> str:
    import sys
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "asset", filename)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "asset", filename))

logo_path = resolve_asset_path("gemmalogocrop.png")
st.set_page_config(page_title="GemmaDesk", layout="wide", page_icon=logo_path)

# Suppress transformer verbosity before imports to keep the logs clean
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path so we can import our custom engines and utilities
sys.path.append(str(Path(__file__).parent.parent / "src"))

import setup

# Run pre-flight dependency checks BEFORE loading heavy AI modules
missing_deps = setup.check_dependencies()
if missing_deps:
    setup.render_setup_page(missing_deps)
    st.stop() # Halts app.py execution until all dependencies are downloaded

# -- Now it is safe to import the heavy engines --
from engines.document import DocumentEngine
from engines.vectorstore import VectorStoreEngine
from rag.gemma import GemmaEngine
from engines.media import MediaEngine
from engines.vision import VisionEngine
from rag.rag import CHROMA_DIR, EMBED_MODEL, IMAGE_DIR, IMAGE_MANIFEST, MODEL_PATH, MultimodalRAG
UPLOAD_DIR = "./uploaded_media"
from utilities import chat_storage, profile
from engines.chat_ingestion import ChatHistoryIngestion
import threading

# Global configurations and logging setup
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
warnings.filterwarnings("ignore")
log = logging.getLogger("gemmadesk.app")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Hide Streamlit's default toolbar (Stop, Deploy, 3-dot menu) for a clean UI
st.markdown("""
    <style>
        /* Hide the entire top-right toolbar: Stop, Deploy, and ⋮ menu */
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        header[data-testid="stHeader"] {
            visibility: hidden !important;
            height: 0 !important;
        }
        /* Also hide the status widget (running indicator) */
        [data-testid="stStatusWidget"] {
            visibility: hidden !important;
        }
    </style>
""", unsafe_allow_html=True)

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
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
    # EXTERNAL RESET: If we hit this block, it means the user hard-refreshed the page.
    # We must explicitly reset the cached GemmaEngine to clear any stuck C++ session locks.
    if hasattr(rag.gemma_engine, "reset"):
        rag.gemma_engine.reset()
    else:
        # If hot-reloading from an old cached class without the method, force wipe the cache.
        st.cache_resource.clear()

if "session_id" not in st.session_state:
    st.session_state.session_id = chat_storage.generate_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

if "edit_text" not in st.session_state:
    st.session_state.edit_text = ""

if "edit_idx" not in st.session_state:
    st.session_state.edit_idx = -1

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "active_stream" not in st.session_state:
    st.session_state.active_stream = None

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# --- Sidebar UI ---
with st.sidebar:
    st.image(logo_path, width=80)
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
        key=f"uploader_{st.session_state.uploader_key}",
    )

    # Process Uploaded Files
    if st.button("Process Files", type="primary", disabled=not uploaded_files):
        for uf in uploaded_files:
            ext = Path(uf.name).suffix.lower()
            
            # Save uploaded file to a persistent location for the engines to read
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            tmp_path = os.path.join(UPLOAD_DIR, uf.name)
            
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
                    if n > 0:
                        st.success(f"{uf.name} → {n} transcript chunks")
                    else:
                        st.warning(f"{uf.name} processed but no clear speech was found.")

                elif ext in (".jpg", ".jpeg", ".png"):
                    with st.spinner(f"Indexing {uf.name}..."):
                        added = vision_engine.ingest_image(tmp_path)
                    if added:
                        st.success(f"{uf.name} registered")
                    else:
                        st.info(f"{uf.name} already indexed")

            except Exception as e:
                st.error(f"{uf.name}: {e}")

        st.session_state.uploader_key += 1  # Clears the file uploader widget
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
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Determine if this is the last user message
        is_last_user = False
        if msg["role"] == "user":
            is_last_user = True
            for subsequent_msg in st.session_state.messages[i+1:]:
                if subsequent_msg["role"] == "user":
                    is_last_user = False
                    break
                    
        # If we are in edit mode and this is the message being edited
        if st.session_state.edit_mode and is_last_user and st.session_state.edit_idx == i:
            with st.container(border=True):
                edited_text = st.text_area("Edit your query:", value=st.session_state.edit_text, label_visibility="collapsed", key=f"edit_area_{i}")
                col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                with col1:
                    if st.button("Cancel", key=f"cancel_{i}", use_container_width=True):
                        st.session_state.edit_mode = False
                        st.session_state.edit_text = ""
                        st.session_state.edit_idx = -1
                        st.rerun()
                with col2:
                    if st.button("Update", key=f"update_{i}", type="primary", use_container_width=True):
                        st.session_state.pending_question = edited_text
                        st.session_state.edit_mode = False
                        st.session_state.edit_text = ""
                        st.session_state.edit_idx = -1
                        
                        # Pop old messages from this index onwards
                        st.session_state.messages = st.session_state.messages[:i]
                        chat_storage.save_session(st.session_state.session_id, st.session_state.messages)
                        st.rerun()
        else:
            # Render normally
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))

# Find the last user message to enable the edit toolbar button
last_user_idx = -1
last_user_content = ""
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        last_user_idx = idx
        last_user_content = msg["content"]

if last_user_idx != -1 and not st.session_state.edit_mode:
    # Use a hidden anchor div to target the button precisely via CSS
    st.markdown('<div class="edit-btn-anchor"></div>', unsafe_allow_html=True)
    if st.button("✏️ Edit Last Query", key="edit_toolbar_btn", help="Edit your last question"):
        st.session_state.edit_text = last_user_content
        st.session_state.edit_mode = True
        st.session_state.edit_idx = last_user_idx
        st.rerun()

    # CSS to float the button exactly in the bottom right corner above chat input
    st.markdown("""
    <style>
    div.element-container:has(.edit-btn-anchor) {
        display: none;
    }
    div.element-container:has(.edit-btn-anchor) + div.element-container {
        position: fixed !important;
        bottom: 130px !important;
        right: 30px !important;
        z-index: 999 !important;
        width: auto !important;
    }
    /* Add a subtle shadow and rounded corners for premium feel */
    div.element-container:has(.edit-btn-anchor) + div.element-container button {
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Handle user input (disabled while editing)
question = st.chat_input("Ask a question about your documents...", disabled=st.session_state.edit_mode)

# Handle template triggers
if st.session_state.get("template_query"):
    question = st.session_state.template_query
    st.session_state.template_query = None

if st.session_state.get("pending_question"):
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    print(f">>> [UI] USER QUERY RECEIVED: '{question}'", flush=True)
    # If the user sent a new query while an active generation was still running,
    # it means Streamlit aborted the previous run. Proactively cancel and reset.
    if st.session_state.get("is_generating", False):
        print("Active generation was interrupted by a new query! Actively terminating old processes...", flush=True)
        if hasattr(rag, "media_engine") and hasattr(rag.media_engine, "kill_active_processes"):
            rag.media_engine.kill_active_processes()
        if hasattr(rag, "gemma_engine") and hasattr(rag.gemma_engine, "reset"):
            rag.gemma_engine.reset()

    st.session_state.is_generating = True

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
                
                if not selected_paths and not rag.get_stats()["text_chunks"] and not rag.get_stats()["images"]:
                    st.info("No documents are currently indexed. I will answer based on my general knowledge.")
                
                print(f"[RAG] Invoking RAG with filters: {selected_paths}", flush=True)
                # Call the RAG orchestrator for a streaming response
                result = rag.query_stream(
                    question,
                    filter_paths=selected_paths,
                    # Addition 2: Only pass the last 8 messages as direct history (short-term memory)
                    history=st.session_state.messages[-9:-1],
                    user_profile=user_profile,
                    # Addition 2a: Pass session_id so RAG can retrieve older chat blocks
                    session_id=st.session_state.session_id,
                )

                print("[RAG] Stream obtained. Writing typewriter response...", flush=True)
                # Use Streamlit's native streaming for a typewriter effect
                st.session_state.active_stream = block_stream = result["stream"]
                full_response = st.write_stream(st.session_state.active_stream)
                st.session_state.active_stream = None
                print(f"[RAG] Response fully streamed. Length: {len(full_response)} characters", flush=True)
                
                if result.get("sources"):
                    st.caption("Sources: " + ", ".join(result["sources"]))

                # Save the complete interaction to history and persistent storage
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": result.get("sources", []),
                })
                
                chat_storage.save_session(st.session_state.session_id, st.session_state.messages)

                # Addition 1: Every 8 messages, vectorize the latest block in the background
                msg_count = len(st.session_state.messages)
                if msg_count % 8 == 0:
                    session_file = chat_storage.SESSION_DIR + f"/{st.session_state.session_id}.jsonl"
                    ingestion = ChatHistoryIngestion(
                        chat_path=session_file,
                        database_dir=CHROMA_DIR,
                        embed_model=EMBED_MODEL,
                        session_id=st.session_state.session_id,
                    )
                    t = threading.Thread(target=ingestion.ingest_latest_block, daemon=True)
                    t.start()
                    log.info(
                        "Chat ingestion thread launched for block %d (session %s)",
                        msg_count // 8,
                        st.session_state.session_id,
                    )

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            finally:
                st.session_state.is_generating = False
