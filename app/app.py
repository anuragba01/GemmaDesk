import os
import sys
import tempfile
import warnings
import logging
from pathlib import Path
import streamlit as st

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from engines.document import DocumentEngine
from engines.gemma import GemmaEngine
from engines.media import MediaEngine
from engines.vision import VisionEngine
from rag.rag import CHROMA_DIR, EMBED_MODEL, IMAGE_DIR, IMAGE_MANIFEST, MODEL_PATH, MultimodalRAG
from utilities import chat_storage, profile

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="GemmaDesk", layout="wide")

@st.cache_resource(show_spinner="Loading RAG engine...")
def load_components():
    doc_engine = DocumentEngine(CHROMA_DIR, EMBED_MODEL)
    media_engine = MediaEngine("base", doc_engine)
    vision_engine = VisionEngine(IMAGE_DIR, IMAGE_MANIFEST)
    gemma_engine = GemmaEngine(MODEL_PATH)
    rag = MultimodalRAG(doc_engine, vision_engine, gemma_engine)
    return {
        "doc_engine": doc_engine,
        "media_engine": media_engine,
        "vision_engine": vision_engine,
        "rag": rag,
    }

components = load_components()
doc_engine = components["doc_engine"]
media_engine = components["media_engine"]
vision_engine = components["vision_engine"]
rag = components["rag"]

if not profile.has_profile():
    @st.dialog("Welcome to GemmaDesk!")
    def setup_dialog():
        st.write("Please set up your profile to personalize your experience.")
        lang = st.selectbox("Preferred Language", ["English", "Spanish", "French", "German", "Hindi"])
        bg = st.text_input("Educational Background (e.g. High School, College, Professional)")
        if st.button("Save Profile"):
            profile.save_profile({"language": lang, "background": bg})
            st.rerun()
    setup_dialog()

user_profile = profile.load_profile()

if "session_id" not in st.session_state:
    st.session_state.session_id = chat_storage.generate_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("GemmaDesk")
    st.caption("Offline Multimodal Study Tool")
    st.divider()

    st.header("Personalization")
    if user_profile:
        langs = ["English", "Spanish", "French", "German", "Hindi"]
        current_lang = user_profile.get("language", "English")
        if current_lang not in langs:
            langs.append(current_lang)
        new_lang = st.selectbox("Language", langs, index=langs.index(current_lang))
        if new_lang != current_lang:
            user_profile["language"] = new_lang
            profile.save_profile(user_profile)
            st.rerun()
    st.divider()

    stats = rag.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Text Chunks", stats["text_chunks"])
    col2.metric("Images", stats["images"])
    st.divider()

    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "PDF · TXT · MP3 · WAV · MP4 · JPG · PNG",
        type=["pdf", "txt", "mp3", "wav", "mp4", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if st.button("Process Files", type="primary", disabled=not uploaded_files):
        for uf in uploaded_files:
            ext = Path(uf.name).suffix.lower()
            
            dest_dir = os.path.join(tempfile.gettempdir(), "gemmadesk_uploads")
            os.makedirs(dest_dir, exist_ok=True)
            tmp_path = os.path.join(dest_dir, uf.name)
            
            with open(tmp_path, "wb") as f:
                f.write(uf.getvalue())

            try:
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

    indexed_image_paths = set(vision_engine.get_valid_images())
    selected_image_paths = [path for path in selected_paths if path in indexed_image_paths]
    if len(selected_image_paths) > 10:
        st.error("Select up to 10 images at once.")

    st.divider()

    st.header("Chat Actions")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.session_id = chat_storage.generate_session_id()
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("Study Templates")
    
    @st.dialog("Template Runner", width="large")
    def run_template_dialog(prompt_text: str):
        st.info(f"**Executing:** {prompt_text}")
        with st.spinner("Thinking..."):
            try:
                result = rag.query_stream(
                    prompt_text,
                    filter_paths=selected_paths,
                    history=[],  # Templates run isolated from chat history
                    user_profile=user_profile
                )
                st.write_stream(result["stream"])
                if result.get("sources"):
                    st.caption("Sources: " + ", ".join(result["sources"]))
            except Exception as e:
                st.error(f"Error: {e}")
        if st.button("Close"):
            st.rerun()

    if st.button("📝 Summarize Material", use_container_width=True):
        run_template_dialog("Provide a comprehensive summary of the provided materials. Highlight the top 3 most important takeaways.")
    if st.button("💡 Explain with Analogy", use_container_width=True):
        run_template_dialog("Explain the core concepts of this material using a relatable, real-world analogy.")
    if st.button("🛠️ Practical Use Cases", use_container_width=True):
        run_template_dialog("Provide 3 specific, real-world practical use cases for the concepts discussed.")

    st.divider()

    st.header("Chat History")
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

st.header("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if len(selected_image_paths) > 10:
                    raise ValueError("Select up to 10 images at once.")
                
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
