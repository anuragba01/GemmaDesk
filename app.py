""

import os
import tempfile
import shutil
import logging
import warnings
from pathlib import Path
import streamlit as st
from rag import MultimodalRAG

# Suppress verbose warnings
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

st.set_page_config(page_title="GemmaDesk", page_icon="", layout="wide")

@st.cache_resource(show_spinner="Loading RAG engine...")
def load_rag():
    return MultimodalRAG()

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR
with st.sidebar:
    st.title(" GemmaDesk")
    st.caption("Offline Multimodal Study Tool")
    st.divider()

    stats = rag.get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Text Chunks", stats["text_chunks"])
    col2.metric("Images", stats["images"])
    st.divider()

    st.header(" Upload Files")
    uploaded_files = st.file_uploader(
        "PDF · TXT · MP3 · WAV · MP4 · JPG · PNG",
        type=["pdf", "txt", "mp3", "wav", "mp4", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if st.button(" Process Files", type="primary", disabled=not uploaded_files):
        for uf in uploaded_files:
            ext = Path(uf.name).suffix.lower()

            # Save uploaded bytes to a temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=ext, prefix="gemmadesk_"
            ) as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name

            try:
                if ext == ".pdf":
                    with st.spinner(f" Indexing {uf.name}..."):
                        n = rag.ingest_pdf(tmp_path)
                    st.success(f" {uf.name} → {n} chunks")

                elif ext == ".txt":
                    with st.spinner(f" Indexing {uf.name}..."):
                        n = rag.ingest_text(tmp_path)
                    st.success(f" {uf.name} → {n} chunks")

                elif ext in (".mp3", ".wav"):
                    with st.spinner(f" Transcribing {uf.name} (Whisper)..."):
                        n = rag.ingest_audio(tmp_path)
                    st.success(f" {uf.name} → {n} transcript chunks")

                elif ext == ".mp4":
                    with st.spinner(f" Extracting & transcribing {uf.name}..."):
                        n = rag.ingest_video(tmp_path)
                    st.success(f" {uf.name} → {n} transcript chunks")

                elif ext in (".jpg", ".jpeg", ".png"):
                    # Images are stored permanently — copy to uploaded_images/
                    dest = os.path.join("./uploaded_images", uf.name)
                    with open(dest, "wb") as f:
                        f.write(uf.getvalue())
                    added = rag.ingest_image(dest)
                    if added:
                        st.success(f" {uf.name} → registered for vision")
                    else:
                        st.info(f" {uf.name} already indexed")

            except Exception as e:
                st.error(f" {uf.name}: {e}")
            finally:
                # Clean up temp file (images were already copied above)
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        st.rerun()  # Refresh stats

    st.divider()

    st.header(" Query Mode")
    mode = st.radio(
        "What are you asking about?",
        [" Text / PDF / Audio / Video", " Images"],
        index=0,
    )
    query_mode = "text" if "Text" in mode else "image"

    st.divider()

    st.header(" Chat Actions")
    if st.button(" New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    with st.expander(" Danger Zone"):
        if st.button(" Clear All Indexed Data", type="secondary"):
            rag.clear_all()
            st.session_state.messages = []
            st.success("Cleared!")
            st.rerun()

# MAIN — Chat
from pathlib import Path

st.header(" Chat")

# Show image gallery if images are indexed
if stats["images"] > 0:
    with st.expander(f" Indexed Images ({stats['images']})", expanded=False):
        img_cols = st.columns(min(stats["images"], 4))
        for i, path in enumerate(rag.image_paths):
            if os.path.exists(path):
                img_cols[i % 4].image(path, caption=os.path.basename(path), use_column_width=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(" Sources: " + ", ".join(msg["sources"]))

# User input
if question := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if query_mode == "image":
                    result = rag.query_image(question)
                else:
                    result = rag.query_text(question)

                st.markdown(result["answer"])
                if result.get("sources"):
                    st.caption(" Sources: " + ", ".join(result["sources"]))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                })

            except Exception as e:
                err = f" Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
