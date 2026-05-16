import os
import streamlit as st
import time

def check_dependencies() -> list:
    """
    Checks if the heavy AI models have been downloaded to the local system.
    Returns a list of missing models.
    """
    missing = []
    
    # Check Gemma Model
    gemma_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "gemma-4-E4B-it.litertlm"))
    if not os.path.exists(gemma_path):
        missing.append("Gemma 4 LiteRT Model (2.5 GB)")
        
    # Check Nomic Embedding Cache
    nomic_cache = os.path.expanduser("~/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5")
    if not os.path.exists(nomic_cache):
        missing.append("Nomic Embedding Model (500 MB)")
        
    # Check Whisper Cache
    whisper_cache = os.path.expanduser("~/.cache/whisper/base.pt")
    if not os.path.exists(whisper_cache):
        missing.append("Whisper Base Model (140 MB)")
        
    return missing

def render_setup_page(missing_deps: list):
    """
    Renders a landing page that prompts the user to download missing dependencies.
    """
    st.set_page_config(page_title="GemmaDesk Setup", layout="centered")
    
    st.title("Welcome to GemmaDesk")
    st.markdown("Before we can start your offline, multimodal RAG experience, we need to download the core AI models to your local machine.")
    
    st.error("The following models are missing from your system:")
    for dep in missing_deps:
        st.markdown(f"- **{dep}**")
        
    st.info("These models will only be downloaded once. After this, GemmaDesk will run entirely offline.")
    
    if st.button("Download Missing Models", type="primary"):
        with st.status("Downloading AI Models...", expanded=True) as status:
            try:
                if "Gemma 4 LiteRT Model (2.5 GB)" in missing_deps:
                    st.write("Downloading Gemma 4 LiteRT... (This may take a while)")
                    from huggingface_hub import hf_hub_download
                    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
                    os.makedirs(model_dir, exist_ok=True)
                    hf_hub_download(
                        repo_id="litert-community/gemma-4-E4B-it-litert-lm",
                        filename="gemma-4-E4B-it.litertlm",
                        local_dir=model_dir
                    )
                    st.write("✅ Gemma 4 downloaded.")
                
                if "Nomic Embedding Model (500 MB)" in missing_deps:
                    st.write("Downloading Nomic Embeddings...")
                    from sentence_transformers import SentenceTransformer
                    SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
                    st.write("✅ Nomic Embeddings downloaded.")
                    
                if "Whisper Base Model (140 MB)" in missing_deps:
                    st.write("Downloading Whisper Base...")
                    import whisper
                    whisper.load_model("base")
                    st.write("✅ Whisper Base downloaded.")
                    
                status.update(label="All models downloaded successfully!", state="complete", expanded=False)
                time.sleep(2)
                st.rerun() # Refresh the page to boot the main app
                
            except Exception as e:
                status.update(label="Download failed.", state="error", expanded=True)
                st.error(f"An error occurred: {e}")
