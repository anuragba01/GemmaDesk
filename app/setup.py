import os
import streamlit as st
import time

# Use a permanent cache dir — fastembed defaults to /tmp which is wiped on reboot
FASTEMBED_CACHE = os.path.expanduser("~/.cache/fastembed_models")
GEMMA_MODEL_LABEL = "Gemma 4 LiteRT Model (3.5+ GB)"

def check_dependencies() -> list:
    """
    Checks if the heavy AI models have been downloaded to the local system.
    Returns a list of missing models.
    """
    missing = []
    
    # Check Gemma Model
    gemma_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "gemma-4-E4B-it.litertlm"))
    if not os.path.exists(gemma_path):
        missing.append(GEMMA_MODEL_LABEL)
        
    # Check FastEmbed Embedding Model — uses permanent cache dir
    # fastembed saves as: <cache_dir>/models--qdrant--bge-small-en-v1.5-onnx-q
    bge_model_dir = os.path.join(FASTEMBED_CACHE, "models--qdrant--bge-small-en-v1.5-onnx-q")
    if not os.path.exists(bge_model_dir):
        missing.append("BGE Embedding Model (130 MB)")
        
    # Check Faster-Whisper
    whisper_cache = os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper-base")
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
    
    if not missing_deps:
        st.success("All models are ready!")
        if st.button("Start GemmaDesk →", type="primary"):
            st.rerun()
        return

    st.error("The following models are missing from your system:")
    for dep in missing_deps:
        st.markdown(f"- **{dep}**")
        
    st.info("These models will only be downloaded once. After this, GemmaDesk will run entirely offline.")
    
    if st.button("Download Missing Models", type="primary"):
        with st.status("Downloading AI Models...", expanded=True) as status:
            try:
                if GEMMA_MODEL_LABEL in missing_deps:
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
                
                if "BGE Embedding Model (130 MB)" in missing_deps:
                    st.write("Downloading BGE Embedding Model (ONNX)...")
                    from fastembed import TextEmbedding
                    os.makedirs(FASTEMBED_CACHE, exist_ok=True)
                    TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=FASTEMBED_CACHE)
                    st.write("✅ BGE Embedding Model downloaded.")
                    
                if "Whisper Base Model (140 MB)" in missing_deps:
                    st.write("Downloading Faster-Whisper Base...")
                    from faster_whisper import WhisperModel
                    WhisperModel("base", device="cpu", compute_type="int8")
                    st.write("✅ Whisper Base downloaded.")
                    
                status.update(label="All models downloaded successfully!", state="complete", expanded=False)
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                status.update(label="Download failed.", state="error", expanded=True)
                st.error(f"An error occurred: {e}")
