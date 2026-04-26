import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please install the huggingface_hub package first:")
    print("uv pip install huggingface_hub")
    sys.exit(1)

# Gemma models are "gated" by Google, so you must accept the terms on HuggingFace first
# and provide your HuggingFace access token.
token = os.environ.get("HF_TOKEN")
if not token:
    print("Error: HF_TOKEN missing.")
    sys.exit(1)

try:
    file_path = hf_hub_download(
        repo_id="google/gemma-4-E4B-it-litert",
        filename="gemma-4-E4B-it.litertlm",
        local_dir=".",
        token=token
    )
    print(f"Saved to: {file_path}")
except Exception as e:
    print(f"Download failed: {e}")
