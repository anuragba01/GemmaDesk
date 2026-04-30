from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="litert-community/gemma-4-E4B-it-litert-lm",
    filename="gemma-4-E4B-it.litertlm",
    local_dir="model/"
)
print(f"Saved to: {file_path}")
