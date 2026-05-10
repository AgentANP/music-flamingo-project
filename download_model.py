from huggingface_hub import snapshot_download
from pathlib import Path
import os

MODEL_REPO = "nvidia/music-flamingo-2601-hf"
TARGET_DIR = Path("hf_cache/hub/models--nvidia--music-flamingo-2601-hf/snapshots")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

def model_present():
    # check for a marker file that indicates the snapshot exists
    for p in TARGET_DIR.rglob("model.safetensors"):
        if p.is_file():
            return True
    return False


def download_model():
    if model_present():
        print("Model already present under", TARGET_DIR)
        return
    print("Downloading model snapshot (this may take a long time)...")
    token = os.getenv("HF_TOKEN") or None
    # snapshot_download will place the repo snapshot into the given local_dir
    path = snapshot_download(repo_id=MODEL_REPO, local_dir=str(TARGET_DIR), token=token)
    print("Downloaded snapshot to", path)


if __name__ == "__main__":
    download_model()
