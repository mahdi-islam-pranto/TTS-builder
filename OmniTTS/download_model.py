import os

from huggingface_hub import snapshot_download

MODEL_ID = "k2-fsa/OmniVoice"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "OmniVoice")
HF_TOKEN = "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"

if __name__ == "__main__":
    print(f"Downloading {MODEL_ID} to {MODEL_DIR} (~3.3 GB) ...")
    # Re-running resumes/skips files that are already downloaded.
    snapshot_download(repo_id=MODEL_ID, token=HF_TOKEN, local_dir=MODEL_DIR)
    print("Done. Files:")
    for root, _, files in os.walk(MODEL_DIR):
        if ".cache" in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            print(f"  {os.path.relpath(p, MODEL_DIR)}  ({os.path.getsize(p) / 1e6:.1f} MB)")
