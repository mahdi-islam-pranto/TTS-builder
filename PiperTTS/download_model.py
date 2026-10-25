import os

from huggingface_hub import hf_hub_download

REPO_ID = "rhasspy/piper-voices"
VOICE = "bn_BD-google-medium"
VOICE_PATH = "bn/bn_BD/google/medium"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

if __name__ == "__main__":
    print(f"Downloading {VOICE} to {MODEL_DIR} (~77 MB) ...")
    for name in [f"{VOICE}.onnx", f"{VOICE}.onnx.json", "MODEL_CARD"]:
        hf_hub_download(REPO_ID, f"{VOICE_PATH}/{name}", local_dir=MODEL_DIR)
    print("Done. Files:")
    for root, _, files in os.walk(MODEL_DIR):
        if ".cache" in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            print(f"  {os.path.relpath(p, MODEL_DIR)}  ({os.path.getsize(p) / 1e6:.1f} MB)")
