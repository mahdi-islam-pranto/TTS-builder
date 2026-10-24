"""Download the Bangla VITS models (by @mobassir94, Apache 2.0) into ./model/.

    python download_model.py            # both voices
    python download_model.py male       # only one voice

The released zips (~900 MB each) are full training checkpoints. After extracting,
the optimizer state is stripped so the model loads faster and uses less disk.
"""

import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(BASE_DIR, "model")
RELEASE_URL = "https://github.com/coqui-ai/TTS/releases/download/v0.13.3_models"
VOICES = {
    "male": "tts_models--bn--custom--vits_male",
    "female": "tts_models--bn--custom--vits_female",
}


def fetch(url, dest, retries=10):
    """Download url to dest, resuming from the partial file if the connection drops."""
    for attempt in range(1, retries + 1):
        done = os.path.getsize(dest) if os.path.exists(dest) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 206:  # server ignored Range: start over
                    done = 0
                total = done + int(resp.headers.get("Content-Length", 0))
                with open(dest, "ab" if done else "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            print()
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # already complete
                return
            raise
        except (OSError, urllib.error.URLError) as e:
            print(f"\n  connection lost ({e}), retry {attempt}/{retries} ...")
    raise RuntimeError(f"Download failed after {retries} retries: {url}")


def strip_checkpoint(path):
    """Keep only the model weights (drop optimizer/scheduler state)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if set(state.keys()) == {"model"}:
        return
    torch.save({"model": state["model"]}, path)


def download_voice(voice):
    name = VOICES[voice]
    out_dir = os.path.join(MODEL_ROOT, f"vits-{voice}")
    model_file = os.path.join(out_dir, "model_file.pth")
    if os.path.isfile(model_file) and os.path.isfile(os.path.join(out_dir, "config.json")):
        print(f"[{voice}] already downloaded in {out_dir}")
        return

    os.makedirs(MODEL_ROOT, exist_ok=True)
    zip_path = os.path.join(MODEL_ROOT, f"{name}.zip")
    if not os.path.isfile(zip_path):
        print(f"[{voice}] downloading {name}.zip ...")
        tmp_path = zip_path + ".part"
        fetch(f"{RELEASE_URL}/{name}.zip", tmp_path)
        os.replace(tmp_path, zip_path)

    print(f"[{voice}] extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(MODEL_ROOT)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(os.path.join(MODEL_ROOT, name), out_dir)
    os.remove(zip_path)

    print(f"[{voice}] stripping training state from checkpoint ...")
    strip_checkpoint(model_file)
    print(f"[{voice}] done: {model_file} ({os.path.getsize(model_file) / 1e6:.0f} MB)")


if __name__ == "__main__":
    voices = sys.argv[1:] or list(VOICES)
    for v in voices:
        if v not in VOICES:
            sys.exit(f"Unknown voice '{v}'. Choose from: {', '.join(VOICES)}")
        download_voice(v)
