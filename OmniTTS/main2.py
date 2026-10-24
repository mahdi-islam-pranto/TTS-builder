import os

# Never contact Hugging Face; load only from the local model folder.
# Must be set before importing omnivoice / transformers / huggingface_hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
from datetime import datetime

import soundfile as sf
import torch
from omnivoice import OmniVoice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "OmniVoice")
SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_local")

# Use GPU if available, otherwise CPU. float16 is only reliable on GPU.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32


def load_model(model_dir=MODEL_DIR):
    if not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        raise FileNotFoundError(
            f"Model not found in {model_dir}. Run `python download_model.py` first."
        )
    print(f"Loading model from {model_dir} on {DEVICE} ({DTYPE}) ...")
    start = time.time()
    model = OmniVoice.from_pretrained(model_dir, device_map=DEVICE, dtype=DTYPE)
    print(f"Model loaded in {time.time() - start:.1f}s")
    return model


def generate_bangla_audio(
    model,
    text,
    filename=None,
    instruct="female",
    num_step=32,
    speed=1.0,
    output_dir=OUTPUT_DIR,
):
    """Generate speech for Bengali text and save it as a .wav file.

    instruct: voice description, e.g. "female", "female, high pitch",
              "female, young adult", "female, whisper".
    num_step: 32 = best quality, 16 = roughly 2x faster (useful on CPU).
    Returns the path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = datetime.now().strftime("bangla_%Y%m%d_%H%M%S.wav")
    if not filename.endswith(".wav"):
        filename += ".wav"
    path = os.path.join(output_dir, filename)

    start = time.time()
    with torch.inference_mode():
        audio = model.generate(
            text=text,
            instruct=instruct,
            num_step=num_step,
            speed=speed,
        )
    sf.write(path, audio[0], SAMPLE_RATE)
    print(f"Audio generated success")
    print(f"Saved {path} ({time.time() - start:.1f}s)")
    return path


if __name__ == "__main__":
    model = load_model()


    # generate_bangla_audio(model, "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।", "sentence1")

    # generate_bangla_audio(model, "ধন্যবাদ", "word1")

    # generate_bangla_audio(
    #     model,
    #     "আজকের আবহাওয়া খুব সুন্দর, চলো আমরা বাইরে ঘুরতে যাই।",
    #     "sentence2",
    #     num_step=16,  # faster on CPU
    # )

    generate_bangla_audio(
            model,
            "ধন্যবাদ",
            # "আজকের আবহাওয়া খুব সুন্দর, চলো আমরা বাইরে ঘুরতে যাই।",
            "sentence3",
            # num_step=16,  # faster on CPU
            num_step=8,  # faster on CPU
        )