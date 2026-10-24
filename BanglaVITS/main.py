import os
import re
import time
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
from bangla import convert_english_digit_to_bangla_digit
from bnnumerizer import numerize
from bnunicodenormalizer import Normalizer
from TTS.api import TTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_local")
SENTENCE_PAUSE = 0.25  # seconds of silence between sentences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_bnorm = Normalizer()


def normalize_text(text):
    """Digits -> Bangla words, fix broken unicode, unify sentence endings."""
    text = convert_english_digit_to_bangla_digit(text)
    text = numerize(text)
    text = text.replace("\n", " ").replace(";", ",").replace(":", ",")
    words = [_bnorm(w)["normalized"] for w in text.split()]
    return " ".join(w for w in words if w)


def split_sentences(text):
    """Split on । ! ? and make sure every sentence ends with one of them."""
    parts = re.split(r"(?<=[।!?])\s*", text)
    sentences = []
    for p in parts:
        p = p.strip(" ,")
        if not p:
            continue
        if p[-1] not in "।!?":
            p += "।"
        sentences.append(p)
    return sentences


def load_model(voice="female", model_root=MODEL_ROOT, use_onnx=True):
    model_dir = os.path.join(model_root, f"vits-{voice}")
    model_path = os.path.join(model_dir, "model_file.pth")
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found in {model_dir}. Run `python download_model.py {voice}` first."
        )
    print(f"Loading {voice} voice from {model_dir} on {DEVICE} ...")
    start = time.time()
    model = TTS(model_path=model_path, config_path=config_path, progress_bar=False).to(DEVICE)

    # On CPU, ONNX Runtime is ~1.5-2x faster than PyTorch for this model.
    model.use_onnx = use_onnx and DEVICE == "cpu"
    if model.use_onnx:
        vits = model.synthesizer.tts_model
        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.isfile(onnx_path):
            print("Exporting to ONNX (one time only) ...")
            vits.export_onnx(output_path=onnx_path, verbose=False)
        vits.load_onnx(onnx_path)
    print(f"Model loaded in {time.time() - start:.1f}s ({'onnx' if model.use_onnx else 'torch'})")
    return model


def generate_bangla_audio(model, text, filename=None, speed=1.0, output_dir=OUTPUT_DIR):
    """Generate speech for Bengali text and save it as a .wav file.

    speed: 1.0 = normal, >1 faster speech, <1 slower speech.
    Returns the path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = datetime.now().strftime("bangla_%Y%m%d_%H%M%S.wav")
    if not filename.endswith(".wav"):
        filename += ".wav"
    path = os.path.join(output_dir, filename)

    vits = model.synthesizer.tts_model
    sample_rate = model.synthesizer.output_sample_rate
    vits.length_scale = 1.0 / speed
    silence = np.zeros(int(SENTENCE_PAUSE * sample_rate), dtype=np.float32)

    start = time.time()
    chunks = []
    with torch.inference_mode():
        for sentence in split_sentences(normalize_text(text)):
            if model.use_onnx:
                ids = np.array([vits.tokenizer.text_to_ids(sentence)], dtype=np.int64)
                wav = vits.inference_onnx(ids).squeeze()
            else:
                wav = model.tts(sentence, split_sentences=False)
            chunks += [np.asarray(wav, dtype=np.float32), silence]
    audio = np.concatenate(chunks[:-1])
    elapsed = time.time() - start

    sf.write(path, audio, sample_rate)
    duration = len(audio) / sample_rate
    print(f"Saved {path}")
    print(f"  audio {duration:.1f}s | generated in {elapsed:.2f}s | RTF {elapsed / duration:.3f}")
    return path


if __name__ == "__main__":
    model = load_model("female")  # or "male"

    generate_bangla_audio(model, "ধন্যবাদ", "word1")

    generate_bangla_audio(model, "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।", "sentence1")

    generate_bangla_audio(
        model,
        "আজকের আবহাওয়া খুব সুন্দর, চলো আমরা বাইরে ঘুরতে যাই। আমার বয়স 25 বছর, আর আজ ২৪ তারিখ।",
        "sentence2",
    )