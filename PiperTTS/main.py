import os
import time
from datetime import datetime

import numpy as np
import soundfile as sf
from bangla import convert_english_digit_to_bangla_digit
from bnnumerizer import numerize
from bnunicodenormalizer import Normalizer
from piper import PiperVoice, SynthesisConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "model", "bn", "bn_BD", "google", "medium", "bn_BD-google-medium.onnx"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_local")
NUM_SPEAKERS = 16
SENTENCE_PAUSE = 0.25  # seconds of silence between sentences

_bnorm = Normalizer()


def normalize_text(text):
    """Digits -> Bangla words, fix broken unicode."""
    text = convert_english_digit_to_bangla_digit(text)
    text = numerize(text)
    text = text.replace("\n", " ")
    words = [_bnorm(w)["normalized"] for w in text.split()]
    return " ".join(w for w in words if w)


def load_model(model_path=MODEL_PATH):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run `python download_model.py` first.")
    print(f"Loading {os.path.basename(model_path)} ...")
    start = time.time()
    voice = PiperVoice.load(model_path)
    list(voice.synthesize("ধন্যবাদ।"))  # warm-up: the first call is several times slower
    print(f"Model loaded in {time.time() - start:.1f}s ({voice.config.num_speakers} speakers)")
    return voice


def show_phonemes(voice, text):
    """Print the espeak phonemes Piper will pronounce (useful to spot wrong pronunciation)."""
    for sentence in voice.phonemize(normalize_text(text)):
        print("".join(sentence))


def generate_bangla_audio(
    voice, text, filename=None, speaker_id=0, speed=1.0, output_dir=OUTPUT_DIR
):
    """Generate speech for Bengali text and save it as a .wav file.

    speaker_id: 0-15, this voice has 16 different speakers.
    speed: 1.0 = normal, >1 faster speech, <1 slower speech.
    Returns the path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = datetime.now().strftime("bangla_%Y%m%d_%H%M%S.wav")
    if not filename.endswith(".wav"):
        filename += ".wav"
    path = os.path.join(output_dir, filename)

    config = SynthesisConfig(speaker_id=speaker_id, length_scale=1.0 / speed)
    sample_rate = voice.config.sample_rate
    silence = np.zeros(int(SENTENCE_PAUSE * sample_rate), dtype=np.float32)

    start = time.time()
    chunks = []
    for chunk in voice.synthesize(normalize_text(text), syn_config=config):
        chunks += [chunk.audio_float_array, silence]
    audio = np.concatenate(chunks[:-1])
    elapsed = time.time() - start

    sf.write(path, audio, sample_rate)
    duration = len(audio) / sample_rate
    print(f"Saved {path}")
    print(f"  audio {duration:.1f}s | generated in {elapsed:.2f}s | RTF {elapsed / duration:.3f}")
    return path


def compare_speakers(voice, text, output_dir=os.path.join(OUTPUT_DIR, "speakers")):
    """Generate the same text with all 16 speakers so you can pick the best one."""
    for speaker_id in range(voice.config.num_speakers):
        generate_bangla_audio(voice, text, f"speaker_{speaker_id:02d}", speaker_id, output_dir=output_dir)


if __name__ == "__main__":
    voice = load_model()

    # 1) Listen to all 16 speakers in outputs_local/speakers/ and pick your favourite.
    compare_speakers(voice, "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি। আজকের আবহাওয়া খুব সুন্দর।")

    # 2) Test sentences with the chosen speaker.
    SPEAKER = 0
    show_phonemes(voice, "আজকের আবহাওয়া খুব সুন্দর, চলো আমরা বাইরে ঘুরতে যাই।")

    generate_bangla_audio(voice, "ধন্যবাদ", "word1", SPEAKER)

    generate_bangla_audio(voice, "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।", "sentence1", SPEAKER)

    generate_bangla_audio(
        voice,
        "আজকের আবহাওয়া খুব সুন্দর, চলো আমরা বাইরে ঘুরতে যাই। আমার বয়স 25 বছর, আর আজ ২৪ তারিখ।",
        "sentence2",
        SPEAKER,
    )

    generate_bangla_audio(
        voice,
        "বাংলাদেশ দক্ষিণ এশিয়ার একটি সুন্দর দেশ। এই দেশের মানুষ খুবই অতিথিপরায়ণ। "
        "এখানে ছয়টি ঋতু আছে, এবং প্রতিটি ঋতুর নিজস্ব সৌন্দর্য রয়েছে। "
        "নদীমাতৃক এই দেশে পদ্মা, মেঘনা ও যমুনা প্রধান নদী। ঢাকা বাংলাদেশের রাজধানী এবং সবচেয়ে বড় শহর।",
        "paragraph",
        SPEAKER,
    )
