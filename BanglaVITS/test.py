





if __name__ == "__main__":
    model = load_model("female")  # or "male"

    generate_bangla_audio(model, "ধন্যবাদ", "word1")

    generate_bangla_audio(model, "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।", "sentence1")

    generate_bangla_audio(
        model,
        "আজকের আবহাওয়া খুব সুন্দর, চলো আমরা বাইরে ঘুরতে যাই। আমার বয়স 25 বছর, আর আজ ২৪ তারিখ।",
        "sentence2",
    )