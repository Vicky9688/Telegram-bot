"""
vision/captioner.py
Generates a caption + keyword tags for an uploaded image.
Model: Salesforce/blip-image-captioning-base
  — Lightweight (~900MB), runs on CPU, no GPU required.
  — Good accuracy for general scene understanding.
  — Alternative: llava via Ollama for richer descriptions.
"""

import re
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

MODEL_NAME = "Salesforce/blip-image-captioning-base"

_processor = None
_model = None


def load_model():
    """Lazy-load BLIP model (only on first call)."""
    global _processor, _model
    if _processor is None:
        print(f"[Vision] Loading model: {MODEL_NAME}")
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        _model.eval()
        print("[Vision] Model loaded.")


def generate_caption(image_path: str) -> str:
    """Generate a descriptive caption for the image."""
    load_model()
    image = Image.open(image_path).convert("RGB")

    # Conditional captioning (steered) for richer descriptions
    text_prompt = "a photograph of"
    inputs = _processor(image, text_prompt, return_tensors="pt")

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=5,          # Beam search for better quality
            early_stopping=True,
        )

    caption = _processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


def extract_tags(caption: str) -> list[str]:
    """
    Extract 3 keyword tags from the caption using simple NLP heuristics.
    Avoids dependency on spaCy/NLTK for lighter footprint.
    """
    # Remove common stopwords
    stopwords = {
        "a", "an", "the", "is", "are", "of", "in", "on", "at", "to",
        "and", "or", "with", "for", "it", "this", "that", "photograph",
        "image", "photo", "picture", "showing", "there", "some", "has",
    }

    words = re.findall(r'\b[a-zA-Z]{3,}\b', caption.lower())
    filtered = [w for w in words if w not in stopwords]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    # Return top 3 (first meaningful words tend to be most descriptive)
    return unique[:3] if len(unique) >= 3 else unique


def describe_image(image_path: str) -> dict:
    """
    Full vision pipeline:
    1. Generate caption using BLIP
    2. Extract 3 keyword tags
    3. Return structured result
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    caption = generate_caption(image_path)
    tags = extract_tags(caption)

    return {
        "caption": caption,
        "tags": tags,
        "model": MODEL_NAME,
    }
