"""
vision/captioner.py
Generates a caption + keyword tags for an uploaded image.
Model: LLaVA via Ollama (local)
  — Multimodal LLM: understands both text and images.
  — Runs locally via Ollama — no GPU required (CPU supported).
  — Much richer, more descriptive captions than BLIP.
  — Same Ollama server already used for RAG pipeline.
"""

import re
import base64
import requests
import os
from pathlib import Path

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL   = f"{OLLAMA_HOST}/api/generate"
VISION_MODEL = os.getenv("VISION_MODEL", "llava")


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for Ollama API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(image_path: str) -> str:
    """Generate a descriptive caption using LLaVA via Ollama."""
    image_b64 = image_to_base64(image_path)

    payload = {
        "model": VISION_MODEL,
        "prompt": (
            "Describe this image in one clear sentence. "
            "Focus on the main subject, setting, and any important details."
        ),
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 80,   # Short caption
            "num_thread": 8,     # Use all CPU threads
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Make sure it's running.")
    except requests.exceptions.Timeout:
        raise RuntimeError("LLaVA timed out on CPU. Please try again.")


def generate_tags(image_path: str) -> list[str]:
    """Generate exactly 3 keyword tags using LLaVA."""
    image_b64 = image_to_base64(image_path)

    payload = {
        "model": VISION_MODEL,
        "prompt": (
            "List exactly 3 single-word keywords that best describe this image. "
            "Reply with ONLY the 3 words separated by commas. "
            "Example format: cat, outdoor, sunny"
        ),
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 20,
            "num_thread": 8,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # Parse comma-separated tags
        tags = [t.strip().lower() for t in raw.split(",")]
        # Clean non-alphabetic characters
        tags = [re.sub(r"[^a-zA-Z]", "", t) for t in tags]
        # Filter empty and return top 3
        tags = [t for t in tags if t][:3]

        # Fallback if parsing fails
        if not tags:
            tags = ["image", "photo", "visual"]

        return tags

    except Exception:
        return ["image", "photo", "visual"]


def describe_image(image_path: str) -> dict:
    """
    Full vision pipeline:
    1. Generate rich caption using LLaVA
    2. Generate 3 keyword tags using LLaVA
    3. Return structured result
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    caption = generate_caption(image_path)
    tags    = generate_tags(image_path)

    return {
        "caption": caption,
        "tags":    tags,
        "model":   VISION_MODEL,
    }