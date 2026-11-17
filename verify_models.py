
"""
verify_models.py
Verify Gemini text generation and AI image generation for MarketPilot AI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Current Gemini API models
TEXT_MODEL = "gemini-3.8-flash"
IMAGE_MODEL = "gemini-3.1-flash-image"


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "your_gemini_api_key_here":
        print("ERROR: GEMINI_API_KEY is not configured.")
        print("Please configure it in the project .env file.")
        sys.exit(1)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    print("=" * 60)
    print("MARKETPILOT AI - MODEL VERIFICATION")
    print("=" * 60)

    # 1. Test text generation
    print(f"\n[1/2] Testing text model: {TEXT_MODEL}")

    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents="Reply with exactly: TEXT_MODEL_OK",
            config=types.GenerateContentConfig(
                temperature=0.0
            ),
        )

        reply = (response.text or "").strip()

        if not reply:
            raise RuntimeError("The text model returned an empty response.")

        print("SUCCESS: Text generation working.")
        print("Response:", reply)

    except Exception as exc:
        print("FAILED: Text generation error.")
        print(str(exc))
        sys.exit(1)

    # 2. Test image generation
    print(f"\n[2/2] Testing image model: {IMAGE_MODEL}")
    print("Generating a test image. Please wait...")

    try:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=(
                "Generate a minimalist image of one green circle "
                "on a clean white background. No text."
            ),
        )

        image_part = None

        for part in response.parts or []:
            if part.inline_data is not None:
                image_part = part
                break

        if image_part is None:
            raise RuntimeError(
                "The model returned no image. Check API access and quota."
            )

        generated_image = image_part.as_image()

        output_dir = PROJECT_ROOT / "outputs" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "verify_test.png"

        generated_image.save(output_path, format="PNG")

        print("SUCCESS: Image generation working.")
        print(f"Image dimensions: {generated_image.size}")
        print(f"Image saved to: {output_path}")

    except Exception as exc:
        print("FAILED: Image generation error.")
        print(str(exc))
        print("\nCheck your API key, model availability and quota.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL MODEL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Run tests: pytest tests/ -v")
    print("2. Start UI: streamlit run ui/app.py")


if __name__ == "__main__":
    main()
