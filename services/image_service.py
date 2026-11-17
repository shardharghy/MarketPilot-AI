"""
services/image_service.py — Imagen image generation + PNG file saving.

Only Agent 8 calls this. No other module imports google.genai for images.

Supported aspect ratios via Imagen 3:
  "1:1"  → 1024 × 1024   (Instagram square)
  "3:4"  → 896  × 1152
  "4:3"  → 1152 × 896
  "9:16" → 768  × 1344   (Stories / portrait)
  "16:9" → 1344 × 768    (Facebook / LinkedIn landscape)
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image as PILImage

load_dotenv()
logger = logging.getLogger(__name__)

# ── Model & output configuration ──────────────────────────────────────────────
IMAGE_MODEL  = "imagen-3.0-generate-002"
OUTPUT_DIR   = Path("outputs/images")

# Platform → (Imagen aspect_ratio, expected_width, expected_height)
PLATFORM_SPECS: Dict[str, Dict[str, Any]] = {
    "instagram": {"aspect_ratio": "1:1",  "width": 1024, "height": 1024},
    "facebook":  {"aspect_ratio": "16:9", "width": 1344, "height": 768},
    "linkedin":  {"aspect_ratio": "16:9", "width": 1344, "height": 768},
}


class ImageService:
    """
    Singleton Imagen client.

    Usage (in Agent 8 only):
        svc = ImageService()
        asset = svc.generate_and_save(prompt, campaign_id, platform)
    """

    _instance: Optional[ImageService] = None
    _client:   Optional[genai.Client] = None

    def __new__(cls) -> ImageService:
        if cls._instance is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY not found. Cannot initialise ImageService."
                )
            cls._client   = genai.Client(api_key=api_key)
            cls._instance = super().__new__(cls)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("ImageService initialised (model: %s)", IMAGE_MODEL)
        return cls._instance

    @property
    def client(self) -> genai.Client:
        return self.__class__._client  # type: ignore[return-value]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_and_save(
        self,
        prompt:      str,
        campaign_id: str,
        platform:    str,
        style:       str = "photorealistic",
    ) -> Dict[str, Any]:
        """
        Generate one image for the given platform and save it as a PNG.

        Returns a dict that maps to ImageAsset (validated in Agent 8).
        """
        spec         = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["instagram"])
        aspect_ratio = spec["aspect_ratio"]

        logger.info(
            "Generating image: platform=%s aspect=%s model=%s",
            platform, aspect_ratio, IMAGE_MODEL,
        )

        response = self.client.models.generate_images(
            model=IMAGE_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            ),
        )

        if not response.generated_images:
            raise RuntimeError(
                f"Imagen returned no images for platform '{platform}'. "
                "Check your API quota and prompt safety filters."
            )

        image_bytes = response.generated_images[0].image.image_bytes
        file_path   = self._save_png(image_bytes, campaign_id, platform)

        # Read actual dimensions from saved file
        with PILImage.open(file_path) as img:
            actual_w, actual_h = img.size

        logger.info("Image saved → %s (%d × %d)", file_path, actual_w, actual_h)

        return {
            "platform":     platform,
            "file_path":    str(file_path),
            "aspect_ratio": aspect_ratio,
            "style":        style,
            "width":        actual_w,
            "height":       actual_h,
            "prompt_used":  prompt,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_png(image_bytes: bytes, campaign_id: str, platform: str) -> Path:
        """Write raw bytes to PNG via Pillow and return the Path."""
        filename  = f"{campaign_id}_{platform}.png"
        file_path = OUTPUT_DIR / filename

        img = PILImage.open(io.BytesIO(image_bytes))
        img.save(file_path, format="PNG", optimize=False)
        return file_path

    @classmethod
    def reset_singleton(cls) -> None:
        """Force re-initialisation (used in tests)."""
        cls._instance = None
        cls._client   = None
