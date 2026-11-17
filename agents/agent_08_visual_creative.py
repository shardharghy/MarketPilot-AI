
"""
agents/agent_08_visual_creative.py

Agent 08: Visual Creative Agent

Reads:
    campaign_brief
    generated_content
    user_input

Writes:
    visual_assets
    workflow_stage

Current mode:
    Text-only visual creative direction.
    Generates detailed image prompts using Gemini.
    Does NOT generate or save actual images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from core.state import (
    MarketPilotState,
    UserCampaignInput,
    VisualAssets,
)
from services.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "prompts"
    / "agent_08_system_prompt.txt"
)

SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _build_imagen_prompt_request(
    brief: Dict[str, Any],
    content: Dict[str, Any],
    platforms: List[str],
) -> str:
    """Build a request for platform-specific visual concepts."""

    vd = brief.get("visual_directives", {})
    key_messages = brief.get("key_messages", [])
    platform_requirements = brief.get("platform_requirements", {})

    platform_specs = ""

    for platform in platforms:
        spec = platform_requirements.get(platform, {})

        platform_specs += (
            f"\n{platform.upper()}: "
            f"aspect ratio {spec.get('aspect_ratio', '1:1')}"
        )

    return f"""
Create detailed visual concepts and image-generation prompts
for the following marketing campaign.

CAMPAIGN:
{brief.get("campaign_name", "")}

CAMPAIGN SUMMARY:
{brief.get("campaign_summary", "")}

MOOD:
{vd.get("mood", "")}

STYLE:
{vd.get("style_preference", "photorealistic")}

COLOR PALETTE:
{", ".join(vd.get("color_palette", []))}

VISUAL ELEMENTS TO AVOID:
{", ".join(vd.get("avoid", []))}

KEY MESSAGES:
{", ".join(key_messages)}

PLATFORMS:
{platform_specs}

GENERATED CONTENT CONTEXT:
Use the marketing content as additional context when developing
the visual concepts.

REQUESTED PLATFORMS:
{", ".join(platforms)}

RULES:
1. Create one detailed image prompt per requested platform.
2. Each prompt must contain approximately 80-150 words.
3. Follow the campaign mood, colors and visual style.
4. Do not include text, letters, logos or brand names.
5. Minimize human subjects.
6. Respect the visual elements to avoid.
7. Match the aspect ratio requirements of each platform.
8. End each prompt with:
   professional marketing photography, high quality,
   4K resolution, commercial advertising use.

Return ONLY valid JSON in this structure:

{{
    "image_prompts": {{
        "instagram": "Detailed visual prompt...",
        "facebook": "Detailed visual prompt...",
        "linkedin": "Detailed visual prompt..."
    }},
    "style_summary": "One sentence describing the overall visual direction."
}}

Include ONLY the platforms requested.
"""


def run_visual_creative(state: MarketPilotState) -> Dict[str, Any]:
    """
    Agent 08 — Text-only Visual Creative Agent.

    Generates visual concepts and image prompts using Gemini.
    No image-generation API is called.
    """

    logger.info("Agent 08 (Visual Creative) starting")

    if not state.get("campaign_brief"):
        msg = "Agent 08 skipped: campaign_brief is missing."

        logger.warning(msg)

        return {
            "workflow_stage": "agent_08_skipped",
            "errors": list(state.get("errors", [])) + [msg],
        }

    try:
        brief = state["campaign_brief"]
        content = state.get("generated_content") or {}

        user_input = UserCampaignInput(**state["user_input"])
        platforms = [platform.value for platform in user_input.platforms]

        logger.info(
            "Generating visual concepts for platforms: %s",
            platforms,
        )

        # Generate image prompts using Gemini text generation.
        provider = GeminiProvider()

        prompt_request = _build_imagen_prompt_request(
            brief=brief,
            content=content,
            platforms=platforms,
        )

        prompt_data = provider.generate_json(
            prompt_request,
            SYSTEM_PROMPT,
        )

        image_prompts = prompt_data.get("image_prompts", {})
        style_summary = prompt_data.get("style_summary", "")

        if not isinstance(image_prompts, dict):
            raise ValueError("Gemini returned an invalid image_prompts object.")

        # Keep only the requested platforms.
        image_prompts = {
            platform: image_prompts[platform]
            for platform in platforms
            if isinstance(image_prompts.get(platform), str)
            and image_prompts[platform].strip()
        }

        if not image_prompts:
            raise ValueError("No valid visual prompts were generated.")

        style = brief.get(
            "visual_directives", {}
        ).get("style_preference", "photorealistic")

        # No PNG files are generated in text-only mode.
        visual_assets = VisualAssets(
            image_prompts=image_prompts,
            generated_images=[],
            generation_model="Text-only (Gemini)",
            metadata={
                "style_summary": style_summary,
                "style": style,
                "mode": "text_only",
                "images_generated": False,
            },
        )

        logger.info(
            "Agent 08 complete: %d visual prompts generated.",
            len(image_prompts),
        )

        return {
            "visual_assets": visual_assets.model_dump(),
            "workflow_stage": "agent_08_complete",
        }

    except Exception as exc:
        logger.error(
            "Agent 08 FAILED: %s",
            exc,
            exc_info=True,
        )

        return {
            "workflow_stage": "agent_08_failed",
            "errors": list(state.get("errors", []))
            + [f"Agent 08 (Visual Creative): {exc}"],
        }
