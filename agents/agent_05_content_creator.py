"""
agents/agent_05_content_creator.py

Agent 5: Content Creator
────────────────────────
Reads  → state["campaign_brief"], state["user_input"]
Writes → state["generated_content"], state["workflow_stage"]

Responsibilities:
  • Generate two (2) content variations per requested platform.
  • Instagram: caption, hashtags, CTA.
  • Facebook:  caption, hashtags, CTA.
  • LinkedIn:  caption, hashtags, CTA.
  • Produce 4-6 campaign headlines.
  • Both variations must be distinctly different in approach.

Does NOT:
  • Generate images or image prompts.
  • Re-read raw user input for strategy decisions (uses CampaignBrief).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from core.state import (
    ContentVariations,
    GeneratedContent,
    MarketPilotState,
    PostContent,
    UserCampaignInput,
)
from services.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "agent_05_system_prompt.txt"
SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(brief: Dict[str, Any], platforms: List[str]) -> str:
    key_messages_str = "\n".join(
        f"  {i+1}. {msg}" for i, msg in enumerate(brief.get("key_messages", []))
    )
    cd = brief.get("content_directives", {})
    pr = brief.get("platform_requirements", {})

    # Build per-platform limit notes
    limit_lines = ""
    for p in platforms:
        specs = pr.get(p, {})
        limit_lines += (
            f"\n  {p.upper()}: max {specs.get('character_limit', 500)} chars, "
            f"max {specs.get('hashtag_limit', 10)} hashtags"
        )

    # Build the JSON schema shape Gemini must return
    platform_schema_parts = []
    for p in platforms:
        platform_schema_parts.append(
            f'  "{p}": {{\n'
            f'    "variation_a": {{"caption": "...", "hashtags": ["tag1", "tag2"], "cta": "..."}},\n'
            f'    "variation_b": {{"caption": "...", "hashtags": ["tag1", "tag2"], "cta": "..."}}\n'
            f'  }}'
        )

    schema_str = ",\n".join(platform_schema_parts)

    return f"""Create social media marketing content for the following campaign.

CAMPAIGN: {brief.get("campaign_name", "")}
SUMMARY : {brief.get("campaign_summary", "")}

KEY MESSAGES:
{key_messages_str}

PRIMARY CTA  : {cd.get("primary_cta", "")}
TONE GUIDANCE: {cd.get("tone_guidance", "")}

PLATFORM LIMITS:{limit_lines}

REQUESTED PLATFORMS: {", ".join(platforms)}

RULES:
- Variation A = Direct, benefit-focused. Lead with the product's outcome or benefit.
- Variation B = Story-driven, emotional. Lead with a relatable problem or human scenario.
- Each variation must have a completely different opening line and overall angle.
- hashtags: list as plain words WITHOUT the # symbol.
- Respect platform character limits strictly.

Return ONLY this JSON (include ONLY the requested platforms):
{{
{schema_str},
  "headlines": ["headline 1", "headline 2", "headline 3", "headline 4"]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse a single platform's content
# ─────────────────────────────────────────────────────────────────────────────

def _parse_variations(platform_data: Dict[str, Any]) -> ContentVariations:
    def _post(raw: Dict[str, Any]) -> PostContent:
        caption = raw.get("caption", "")
        return PostContent(
            caption=caption,
            hashtags=raw.get("hashtags", []),
            cta=raw.get("cta", ""),
            char_count=len(caption),
        )

    return ContentVariations(
        variation_a=_post(platform_data.get("variation_a", {})),
        variation_b=_post(platform_data.get("variation_b", {})),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node function
# ─────────────────────────────────────────────────────────────────────────────

def run_content_creator(state: MarketPilotState) -> Dict[str, Any]:
    """
    LangGraph node: Agent 5 — Content Creator.
    """
    logger.info("═══ Agent 5 (Content Creator) starting ═══")

    # Graceful skip if upstream agent failed
    if not state.get("campaign_brief"):
        msg = "Agent 5 skipped: campaign_brief is missing (Agent 1 may have failed)."
        logger.warning(msg)
        return {
            "workflow_stage": "agent_05_skipped",
            "errors": list(state.get("errors", [])) + [msg],
        }

    try:
        brief      = state["campaign_brief"]
        user_input = UserCampaignInput(**state["user_input"])
        platforms  = [p.value for p in user_input.platforms]

        logger.info("Generating content for platforms: %s", platforms)

        # ── Call Gemini ───────────────────────────────────────────────────────
        provider = GeminiProvider()
        raw_data = provider.generate_json(_build_prompt(brief, platforms), SYSTEM_PROMPT)

        # ── Parse platform content ────────────────────────────────────────────
        content_kwargs: Dict[str, Any] = {}
        for platform in platforms:
            if platform in raw_data and raw_data[platform]:
                content_kwargs[platform] = _parse_variations(raw_data[platform])
            else:
                logger.warning("No content returned for platform: %s", platform)

        content = GeneratedContent(
            headlines=raw_data.get("headlines", []),
            **content_kwargs,
        )

        logger.info(
            "Agent 5 complete → platforms: %s | headlines: %d",
            list(content_kwargs.keys()),
            len(content.headlines),
        )

        return {
            "generated_content": content.model_dump(),
            "workflow_stage":    "agent_05_complete",
        }

    except Exception as exc:
        logger.error("Agent 5 FAILED: %s", exc, exc_info=True)
        return {
            "workflow_stage": "agent_05_failed",
            "errors": list(state.get("errors", [])) + [f"Agent 5 (Content Creator): {exc}"],
        }
