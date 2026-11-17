"""
agents/agent_01_campaign_supervisor.py

Agent 1: Campaign Supervisor
─────────────────────────────
Reads  → state["user_input"]
Writes → state["campaign_brief"], state["campaign_id"], state["workflow_stage"]

Responsibilities:
  • Parse and validate user campaign information.
  • Identify the core value proposition.
  • Produce a structured CampaignBrief that all downstream agents read.
  • Set content and visual directives once, so Agents 5 and 8 never need to
    re-interpret the original user input.

Does NOT:
  • Generate copy, hashtags, or images.
  • Call any agent directly.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from core.state import (
    CampaignBrief,
    ContentDirectives,
    MarketPilotState,
    Platform,
    PlatformSpec,
    UserCampaignInput,
    VisualDirectives,
)
from services.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)

# ── Load system prompt once at import time ────────────────────────────────────
_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "agent_01_system_prompt.txt"
SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

# ── Default platform specifications ───────────────────────────────────────────
_PLATFORM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "instagram": {"aspect_ratio": "1:1",  "character_limit": 2200,  "hashtag_limit": 30},
    "facebook":  {"aspect_ratio": "16:9", "character_limit": 63206, "hashtag_limit": 10},
    "linkedin":  {"aspect_ratio": "16:9", "character_limit": 3000,  "hashtag_limit": 5},
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(user_input: UserCampaignInput) -> str:
    platforms      = [p.value for p in user_input.platforms]
    platform_specs = {p: _PLATFORM_DEFAULTS[p] for p in platforms if p in _PLATFORM_DEFAULTS}

    return f"""Create a comprehensive marketing campaign brief from the following campaign details.

CAMPAIGN INPUT:
  Product Name        : {user_input.product_name}
  Product Description : {user_input.product_description}
  Target Audience     : {user_input.target_audience}
  Campaign Goal       : {user_input.campaign_goal.value.upper()}
  Tone                : {user_input.tone.value.upper()}
  Target Platforms    : {", ".join(platforms)}
  Budget Tier         : {user_input.budget_tier.value.upper()}
  Brand Guidelines    : {user_input.brand_guidelines or "None specified"}

Return ONLY the following JSON object — no other text:
{{
  "campaign_id": "",
  "campaign_name": "short memorable campaign name (3-5 words)",
  "campaign_summary": "2-3 sentence overview of the marketing approach and unique angle",
  "key_messages": [
    "specific message 1 about this product",
    "specific message 2",
    "specific message 3",
    "specific message 4"
  ],
  "content_directives": {{
    "primary_cta": "compelling call-to-action phrase (5-10 words)",
    "tone_guidance": "detailed paragraph describing voice, style and tone for writers",
    "word_count_range": [100, 300],
    "forbidden_words": []
  }},
  "visual_directives": {{
    "color_palette": ["color descriptor or hex", "color descriptor or hex"],
    "mood": "visual mood and atmosphere in one evocative sentence",
    "style_preference": "photorealistic",
    "avoid": ["visual element to avoid", "another element to avoid"]
  }},
  "platform_requirements": {json.dumps(platform_specs)},
  "coordination_notes": "Specific guidance for the Content Creator (Agent 5) and Visual Creative Agent (Agent 8) — reference key messages, tone rules, and visual direction they must follow."
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node function
# ─────────────────────────────────────────────────────────────────────────────

def run_campaign_supervisor(state: MarketPilotState) -> Dict[str, Any]:
    """
    LangGraph node: Agent 1 — Campaign Supervisor.

    Returns a partial state update dict. LangGraph merges it into the full state.
    """
    logger.info("═══ Agent 1 (Campaign Supervisor) starting ═══")

    try:
        # ── 1. Parse + validate user input ───────────────────────────────────
        user_input = UserCampaignInput(**state["user_input"])
        logger.info("Product: %s | Goal: %s | Platforms: %s",
                    user_input.product_name,
                    user_input.campaign_goal.value,
                    [p.value for p in user_input.platforms])

        # ── 2. Build prompt ───────────────────────────────────────────────────
        prompt = _build_prompt(user_input)

        # ── 3. Call Gemini ────────────────────────────────────────────────────
        provider = GeminiProvider()
        raw_data = provider.generate_json(prompt, SYSTEM_PROMPT)

        # ── 4. Inject campaign_id if Gemini left it blank ─────────────────────
        if not raw_data.get("campaign_id"):
            raw_data["campaign_id"] = state.get("campaign_id") or str(uuid.uuid4())

        # ── 5. Validate with Pydantic ─────────────────────────────────────────
        brief = CampaignBrief(**raw_data)
        logger.info("Agent 1 complete → Campaign: '%s'", brief.campaign_name)

        return {
            "campaign_brief":  brief.model_dump(),
            "campaign_id":     brief.campaign_id,
            "workflow_stage":  "agent_01_complete",
        }

    except Exception as exc:
        logger.error("Agent 1 FAILED: %s", exc, exc_info=True)
        current_errors: List[str] = list(state.get("errors", []))
        return {
            "workflow_stage": "agent_01_failed",
            "errors":         current_errors + [f"Agent 1 (Campaign Supervisor): {exc}"],
        }
