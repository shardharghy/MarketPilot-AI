"""
tests/test_agent_05.py — Unit tests for Agent 5: Content Creator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

REAL_API = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Requires GEMINI_API_KEY environment variable",
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — a pre-built state with a realistic CampaignBrief
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def state_with_brief():
    return {
        "user_input": {
            "product_name":        "EcoBreeze Air Purifier",
            "product_description": "Smart HEPA air purifier removing 99.9% of pollutants",
            "target_audience":     "Urban professionals aged 25-45",
            "campaign_goal":       "awareness",
            "tone":                "inspirational",
            "platforms":           ["instagram", "facebook"],
            "budget_tier":         "growth",
            "brand_guidelines":    None,
        },
        "campaign_brief": {
            "campaign_id":      "test-brief-001",
            "campaign_name":    "Breathe Better Campaign",
            "campaign_summary": (
                "A campaign promoting cleaner indoor air for healthier, "
                "more productive living environments."
            ),
            "key_messages": [
                "Breathe cleaner air every single day",
                "Advanced 3-stage HEPA filtration technology",
                "Real-time air quality visibility at your fingertips",
            ],
            "content_directives": {
                "primary_cta":      "Shop Now and Breathe Better",
                "tone_guidance":    "Inspirational and health-focused. Empowering, never preachy.",
                "word_count_range": [100, 300],
                "forbidden_words":  [],
            },
            "visual_directives": {
                "color_palette":    ["forest green", "clean white", "sky blue"],
                "mood":             "Clean, fresh, optimistic — the feeling of a first deep breath outdoors",
                "style_preference": "photorealistic",
                "avoid":            ["dark colors", "cluttered scenes", "clinical sterile look"],
            },
            "platform_requirements": {
                "instagram": {"aspect_ratio": "1:1",  "character_limit": 2200,  "hashtag_limit": 30},
                "facebook":  {"aspect_ratio": "16:9", "character_limit": 63206, "hashtag_limit": 10},
            },
            "coordination_notes": "Focus on clean air benefits and lifestyle improvement.",
            "created_at": "2026-11-17T00:00:00+00:00",
        },
        "marketing_strategy": None,
        "audience_profile":   None,
        "content_plan":       None,
        "generated_content":  None,
        "seo_data":           None,
        "quality_review":     None,
        "visual_assets":      None,
        "campaign_id":        "test-agent05-001",
        "workflow_stage":     "agent_01_complete",
        "errors":             [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_05_skips_when_brief_missing():
    """Agent 5 must skip gracefully when campaign_brief is None."""
    from agents.agent_05_content_creator import run_content_creator
    state = {
        "user_input": {
            "product_name": "X", "product_description": "Y " * 5,
            "target_audience": "Z Z Z",
            "campaign_goal": "awareness", "tone": "casual",
            "platforms": ["instagram"], "budget_tier": "starter",
            "brand_guidelines": None,
        },
        "campaign_brief":     None,
        "marketing_strategy": None, "audience_profile": None,
        "content_plan":       None, "generated_content": None,
        "seo_data":           None, "quality_review":    None,
        "visual_assets":      None,
        "campaign_id": "test", "workflow_stage": "agent_01_failed",
        "errors": ["Agent 1 failed"],
    }
    result = run_content_creator(state)
    assert result.get("workflow_stage") in ("agent_05_skipped", "agent_05_failed")


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────────────────

@REAL_API
def test_agent_05_returns_generated_content(state_with_brief):
    from agents.agent_05_content_creator import run_content_creator
    result = run_content_creator(state_with_brief)

    assert result.get("workflow_stage") == "agent_05_complete", (
        f"Stage: {result.get('workflow_stage')}  Errors: {result.get('errors')}"
    )
    assert result.get("generated_content") is not None


@REAL_API
def test_agent_05_generates_two_distinct_variations(state_with_brief):
    from agents.agent_05_content_creator import run_content_creator
    result  = run_content_creator(state_with_brief)
    content = result["generated_content"]

    for platform in ("instagram", "facebook"):
        if platform in content and content[platform]:
            va = content[platform]["variation_a"]["caption"]
            vb = content[platform]["variation_b"]["caption"]
            assert va, f"{platform} variation_a caption must not be empty"
            assert vb, f"{platform} variation_b caption must not be empty"
            assert va != vb, f"{platform}: variation_a and variation_b must differ"


@REAL_API
def test_agent_05_includes_hashtags(state_with_brief):
    from agents.agent_05_content_creator import run_content_creator
    result  = run_content_creator(state_with_brief)
    content = result["generated_content"]

    if "instagram" in content and content["instagram"]:
        hashtags_a = content["instagram"]["variation_a"].get("hashtags", [])
        assert len(hashtags_a) > 0, "Instagram variation_a must have hashtags"


@REAL_API
def test_agent_05_produces_headlines(state_with_brief):
    from agents.agent_05_content_creator import run_content_creator
    result  = run_content_creator(state_with_brief)
    content = result["generated_content"]

    assert len(content.get("headlines", [])) >= 2, "Need at least 2 headlines"


@REAL_API
def test_agent_05_only_includes_requested_platforms(state_with_brief):
    """Agent 5 must not generate content for unrequested platforms."""
    from agents.agent_05_content_creator import run_content_creator
    result  = run_content_creator(state_with_brief)
    content = result["generated_content"]

    # linkedin was NOT in user_input.platforms
    assert content.get("linkedin") is None, (
        "linkedin content should not be generated when not requested"
    )
