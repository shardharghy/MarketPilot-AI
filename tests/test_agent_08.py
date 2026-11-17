
"""
tests/test_agent_08.py — Unit tests for Agent 08: Visual Creative Agent.

Agent 08 currently generates text-only visual concepts and image prompts.
It does not generate or save PNG files.
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


@pytest.fixture
def state_with_brief_and_content():
    return {
        "user_input": {
            "product_name": "EcoBreeze Air Purifier",
            "product_description": "Smart HEPA air purifier",
            "target_audience": "Urban professionals 25-45",
            "campaign_goal": "awareness",
            "tone": "inspirational",
            "platforms": ["instagram"],
            "budget_tier": "growth",
            "brand_guidelines": None,
        },
        "campaign_brief": {
            "campaign_id": "test-brief-visual-001",
            "campaign_name": "Breathe Better Campaign",
            "campaign_summary": "Campaign promoting clean indoor air.",
            "key_messages": ["Breathe cleaner air every day"],
            "content_directives": {
                "primary_cta": "Shop Now",
                "tone_guidance": "Inspirational",
                "word_count_range": [100, 300],
                "forbidden_words": [],
            },
            "visual_directives": {
                "color_palette": ["forest green", "white"],
                "mood": "Clean, fresh, optimistic",
                "style_preference": "photorealistic",
                "avoid": ["dark backgrounds", "text overlays"],
            },
            "platform_requirements": {
                "instagram": {
                    "aspect_ratio": "1:1",
                    "character_limit": 2200,
                    "hashtag_limit": 30,
                },
            },
            "coordination_notes": "Focus on clean air benefits.",
            "created_at": "2026-11-17T00:00:00+00:00",
        },
        "marketing_strategy": None,
        "audience_profile": None,
        "content_plan": None,
        "generated_content": {
            "instagram": {
                "variation_a": {
                    "caption": "Breathe better today!",
                    "hashtags": ["airpurifier", "cleanair"],
                    "cta": "Shop Now",
                    "char_count": 20,
                },
                "variation_b": {
                    "caption": "Your lungs deserve the best.",
                    "hashtags": ["healthyliving", "pureair"],
                    "cta": "Learn More",
                    "char_count": 28,
                },
            },
            "headlines": ["Breathe Better with EcoBreeze"],
            "generated_at": "2026-11-17T00:00:00+00:00",
            "model_used": "gemini-3.8-flash",
        },
        "seo_data": None,
        "quality_review": None,
        "visual_assets": None,
        "campaign_id": "test-agent08-001",
        "workflow_stage": "agent_05_complete",
        "errors": [],
    }


def test_agent_08_skips_when_brief_missing():
    from agents.agent_08_visual_creative import run_visual_creative

    state = {
        "user_input": {
            "product_name": "X",
            "product_description": "Y " * 5,
            "target_audience": "Z Z Z",
            "campaign_goal": "awareness",
            "tone": "casual",
            "platforms": ["instagram"],
            "budget_tier": "starter",
            "brand_guidelines": None,
        },
        "campaign_brief": None,
        "generated_content": None,
        "errors": ["Agent 1 failed"],
    }

    result = run_visual_creative(state)

    assert result["workflow_stage"] == "agent_08_skipped"
    assert "visual_assets" not in result


def test_agent_08_generates_text_only_visual_assets(
    state_with_brief_and_content,
    monkeypatch,
):
    from agents import agent_08_visual_creative

    class FakeGeminiProvider:
        def generate_json(self, prompt, system_prompt):
            return {
                "image_prompts": {
                    "instagram": (
                        "A bright, minimalist product photograph of a modern "
                        "air purifier in a fresh, naturally lit living room. "
                        "Forest green plants and clean white surroundings "
                        "create a calm and refreshing atmosphere."
                    )
                },
                "style_summary": "Fresh, minimalist and optimistic visual direction.",
            }

    monkeypatch.setattr(
        agent_08_visual_creative,
        "GeminiProvider",
        FakeGeminiProvider,
    )

    result = agent_08_visual_creative.run_visual_creative(
        state_with_brief_and_content
    )

    assert result["workflow_stage"] == "agent_08_complete"

    assets = result["visual_assets"]

    assert assets is not None
    assert "instagram" in assets["image_prompts"]
    assert len(assets["image_prompts"]["instagram"]) > 50

    assert assets["generated_images"] == []
    assert assets["generation_model"] == "Text-only (Gemini)"

    assert assets["metadata"]["mode"] == "text_only"
    assert assets["metadata"]["images_generated"] is False
    assert assets["metadata"]["style_summary"]


def test_agent_08_only_generates_requested_platforms(
    state_with_brief_and_content,
    monkeypatch,
):
    from agents import agent_08_visual_creative

    class FakeGeminiProvider:
        def generate_json(self, prompt, system_prompt):
            return {
                "image_prompts": {
                    "instagram": "Instagram visual concept " * 5,
                    "facebook": "Facebook visual concept " * 5,
                    "linkedin": "LinkedIn visual concept " * 5,
                },
                "style_summary": "Professional visual direction.",
            }

    monkeypatch.setattr(
        agent_08_visual_creative,
        "GeminiProvider",
        FakeGeminiProvider,
    )

    result = agent_08_visual_creative.run_visual_creative(
        state_with_brief_and_content
    )

    assert result["workflow_stage"] == "agent_08_complete"
    assert list(result["visual_assets"]["image_prompts"].keys()) == [
        "instagram"
    ]


@REAL_API
def test_agent_08_generates_image_prompts_with_gemini(
    state_with_brief_and_content,
):
    from agents.agent_08_visual_creative import run_visual_creative

    result = run_visual_creative(state_with_brief_and_content)

    assert result["workflow_stage"] == "agent_08_complete", (
        f"Stage: {result.get('workflow_stage')} "
        f"Errors: {result.get('errors')}"
    )

    assets = result["visual_assets"]

    assert assets is not None
    assert assets["image_prompts"]
    assert assets["generated_images"] == []
    assert assets["metadata"]["mode"] == "text_only"

    for platform, prompt in assets["image_prompts"].items():
        assert len(prompt) > 50, (
            f"Prompt for {platform} is too short."
        )
