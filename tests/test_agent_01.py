"""
tests/test_agent_01.py — Unit tests for Agent 1: Campaign Supervisor.

Integration tests (marked with REAL_API) make actual Gemini API calls.
They are skipped automatically if GEMINI_API_KEY is not set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state import BudgetTier, CampaignGoal, Platform, ToneStyle, UserCampaignInput

# ── Skip marker ───────────────────────────────────────────────────────────────
REAL_API = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Requires GEMINI_API_KEY environment variable",
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_user_input():
    return UserCampaignInput(
        product_name="EcoBreeze Air Purifier",
        product_description=(
            "A smart air purifier with 3-stage HEPA filtration that removes 99.9% "
            "of airborne pollutants including PM2.5, allergens, and VOCs. "
            "Features real-time air quality monitoring via a companion app."
        ),
        target_audience="Health-conscious urban professionals aged 25–45",
        campaign_goal=CampaignGoal.AWARENESS,
        tone=ToneStyle.INSPIRATIONAL,
        platforms=[Platform.INSTAGRAM, Platform.LINKEDIN],
        budget_tier=BudgetTier.GROWTH,
    )


@pytest.fixture
def sample_state(sample_user_input):
    return {
        "user_input":          sample_user_input.model_dump(),
        "campaign_brief":      None,
        "marketing_strategy":  None,
        "audience_profile":    None,
        "content_plan":        None,
        "generated_content":   None,
        "seo_data":            None,
        "quality_review":      None,
        "visual_assets":       None,
        "campaign_id":         "test-agent01-001",
        "workflow_stage":      "starting",
        "errors":              [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unit test (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_01_raises_on_missing_api_key(monkeypatch, sample_state):
    """Agent 1 must surface EnvironmentError when API key is absent."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from services.gemini_provider import GeminiProvider
    GeminiProvider.reset_singleton()

    from agents.agent_01_campaign_supervisor import run_campaign_supervisor
    result = run_campaign_supervisor(sample_state)

    # Must return an error state, not raise an unhandled exception
    assert "errors" in result or result.get("workflow_stage") == "agent_01_failed"


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests (require real API key)
# ─────────────────────────────────────────────────────────────────────────────

@REAL_API
def test_agent_01_returns_campaign_brief(sample_state):
    from agents.agent_01_campaign_supervisor import run_campaign_supervisor
    result = run_campaign_supervisor(sample_state)

    assert result.get("workflow_stage") == "agent_01_complete", (
        f"Expected agent_01_complete, got {result.get('workflow_stage')}. "
        f"Errors: {result.get('errors')}"
    )
    assert result.get("campaign_brief") is not None


@REAL_API
def test_agent_01_brief_has_required_fields(sample_state):
    from agents.agent_01_campaign_supervisor import run_campaign_supervisor
    result    = run_campaign_supervisor(sample_state)
    brief     = result["campaign_brief"]

    assert brief.get("campaign_name"), "campaign_name must not be empty"
    assert brief.get("campaign_summary"), "campaign_summary must not be empty"
    assert len(brief.get("key_messages", [])) >= 2, "Need at least 2 key messages"
    assert brief.get("content_directives", {}).get("primary_cta"), "primary_cta required"
    assert brief.get("visual_directives", {}).get("mood"), "mood required"
    assert brief.get("platform_requirements"), "platform_requirements must be populated"


@REAL_API
def test_agent_01_includes_requested_platforms(sample_state):
    from agents.agent_01_campaign_supervisor import run_campaign_supervisor
    result = run_campaign_supervisor(sample_state)
    brief  = result["campaign_brief"]
    pr     = brief.get("platform_requirements", {})

    # State has instagram + linkedin requested
    assert any(p in pr for p in ("instagram", "linkedin")), (
        f"Expected instagram or linkedin in platform_requirements, got: {list(pr.keys())}"
    )


@REAL_API
def test_agent_01_sets_campaign_id(sample_state):
    from agents.agent_01_campaign_supervisor import run_campaign_supervisor
    result = run_campaign_supervisor(sample_state)

    assert result.get("campaign_id"), "campaign_id must be set in returned state"
    assert result["campaign_brief"].get("campaign_id"), "brief.campaign_id must be populated"
