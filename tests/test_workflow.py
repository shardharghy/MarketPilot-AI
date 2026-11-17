
"""
tests/test_workflow.py — Tests for MarketPilot AI sequential workflow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state import (
    BudgetTier,
    CampaignGoal,
    Platform,
    ToneStyle,
    UserCampaignInput,
)


@pytest.fixture
def test_user_input():
    return UserCampaignInput(
        product_name="TestPilot Pro",
        product_description=(
            "A cloud-based project management tool for distributed teams "
            "with AI-powered task prioritization and automated status reports."
        ),
        target_audience="Remote-first startup teams and freelancers aged 22-40",
        campaign_goal=CampaignGoal.AWARENESS,
        tone=ToneStyle.PROFESSIONAL,
        platforms=[Platform.LINKEDIN],
        budget_tier=BudgetTier.STARTER,
    )


def test_workflow_runs_without_langgraph():
    """Verify the workflow exposes the sequential campaign entry point."""

    from core.workflow import run_campaign, load_campaign_history

    assert callable(run_campaign)
    assert callable(load_campaign_history)


def test_workflow_has_exactly_three_agents(monkeypatch, test_user_input):
    """Verify that exactly the three existing agents execute in order."""

    import core.workflow as wf

    execution_order = []

    def fake_agent_01(state):
        execution_order.append("agent_01")
        return {
            "campaign_brief": {"campaign_name": "Test Campaign"},
            "workflow_stage": "agent_01_complete",
        }

    def fake_agent_05(state):
        execution_order.append("agent_05")
        assert state["campaign_brief"] is not None
        return {
            "generated_content": {"headlines": ["Test headline"]},
            "workflow_stage": "agent_05_complete",
        }

    def fake_agent_08(state):
        execution_order.append("agent_08")
        assert state["generated_content"] is not None
        return {
            "visual_assets": {
                "image_prompts": {"linkedin": "Test visual prompt"},
                "generated_images": [],
                "generation_model": "Text-only (Gemini)",
                "metadata": {"mode": "text_only"},
            },
            "workflow_stage": "agent_08_complete",
        }

    monkeypatch.setattr(wf, "run_campaign_supervisor", fake_agent_01)
    monkeypatch.setattr(wf, "run_content_creator", fake_agent_05)
    monkeypatch.setattr(wf, "run_visual_creative", fake_agent_08)

    monkeypatch.setattr(wf, "_save_campaign_history", lambda *args, **kwargs: None)

    result = wf.run_campaign(test_user_input)

    assert execution_order == ["agent_01", "agent_05", "agent_08"]
    assert result["workflow_stage"] == "completed"
    assert result["campaign_brief"] is not None
    assert result["generated_content"] is not None
    assert result["visual_assets"] is not None


def test_workflow_stops_after_agent_failure(monkeypatch, test_user_input):
    """Verify downstream agents do not run after an upstream failure."""

    import core.workflow as wf

    execution_order = []

    def fake_agent_01(state):
        execution_order.append("agent_01")
        return {
            "workflow_stage": "agent_01_failed",
            "errors": ["Test failure"],
        }

    def fake_agent_05(state):
        execution_order.append("agent_05")
        return {}

    def fake_agent_08(state):
        execution_order.append("agent_08")
        return {}

    monkeypatch.setattr(wf, "run_campaign_supervisor", fake_agent_01)
    monkeypatch.setattr(wf, "run_content_creator", fake_agent_05)
    monkeypatch.setattr(wf, "run_visual_creative", fake_agent_08)
    monkeypatch.setattr(wf, "_save_campaign_history", lambda *args, **kwargs: None)

    result = wf.run_campaign(test_user_input)

    assert execution_order == ["agent_01"]
    assert result["workflow_stage"] == "failed"
    assert "Test failure" in result["errors"]


def test_workflow_future_slots_remain_none(monkeypatch, test_user_input):
    """Verify unimplemented agent slots remain empty."""

    import core.workflow as wf

    def fake_agent_01(state):
        return {
            "campaign_brief": {"campaign_name": "Test"},
            "workflow_stage": "agent_01_complete",
        }

    def fake_agent_05(state):
        return {
            "generated_content": {"headlines": ["Test"]},
            "workflow_stage": "agent_05_complete",
        }

    def fake_agent_08(state):
        return {
            "visual_assets": {
                "image_prompts": {"linkedin": "Test prompt"},
                "generated_images": [],
                "generation_model": "Text-only (Gemini)",
                "metadata": {"mode": "text_only"},
            },
            "workflow_stage": "agent_08_complete",
        }

    monkeypatch.setattr(wf, "run_campaign_supervisor", fake_agent_01)
    monkeypatch.setattr(wf, "run_content_creator", fake_agent_05)
    monkeypatch.setattr(wf, "run_visual_creative", fake_agent_08)
    monkeypatch.setattr(wf, "_save_campaign_history", lambda *args, **kwargs: None)

    result = wf.run_campaign(test_user_input)

    assert result["marketing_strategy"] is None
    assert result["audience_profile"] is None
    assert result["content_plan"] is None
    assert result["seo_data"] is None
    assert result["quality_review"] is None


def test_workflow_saves_history_json(monkeypatch, test_user_input, tmp_path):
    """Verify campaign history is saved as valid JSON."""

    import core.workflow as wf

    monkeypatch.setattr(wf, "CAMPAIGN_HISTORY_DIR", tmp_path)

    def fake_agent_01(state):
        return {
            "campaign_brief": {"campaign_name": "History Test"},
            "workflow_stage": "agent_01_complete",
        }

    def fake_agent_05(state):
        return {
            "generated_content": {"headlines": ["History headline"]},
            "workflow_stage": "agent_05_complete",
        }

    def fake_agent_08(state):
        return {
            "visual_assets": {
                "image_prompts": {"linkedin": "History visual prompt"},
                "generated_images": [],
                "generation_model": "Text-only (Gemini)",
                "metadata": {"mode": "text_only"},
            },
            "workflow_stage": "agent_08_complete",
        }

    monkeypatch.setattr(wf, "run_campaign_supervisor", fake_agent_01)
    monkeypatch.setattr(wf, "run_content_creator", fake_agent_05)
    monkeypatch.setattr(wf, "run_visual_creative", fake_agent_08)

    result = wf.run_campaign(test_user_input)

    history_file = tmp_path / f"{result['campaign_id']}.json"

    assert history_file.exists()

    with open(history_file, encoding="utf-8") as file:
        data = json.load(file)

    assert data["campaign_id"] == result["campaign_id"]
    assert "user_input" in data
    assert data["campaign_brief"]["campaign_name"] == "History Test"


def test_visual_assets_are_text_only(monkeypatch, test_user_input):
    """Verify the workflow supports visual prompts without generated images."""

    import core.workflow as wf

    def fake_agent_01(state):
        return {
            "campaign_brief": {"campaign_name": "Visual Test"},
            "workflow_stage": "agent_01_complete",
        }

    def fake_agent_05(state):
        return {
            "generated_content": {"headlines": ["Visual headline"]},
            "workflow_stage": "agent_05_complete",
        }

    def fake_agent_08(state):
        return {
            "visual_assets": {
                "image_prompts": {
                    "linkedin": "Professional clean visual concept."
                },
                "generated_images": [],
                "generation_model": "Text-only (Gemini)",
                "metadata": {
                    "mode": "text_only",
                    "images_generated": False,
                },
            },
            "workflow_stage": "agent_08_complete",
        }

    monkeypatch.setattr(wf, "run_campaign_supervisor", fake_agent_01)
    monkeypatch.setattr(wf, "run_content_creator", fake_agent_05)
    monkeypatch.setattr(wf, "run_visual_creative", fake_agent_08)
    monkeypatch.setattr(wf, "_save_campaign_history", lambda *args, **kwargs: None)

    result = wf.run_campaign(test_user_input)

    assets = result["visual_assets"]

    assert assets["image_prompts"]["linkedin"]
    assert assets["generated_images"] == []
    assert assets["metadata"]["mode"] == "text_only"
    assert assets["metadata"]["images_generated"] is False
