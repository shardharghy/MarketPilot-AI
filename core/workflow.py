
"""
core/workflow.py — Sequential Python orchestration for MarketPilot AI.

Current workflow:
START → Agent 01 → Agent 05 → Agent 08 → END

Uses ordinary Python instead of LangGraph.
Preserves all three existing agents and the shared state structure.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from agents.agent_01_campaign_supervisor import run_campaign_supervisor
from agents.agent_05_content_creator import run_content_creator
from agents.agent_08_visual_creative import run_visual_creative
from core.state import MarketPilotState, UserCampaignInput

logger = logging.getLogger(__name__)

# Campaign history directory
CAMPAIGN_HISTORY_DIR = Path("data/campaigns")
CAMPAIGN_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _merge_update(
    state: Dict[str, Any],
    update: Dict[str, Any],
) -> None:
    """Merge an agent's partial output into the shared campaign state."""

    for key, value in update.items():
        if key == "errors":
            existing_errors = state.get("errors", [])
            state["errors"] = existing_errors + [
                error for error in value
                if error not in existing_errors
            ]
        else:
            state[key] = value


def run_campaign(user_input: UserCampaignInput) -> Dict[str, Any]:
    """
    Execute all three agents sequentially.

    Agent 01 → Agent 05 → Agent 08

    Returns the final campaign state and saves campaign history.
    """

    campaign_id = str(uuid.uuid4())

    logger.info(
        "Starting campaign | ID: %s | Product: %s",
        campaign_id,
        user_input.product_name,
    )

    # Initialize the shared state.
    state: MarketPilotState = {
        "user_input": user_input.model_dump(),

        # Agent output slots
        "campaign_brief": None,
        "marketing_strategy": None,
        "audience_profile": None,
        "content_plan": None,
        "generated_content": None,
        "seo_data": None,
        "quality_review": None,
        "visual_assets": None,

        # Workflow metadata
        "campaign_id": campaign_id,
        "workflow_stage": "starting",
        "errors": [],
    }

    # Exactly three implemented agents.
    agents = [
        ("Agent 01 — Campaign Supervisor", run_campaign_supervisor),
        ("Agent 05 — Content Creator", run_content_creator),
        ("Agent 08 — Visual Creative", run_visual_creative),
    ]

    for agent_name, agent_function in agents:

        logger.info("Starting %s", agent_name)

        try:
            # Each agent reads the shared state and returns an update.
            update = agent_function(state)

            # Merge the returned update.
            _merge_update(state, update)

            # Stop if an agent reports failure.
            if state.get("workflow_stage", "").endswith("_failed"):
                logger.error("%s failed. Stopping workflow.", agent_name)
                break

        except Exception as exc:
            message = f"{agent_name} failed: {exc}"

            logger.exception(message)

            state["errors"].append(message)
            state["workflow_stage"] = "workflow_failed"
            break

    # Set the final workflow status.
    if state.get("workflow_stage") == "agent_08_complete":
        state["workflow_stage"] = "completed"

    elif state.get("workflow_stage", "").endswith("_failed"):
        state["workflow_stage"] = "failed"

    # Save campaign history.
    _save_campaign_history(
        campaign_id=campaign_id,
        user_input=user_input,
        state=state,
    )

    logger.info(
        "Campaign finished | ID: %s | Status: %s",
        campaign_id,
        state["workflow_stage"],
    )

    return state


def _save_campaign_history(
    campaign_id: str,
    user_input: UserCampaignInput,
    state: Dict[str, Any],
) -> None:
    """Save the campaign results as a JSON file."""

    try:
        record = {
            "campaign_id": campaign_id,
            "user_input": user_input.model_dump(),
            "workflow_stage": state.get("workflow_stage"),
            "errors": state.get("errors", []),
            "campaign_brief": state.get("campaign_brief"),
            "generated_content": state.get("generated_content"),
            "visual_assets": state.get("visual_assets"),
        }

        file_path = CAMPAIGN_HISTORY_DIR / f"{campaign_id}.json"

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(record, file, indent=2, default=str)

        logger.info("Campaign history saved: %s", file_path)

    except Exception as exc:
        logger.warning("Could not save campaign history: %s", exc)


def load_campaign_history(limit: int = 10) -> list:
    """Load recent campaign history, newest first."""

    histories = []

    files = sorted(
        CAMPAIGN_HISTORY_DIR.glob("*.json"),
        key=lambda file: file.stat().st_mtime,
        reverse=True,
    )

    for file_path in files[:limit]:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                histories.append(json.load(file))

        except Exception as exc:
            logger.warning(
                "Could not load campaign history %s: %s",
                file_path,
                exc,
            )

    return histories
