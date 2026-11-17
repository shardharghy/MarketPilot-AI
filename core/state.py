"""
core/state.py — Shared typed state for MarketPilot AI.

Defines:
  • All Pydantic schemas for every agent (implemented and future).
  • MarketPilotState — the LangGraph TypedDict that travels through the graph.

DESIGN RULE: All 8 agent output slots are pre-declared here. Future agents only
need to populate their slot — no existing agent or schema changes are needed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class CampaignGoal(str, Enum):
    AWARENESS  = "awareness"
    CONVERSION = "conversion"
    ENGAGEMENT = "engagement"
    RETENTION  = "retention"


class ToneStyle(str, Enum):
    PROFESSIONAL  = "professional"
    CASUAL        = "casual"
    PLAYFUL       = "playful"
    INSPIRATIONAL = "inspirational"
    URGENT        = "urgent"


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    FACEBOOK  = "facebook"
    LINKEDIN  = "linkedin"


class BudgetTier(str, Enum):
    STARTER    = "starter"
    GROWTH     = "growth"
    ENTERPRISE = "enterprise"


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point: User Campaign Input
# ─────────────────────────────────────────────────────────────────────────────

class UserCampaignInput(BaseModel):
    """Raw input from the user / Streamlit form."""
    product_name:        str           = Field(..., min_length=1,  max_length=200)
    product_description: str           = Field(..., min_length=10, max_length=1000)
    target_audience:     str           = Field(..., min_length=5,  max_length=300)
    campaign_goal:       CampaignGoal
    tone:                ToneStyle
    platforms:           List[Platform] = Field(..., min_length=1)
    budget_tier:         BudgetTier
    brand_guidelines:    Optional[str] = Field(None, max_length=500)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 Output: CampaignBrief
# ─────────────────────────────────────────────────────────────────────────────

class ContentDirectives(BaseModel):
    primary_cta:      str
    tone_guidance:    str
    word_count_range: List[int]  = Field(default=[100, 300])  # [min, max]
    forbidden_words:  List[str]  = Field(default_factory=list)


class VisualDirectives(BaseModel):
    color_palette:    List[str]  = Field(default_factory=list)
    mood:             str
    style_preference: str        = "photorealistic"
    avoid:            List[str]  = Field(default_factory=list)


class PlatformSpec(BaseModel):
    aspect_ratio:    str
    character_limit: int
    hashtag_limit:   int


class CampaignBrief(BaseModel):
    """Produced by Agent 1. Read by Agents 5 and 8."""
    campaign_id:           str = Field(default_factory=lambda: str(uuid.uuid4()))
    campaign_name:         str
    campaign_summary:      str
    key_messages:          List[str]              = Field(..., min_length=2)
    content_directives:    ContentDirectives
    visual_directives:     VisualDirectives
    platform_requirements: Dict[str, PlatformSpec]  # keyed by Platform.value
    coordination_notes:    str
    created_at:            str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 Output: GeneratedContent
# ─────────────────────────────────────────────────────────────────────────────

class PostContent(BaseModel):
    caption:    str
    hashtags:   List[str] = Field(default_factory=list)
    cta:        str       = ""
    char_count: int       = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            object.__setattr__(self, "char_count", len(self.caption))


class ContentVariations(BaseModel):
    variation_a: PostContent
    variation_b: PostContent


class GeneratedContent(BaseModel):
    """Produced by Agent 5. Read by Agent 8 (for context) and the UI."""
    instagram:    Optional[ContentVariations] = None
    facebook:     Optional[ContentVariations] = None
    linkedin:     Optional[ContentVariations] = None
    headlines:    List[str] = Field(default_factory=list)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_used: str = "gemini-2.0-flash"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 8 Output: VisualAssets
# ─────────────────────────────────────────────────────────────────────────────

class ImageAsset(BaseModel):
    platform:     str
    file_path:    str
    aspect_ratio: str
    style:        str
    width:        int
    height:       int
    prompt_used:  str = ""
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class VisualAssets(BaseModel):
    """Produced by Agent 8. Contains paths to saved PNG files."""
    image_prompts:    Dict[str, str]      # platform → prompt string
    generated_images: List[ImageAsset]
    generation_model: str
    metadata:         Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Future Agent Stub Schemas  (pre-declared; logic NOT implemented)
# ─────────────────────────────────────────────────────────────────────────────

class MarketingStrategy(BaseModel):
    """Agent 2 — Marketing Strategist. Slot reserved; not yet implemented."""
    positioning:             str            = ""
    competitive_advantages:  List[str]      = Field(default_factory=list)
    campaign_pillars:        List[str]      = Field(default_factory=list)
    budget_allocation:       Dict[str, float] = Field(default_factory=dict)


class AudienceProfile(BaseModel):
    """Agent 3 — Audience Researcher. Slot reserved; not yet implemented."""
    personas:          List[Dict[str, Any]] = Field(default_factory=list)
    segments:          List[str]            = Field(default_factory=list)
    channel_affinities: Dict[str, float]   = Field(default_factory=dict)


class ContentPlan(BaseModel):
    """Agent 4 — Content Planner. Slot reserved; not yet implemented."""
    topics:      List[str]       = Field(default_factory=list)
    content_mix: Dict[str, int]  = Field(default_factory=dict)
    schedule:    Dict[str, Any]  = Field(default_factory=dict)


class SEOData(BaseModel):
    """Agent 6 — SEO Optimizer. Slot reserved; not yet implemented."""
    keywords:          List[str]        = Field(default_factory=list)
    meta_descriptions: Dict[str, str]   = Field(default_factory=dict)
    search_intent_tags: List[str]       = Field(default_factory=list)


class QualityReview(BaseModel):
    """Agent 7 — Quality Reviewer. Slot reserved; not yet implemented."""
    scores:          Dict[str, float] = Field(default_factory=dict)
    approved:        bool             = False
    revision_flags:  List[str]        = Field(default_factory=list)
    reviewed_at:     str              = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Shared State
# ─────────────────────────────────────────────────────────────────────────────

class MarketPilotState(TypedDict):
    """
    The single shared state dict that flows through the LangGraph graph.

    Convention:
      • user_input and all agent outputs are stored as plain dicts
        (result of model.model_dump()) for maximum LangGraph compatibility.
      • Each agent validates its upstream input with the matching Pydantic model.
      • Future agent slots default to None in the initial state; adding a future
        agent never requires changing this class.
    """

    # ── Entry point ───────────────────────────────────────────────────────────
    user_input:          Dict[str, Any]           # UserCampaignInput.model_dump()

    # ── Agent 1: Campaign Supervisor [IMPLEMENTED] ────────────────────────────
    campaign_brief:      Optional[Dict[str, Any]]

    # ── Agent 2: Marketing Strategist [FUTURE SLOT] ───────────────────────────
    marketing_strategy:  Optional[Dict[str, Any]]

    # ── Agent 3: Audience Researcher [FUTURE SLOT] ────────────────────────────
    audience_profile:    Optional[Dict[str, Any]]

    # ── Agent 4: Content Planner [FUTURE SLOT] ────────────────────────────────
    content_plan:        Optional[Dict[str, Any]]

    # ── Agent 5: Content Creator [IMPLEMENTED] ───────────────────────────────
    generated_content:   Optional[Dict[str, Any]]

    # ── Agent 6: SEO Optimizer [FUTURE SLOT] ─────────────────────────────────
    seo_data:            Optional[Dict[str, Any]]

    # ── Agent 7: Quality Reviewer [FUTURE SLOT] ──────────────────────────────
    quality_review:      Optional[Dict[str, Any]]

    # ── Agent 8: Visual Creative Agent [IMPLEMENTED] ─────────────────────────
    visual_assets:       Optional[Dict[str, Any]]

    # ── Workflow metadata ─────────────────────────────────────────────────────
    campaign_id:         str
    workflow_stage:      str
    errors:              List[str]
