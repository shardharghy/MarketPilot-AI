"""
ui/app.py — MarketPilot AI Streamlit Frontend.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.state import BudgetTier, CampaignGoal, Platform, ToneStyle, UserCampaignInput
from core.workflow import load_campaign_history, run_campaign

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MarketPilot AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Tighten top padding */
.block-container { padding-top: 1.5rem; }

/* Agent section badges */
.agent-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.badge-01 { background: #e8f0fe; color: #1a73e8; border: 1px solid #c5d9fb; }
.badge-05 { background: #e6f4ea; color: #1e8e3e; border: 1px solid #bde0c6; }
.badge-08 { background: #fef7e0; color: #b06000; border: 1px solid #fce09b; }

/* Variation cards */
.var-label { font-size: 0.75rem; font-weight: 700; color: #5f6368;
             text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚀 MarketPilot AI")
    st.caption("Multi-Agent Generative AI · B.Tech Project")
    st.divider()

    # API key status
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "your_gemini_api_key_here":
        st.success("✅ Gemini API Connected")
    else:
        st.error("❌ GEMINI_API_KEY not configured")
        st.code("# .env\nGEMINI_API_KEY=your_key_here", language="bash")
        st.markdown("Get a free key: [aistudio.google.com](https://aistudio.google.com/apikey)")

    st.divider()

    # Agent pipeline visualization
    st.markdown("**🔁 Active Agent Pipeline**")
    st.markdown(
        "```\n"
        "Agent 1 ● Campaign Supervisor\n"
        "       ↓\n"
        "Agent 5 ● Content Creator\n"
        "       ↓\n"
        "Agent 8 ● Visual Creative\n"
        "```"
    )

    st.divider()

    # Campaign history
    st.markdown("**📋 Recent Campaigns**")
    histories = load_campaign_history(limit=5)
    if histories:
        for h in histories:
            brief = h.get("campaign_brief") or {}
            name  = brief.get("campaign_name", "Unnamed")
            cid   = h.get("campaign_id", "")[:8]
            stage = h.get("workflow_stage", "unknown")
            icon = "✅" if stage == "completed" else "⚠️"
            st.markdown(f"{icon} **{name}**")
            st.caption(f"ID {cid}… · {stage.replace('_', ' ')}")
    else:
        st.caption("No campaigns yet — create your first one!")


# ─────────────────────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🚀 MarketPilot AI")
st.markdown("*Generate professional marketing campaigns with a 3-agent AI pipeline*")
st.divider()

tab_create, tab_results = st.tabs(["✨ Create Campaign", "📊 Results"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Create Campaign
# ─────────────────────────────────────────────────────────────────────────────

with tab_create:
    with st.form("campaign_form", clear_on_submit=False):
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.subheader("📦 Product Details")
            product_name = st.text_input(
                "Product Name *",
                placeholder="e.g. EcoBreeze Air Purifier",
            )
            product_description = st.text_area(
                "Product Description *",
                placeholder="Describe your product in detail — features, benefits, what makes it unique...",
                height=130,
            )
            target_audience = st.text_input(
                "Target Audience *",
                placeholder="e.g. Health-conscious urban professionals aged 25–45",
            )
            brand_guidelines = st.text_area(
                "Brand Guidelines (optional)",
                placeholder="e.g. Brand colors: forest green and white. Avoid: cluttered layouts, dark themes.",
                height=80,
            )

        with col_right:
            st.subheader("🎯 Campaign Settings")
            campaign_goal = st.selectbox(
                "Campaign Goal *",
                options=[g.value for g in CampaignGoal],
                format_func=lambda x: x.replace("_", " ").title(),
            )
            tone = st.selectbox(
                "Brand Tone *",
                options=[t.value for t in ToneStyle],
                format_func=lambda x: x.replace("_", " ").title(),
            )
            platforms = st.multiselect(
                "Target Platforms *",
                options=[p.value for p in Platform],
                default=["instagram"],
                format_func=str.title,
            )
            budget_tier = st.selectbox(
                "Budget Tier *",
                options=[b.value for b in BudgetTier],
                format_func=lambda x: x.replace("_", " ").title(),
            )

            st.markdown("---")
            st.info(
    "⏱️ **Estimated time:** 30–60 seconds\n\n"
    "🤖 **Pipeline:** Supervisor → Content → Visual\n\n"
    "🎨 Text-based visual concepts and image-generation prompts"
)

        submitted = st.form_submit_button(
            "🚀 Generate Marketing Campaign",
            type="primary",
            use_container_width=True,
        )

    # ── Validation & execution ────────────────────────────────────────────────
    if submitted:
        form_errors = []
        if not product_name.strip():
            form_errors.append("Product name is required.")
        if not product_description.strip() or len(product_description.strip()) < 10:
            form_errors.append("Product description must be at least 10 characters.")
        if not target_audience.strip():
            form_errors.append("Target audience is required.")
        if not platforms:
            form_errors.append("Select at least one platform.")
        if not api_key or api_key == "your_gemini_api_key_here":
            form_errors.append("Add your GEMINI_API_KEY to the .env file before running.")

        if form_errors:
            for err in form_errors:
                st.error(f"❌ {err}")
        else:
            try:
                user_input = UserCampaignInput(
                    product_name=product_name.strip(),
                    product_description=product_description.strip(),
                    target_audience=target_audience.strip(),
                    campaign_goal=CampaignGoal(campaign_goal),
                    tone=ToneStyle(tone),
                    platforms=[Platform(p) for p in platforms],
                    budget_tier=BudgetTier(budget_tier),
                    brand_guidelines=brand_guidelines.strip() or None,
                )
            except Exception as e:
                st.error(f"❌ Input validation error: {e}")
                st.stop()

            with st.spinner(
                "🤖 Running 3-agent pipeline…  "
                "Agent 1 (Supervisor) → Agent 5 (Content) → Agent 8 (Visual Creative)  "
                "— this takes 30–60 seconds"
            ):
                try:
                    result = run_campaign(user_input)
                    st.session_state["last_result"] = result
                    st.session_state["last_input"]  = user_input

                    if result.get("errors"):
                        st.warning(
                            "⚠️ Campaign completed with warnings:\n"
                            + "\n".join(f"• {e}" for e in result["errors"])
                        )
                    else:
                        st.success(
                            "✅ Campaign generated!  "
                            "Open the **📊 Results** tab to view content and images."
                        )
                except Exception as e:
                    st.error(f"❌ Workflow error: {e}")
                    st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Results
# ─────────────────────────────────────────────────────────────────────────────

with tab_results:
    if "last_result" not in st.session_state:
        st.info("👆 Create a campaign in the **✨ Create Campaign** tab to see results here.")
        st.stop()

    result = st.session_state["last_result"]

    # ── Agent 1: Campaign Brief ───────────────────────────────────────────────
    st.markdown(
        '<div class="agent-badge badge-01">🧠 Agent 1 — Campaign Supervisor</div>',
        unsafe_allow_html=True,
    )

    if brief := result.get("campaign_brief"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Campaign", brief.get("campaign_name", "—"))
        m2.metric("Status",   result.get("workflow_stage", "—").replace("_", " "))
        m3.metric("ID",       (result.get("campaign_id") or "")[:12] + "…")

        with st.expander("📋 Full Campaign Brief", expanded=False):
            st.markdown(f"**Summary:** {brief.get('campaign_summary', '')}")
            st.markdown("**Key Messages:**")
            for msg in brief.get("key_messages", []):
                st.markdown(f"- {msg}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Content Directives**")
                cd = brief.get("content_directives", {})
                st.markdown(f"- **CTA:** {cd.get('primary_cta', '')}")
                st.markdown(f"- **Tone:** {cd.get('tone_guidance', '')}")
            with c2:
                st.markdown("**Visual Directives**")
                vd = brief.get("visual_directives", {})
                st.markdown(f"- **Mood:** {vd.get('mood', '')}")
                st.markdown(f"- **Style:** {vd.get('style_preference', '')}")
                if vd.get("color_palette"):
                    st.markdown(f"- **Colors:** {', '.join(vd['color_palette'])}")

            if brief.get("coordination_notes"):
                st.markdown(f"**Coordination Notes:** {brief['coordination_notes']}")
    else:
        st.error("⚠️ Campaign brief was not generated. Check the errors below.")

    st.divider()

    # ── Agent 5: Generated Content ────────────────────────────────────────────
    st.markdown(
        '<div class="agent-badge badge-05">✍️ Agent 5 — Content Creator</div>',
        unsafe_allow_html=True,
    )

    if content := result.get("generated_content"):
        # Headlines
        if headlines := content.get("headlines"):
            with st.expander("📰 Campaign Headlines", expanded=True):
                cols = st.columns(2)
                for i, headline in enumerate(headlines):
                    cols[i % 2].markdown(f"**{i+1}.** {headline}")

        # Platform content tabs
        platform_map = {
            "instagram": "📸 Instagram",
            "facebook":  "👥 Facebook",
            "linkedin":  "💼 LinkedIn",
        }
        active_platforms = [
            (k, v) for k, v in platform_map.items()
            if k in content and content[k]
        ]

        if active_platforms:
            platform_tabs = st.tabs([label for _, label in active_platforms])
            for tab, (platform, _) in zip(platform_tabs, active_platforms):
                with tab:
                    pdata = content[platform]
                    col_a, col_b = st.columns(2, gap="medium")

                    with col_a:
                        st.markdown('<div class="var-label">📌 Variation A — Direct & Benefit-Focused</div>',
                                    unsafe_allow_html=True)
                        va = pdata.get("variation_a", {})
                        st.text_area(
                            "Caption",
                            value=va.get("caption", ""),
                            height=160,
                            key=f"{platform}_a_caption",
                            disabled=True,
                        )
                        if va.get("hashtags"):
                            st.code(
                                " ".join(f"#{h}" for h in va["hashtags"]),
                                language=None,
                            )
                        if va.get("cta"):
                            st.markdown(f"**CTA →** {va['cta']}")
                        st.caption(f"{va.get('char_count', 0)} characters")

                    with col_b:
                        st.markdown('<div class="var-label">💡 Variation B — Story-Driven & Emotional</div>',
                                    unsafe_allow_html=True)
                        vb = pdata.get("variation_b", {})
                        st.text_area(
                            "Caption",
                            value=vb.get("caption", ""),
                            height=160,
                            key=f"{platform}_b_caption",
                            disabled=True,
                        )
                        if vb.get("hashtags"):
                            st.code(
                                " ".join(f"#{h}" for h in vb["hashtags"]),
                                language=None,
                            )
                        if vb.get("cta"):
                            st.markdown(f"**CTA →** {vb['cta']}")
                        st.caption(f"{vb.get('char_count', 0)} characters")
        else:
            st.warning("No platform content was generated.")
    else:
        st.error("⚠️ Content was not generated. Check errors below.")

    st.divider()

    # ── Agent 8: Visual Creative ───────────────────────────────────────────────
    st.markdown(
        '<div class="agent-badge badge-08">🎨 Agent 8 — Visual Creative Agent</div>',
        unsafe_allow_html=True,
    )

    if visual := result.get("visual_assets"):
        metadata = visual.get("metadata", {})
        image_prompts = visual.get("image_prompts", {})

        st.success("Visual concepts and image-generation prompts created.")

        if metadata.get("style_summary"):
            st.markdown("**🎨 Overall Visual Direction**")
            st.write(metadata["style_summary"])

        if metadata.get("style"):
            st.caption(f"Preferred visual style: {metadata['style']}")

        st.caption("Mode: Text-only visual creative generation")

        if image_prompts:
            st.markdown("### 🖼️ Platform-specific Visual Prompts")

            for platform, prompt_text in image_prompts.items():
                with st.expander(
                    f"🎨 {platform.title()} Visual Concept",
                    expanded=True,
                ):
                    st.text_area(
                        "Image-generation prompt",
                        value=prompt_text,
                        height=220,
                        key=f"visual_prompt_{platform}",
                        disabled=True,
                    )

                    st.download_button(
                        label=f"⬇️ Download {platform.title()} Prompt",
                        data=prompt_text,
                        file_name=f"{platform}_visual_prompt.txt",
                        mime="text/plain",
                        key=f"download_prompt_{platform}",
                    )
        else:
            st.warning("No visual prompts were generated.")

        st.info(
            "Agent 08 currently generates text-based visual concepts "
            "and prompts. Actual image generation is disabled."
        )

    else:
        st.error("⚠️ Visual concepts were not generated. Check errors below.")

    # ── Errors ────────────────────────────────────────────────────────────────
    if errors := result.get("errors"):
        st.divider()
        st.markdown("**⚠️ Errors / Warnings**")
        for err in errors:
            st.error(err)
