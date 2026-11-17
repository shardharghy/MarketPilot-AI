# MarketPilot AI

**Multi-Agent Generative AI System for Content Generation and Marketing**
*B.Tech Final Year Project*

---

## Three-Agent Architecture

| # | Agent | Role | Model |
|---|-------|------|-------|
| 1 | **Campaign Supervisor** | Receives user input, creates structured `CampaignBrief`, coordinates downstream agents | Gemini 2.0 Flash |
| 5 | **Content Creator** | Generates 2 content variations per platform (Instagram / Facebook / LinkedIn) with captions, hashtags, CTAs, and headlines | Gemini 2.0 Flash |
| 8 | **Visual Creative Agent** | Builds detailed Imagen prompts, generates real images, saves PNG files | Imagen 3 (`imagen-3.0-generate-002`) |

> Agents 2, 3, 4, 6, 7 are pre-wired into the shared state but not yet implemented. See **Adding Future Agents** below.

---

## Quick Start

### 1 — Prerequisites

- Python 3.12
- Gemini API key (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### 2 — Install dependencies

```bash
# Activate your virtual environment first
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

### 3 — Configure API key

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Mac / Linux
```

Edit `.env` and replace `your_gemini_api_key_here` with your real key:

```
GEMINI_API_KEY=AIza...
```

### 4 — Verify models

```bash
python verify_models.py
```

Expected output:
```
[1/2] Testing text model: gemini-2.0-flash      ✅
[2/2] Testing image model: imagen-3.0-generate-002  ✅
All models verified. MarketPilot AI is ready to run.
```

### 5 — Run tests

```bash
pytest tests/ -v
```

> Tests that make real API calls are automatically skipped if `GEMINI_API_KEY` is not set.

### 6 — Launch the app

```bash
streamlit run ui/app.py
```

---

## Project Structure

```
MarketPilotAI/
│
├── agents/                                  # One file per implemented agent
│   ├── agent_01_campaign_supervisor.py      ← Agent 1
│   ├── agent_05_content_creator.py          ← Agent 5
│   └── agent_08_visual_creative.py          ← Agent 8
│
├── core/
│   ├── state.py                             # All Pydantic schemas + LangGraph TypedDict
│   └── workflow.py                          # LangGraph StateGraph + run_campaign()
│
├── services/
│   ├── gemini_provider.py                   # Gemini text client (all agents share this)
│   └── image_service.py                     # Imagen client + PNG saving
│
├── prompts/                                 # System prompts as plain .txt files
│   ├── agent_01_system_prompt.txt
│   ├── agent_05_system_prompt.txt
│   └── agent_08_system_prompt.txt
│
├── ui/
│   └── app.py                               # Streamlit frontend
│
├── tests/
│   ├── test_agent_01.py
│   ├── test_agent_05.py
│   ├── test_agent_08.py
│   └── test_workflow.py
│
├── data/campaigns/                          # JSON campaign history (auto-created)
├── outputs/images/                          # Generated PNG files (auto-created)
│
├── verify_models.py                         # Pre-flight model verification script
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## LangGraph Workflow

```
START
  │
  ▼
Agent 1: Campaign Supervisor
  │  Writes: campaign_brief
  ▼
Agent 5: Content Creator
  │  Writes: generated_content
  ▼
Agent 8: Visual Creative Agent
  │  Writes: visual_assets  (PNG files saved to outputs/images/)
  ▼
END  →  Final state returned to Streamlit UI
```

---

## Adding Future Agents

The state pre-declares slots for all 8 agents. Adding one touches **exactly 3 things**:

**Example: Add Agent 2 — Marketing Strategist**

**Step 1** — Create `agents/agent_02_marketing_strategist.py`:
```python
def run_marketing_strategist(state: MarketPilotState) -> dict:
    # Reads:  state["campaign_brief"]
    # Writes: state["marketing_strategy"]
    ...
```

**Step 2** — Add `prompts/agent_02_system_prompt.txt`.

**Step 3** — In `core/workflow.py`, update `_build_graph()`:
```python
graph.add_node("agent_02_strategist", run_marketing_strategist)
graph.add_edge("agent_01_supervisor",  "agent_02_strategist")
graph.add_edge("agent_02_strategist",  "agent_05_content_creator")
# Remove the old edge: agent_01_supervisor → agent_05_content_creator
```

**Zero changes** to `agent_01`, `agent_05`, or `agent_08`.

### Recommended integration order

| Sprint | Agent | Inserts after |
|--------|-------|---------------|
| 2 | Agent 7 — Quality Reviewer | Agent 5 → (review loop) → Agent 8 |
| 3 | Agent 6 — SEO Optimizer | Agent 5 → Agent 6 → Agent 8 |
| 4 | Agent 2 — Marketing Strategist | Agent 1 → Agent 2 → Agent 5 |
| 5 | Agent 3 — Audience Researcher | Agent 2 → Agent 3 → Agent 5 |
| 6 | Agent 4 — Content Planner | Agent 3 → Agent 4 → Agent 5 |

---

## Technology Stack

| Technology | Version | Used For |
|------------|---------|----------|
| Python | 3.12 | All agent and service code |
| LangGraph | ≥0.2.55 | StateGraph orchestration |
| google-genai | ≥1.0.0 | Gemini text + Imagen image generation |
| Streamlit | ≥1.40.0 | Web UI |
| Pydantic | ≥2.0.0 | Schema validation for all agent I/O |
| Pillow | ≥10.0.0 | PNG verification and saving |
| python-dotenv | ≥1.0.0 | API key management |
| pytest | ≥8.0.0 | Automated tests |

---

## Design Principles

1. **One agent, one file** — no agent imports another agent's module.
2. **State is the contract** — agents communicate only via `MarketPilotState`. No direct function calls between agents.
3. **Provider abstraction** — no agent imports `google.genai` directly; all calls go through `gemini_provider.py` or `image_service.py`.
4. **No fake responses** — every Gemini and Imagen call is real.
5. **Future-safe state** — all 8 agent slots exist in `MarketPilotState` from day one, defaulting to `None`.
6. **Prompts as files** — system prompts live in `prompts/*.txt`, not hardcoded strings.
