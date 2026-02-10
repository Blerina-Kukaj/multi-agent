# Enterprise Multi-Agent Copilot

> **Giga Academy Cohort IV – Project #6**
> A multi-agent system that turns a business request into a structured, decision-ready deliverable using coordinated AI agents grounded in retrieved evidence.

| Attribute | Value |
|---|---|
| **Industry** | Healthcare & Life Sciences |
| **Stack** | LangGraph + OpenAI + ChromaDB |
| **Nice-to-haves** | Observability table, Evaluation set (10 test questions) |

---

## Architecture

```
User Request
    │
    ▼
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Planner │───▶│  Researcher  │───▶│   Writer   │───▶│  Verifier  │───▶│  Deliver   │
│  Agent   │    │  Agent       │    │   Agent    │    │  Agent     │    │  Output    │
└──────────┘    └──────────────┘    └────────────┘    └────────────┘    └────────────┘
                       │                                     │
                       ▼                                     ▼
                 ┌───────────┐                        Hallucination
                 │ ChromaDB  │                        & Evidence
                 │ Vector DB │                        Checking
                 └───────────┘
```

### Workflow: Plan → Research → Draft → Verify → Deliver

1. **Planner Agent** – Decomposes the user task into sub-questions and creates an execution plan.
2. **Research Agent** – Retrieves grounded evidence from the document store with citations (`DocumentName + ChunkID`).
3. **Writer Agent** – Produces the final structured deliverable using the research notes.
4. **Verifier Agent** – Checks for hallucinations, missing evidence, and contradictions. Blocks unsupported claims.

### Deliverable Output

Every run produces:
- **Executive Summary** (max 150 words)
- **Client-ready Email**
- **Action List** (owner, due date, confidence)
- **Sources & Citations**

---

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI API key

### 1. Clone & Install

```bash
git clone <repo-url>
cd Multi-Agent

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest Documents

```bash
python -m retrieval.ingest
```

This loads the sample Healthcare & Life Sciences documents from `/data` into the ChromaDB vector store.

### 4. Run the App

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run Evaluation Suite

```bash
python -m eval.evaluate
```

---

## Project Structure

```
/app                    – Streamlit UI
    streamlit_app.py    – Main application interface
/agents                 – Agent definitions and prompts
    __init__.py
    state.py            – Shared LangGraph state schema
    prompts.py          – All agent system/user prompts
    planner.py          – Planner agent
    researcher.py       – Research agent
    writer.py           – Writer agent
    verifier.py         – Verifier agent
    graph.py            – LangGraph workflow orchestration
/retrieval              – Document loaders and vector search
    __init__.py
    loader.py           – Document chunking & loading
    vector_store.py     – ChromaDB wrapper
    ingest.py           – Ingestion script
/data                   – Sample documents + README
    README.md           – Document descriptions
    *.txt               – 10 synthetic Healthcare & Life Sciences documents
/eval                   – Test prompts & evaluation
    __init__.py
    test_prompts.py     – 10 evaluation test questions
    evaluate.py         – Automated evaluation runner
config.py               – Central configuration
requirements.txt        – Python dependencies
.env.example            – Environment variable template
.gitignore              – Git ignore rules
```

---

## Example Usage

**Input:**
> "Analyze the impact of recent FDA guidelines on clinical trial timelines for oncology drugs and recommend process improvements."

**Output:**
- Executive summary of FDA guideline impacts
- Email draft to clinical ops leadership
- Action list with owners, deadlines, and confidence scores
- All claims cite source documents; unsupported claims flagged as "Not found in sources"

---

## Observability

The UI includes an **Observability Table** showing per-agent metrics:

| Agent | Latency (s) | Tokens (in/out) | Status | Errors |
|---|---|---|---|---|
| Planner | 1.2 | 320 / 180 | ✅ | – |
| Researcher | 3.4 | 850 / 620 | ✅ | – |
| Writer | 2.8 | 1200 / 980 | ✅ | – |
| Verifier | 1.5 | 900 / 450 | ✅ | – |

---

## Team

Giga Academy Cohort IV

## License

Internal use only – no confidential Genpact/client data included.
