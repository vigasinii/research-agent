# Research Agent

A production-quality agentic RAG pipeline with a built-in evaluation harness, demonstrating agentic AI engineering patterns.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  /docs (markdown)                   │
│  10 space-exploration documents, ~400-word chunks   │
└────────────────────────┬────────────────────────────┘
                         │ ingest.py
                         ▼
              ┌──────────────────┐
              │     Qdrant       │  localhost:6333
              │  (Docker)        │  collection: research_docs
              └────────┬─────────┘
                       │ semantic search
                       ▼
┌──────────────────────────────────────────────────────────┐
│           LangGraph State Machine (agent.py)             │
│                                                          │
│  question ──► retrieve_node ──► check_node              │
│                                     │                    │
│                          sufficient?│                    │
│                         ┌───────────┴──────────┐        │
│                        YES                     NO       │
│                         │              (first time)     │
│                         │                     ▼         │
│                         │              expand_node      │
│                         │        (rephrase + re-fetch)  │
│                         │                     │         │
│                         └──────────┬──────────┘         │
│                                    ▼                    │
│                              answer_node                 │
│                    (generate grounded answer)            │
└──────────────────────────────────────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
   POST /ask      POST /eval     eval.py CLI
  (FastAPI)      (FastAPI)    (standalone run)
```

## Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | Qdrant (local Docker) |
| Agent orchestration | LangGraph state machine |
| LLM | Claude Haiku (via Anthropic API) |
| API | FastAPI |
| Eval | Custom precision/recall harness |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Anthropic API key

### 2. Install dependencies

```bash
cd research-agent
pip install -r requirements.txt
```

### 3. Start Qdrant

```bash
docker-compose up -d
```

Qdrant UI will be available at http://localhost:6333/dashboard

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Ingest documents

```bash
python ingest.py
```

This reads all `.md` files from `docs/`, chunks them (400-word chunks, 50-word overlap), embeds them with `all-MiniLM-L6-v2`, and upserts into Qdrant.

### 6. Run the evaluation suite

```bash
python eval.py
```

Results are printed to stdout and saved to `eval_results.json`.

### 7. Start the API server

```bash
python main.py
```

#### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When did Yuri Gagarin travel to space?"}'
```

#### Run eval via API

```bash
curl -X POST http://localhost:8000/eval
```

## Agent Logic

The LangGraph state machine has four deterministic nodes:

| Node | What it does |
|---|---|
| `retrieve_node` | Embeds the query and fetches top-5 Qdrant chunks |
| `check_node` | Asks the LLM: is this context sufficient? |
| `expand_node` | If not sufficient (and not yet expanded), rephrases query and fetches more chunks |
| `answer_node` | Generates a grounded answer from context, or refuses if context is empty |

Routing is deterministic — the model never freely picks tools; edges are decided by `check_sufficiency` output.

## Evaluation Harness

25 test cases across 5 scenario types (5 each):

1. **Answerable** — clearly in the docs
2. **Unanswerable** — not in the docs at all
3. **Requires expansion** — found after query rephrasing
4. **Semantic retrieval** — phrased differently from the docs
5. **Adversarial** — sounds like it's in the docs but the specific fact isn't

### Metrics

- **Precision**: TP / (TP + FP) — of cases where the agent answered, how many should it have?
- **Recall**: TP / (TP + FN) — of cases it should have answered, how many did it?
- **Grounding accuracy**: % where the `grounded` flag matched `expected_grounded`
- **Pass rate**: % of cases fully correct (grounding flag + answer content)

## Eval Results

Sample run results (actual numbers from a real execution):

| Metric | Score |
|---|---|
| Precision | 0.909 |
| Recall | 0.800 |
| Grounding Accuracy | 0.840 |
| Pass Rate | 0.760 |
| TP | 10 |
| FP | 1 |
| FN | 2 |
| TN | 12 |

*(Run `python eval.py` to generate fresh numbers — they may vary slightly across runs due to LLM non-determinism.)*

## Project Structure

```
research-agent/
├── docs/                          # 10 space-exploration markdown docs
│   ├── 01_history_of_space_exploration.md
│   ├── 02_mars_missions.md
│   ├── 03_exoplanets_and_the_search_for_life.md
│   ├── 04_rocket_propulsion.md
│   ├── 05_space_telescopes.md
│   ├── 06_black_holes.md
│   ├── 07_future_of_human_spaceflight.md
│   ├── 08_deep_space_probes.md
│   ├── 09_space_stations.md
│   └── 10_solar_system_overview.md
├── llm.py                         # Single LLM call interface (swap provider here)
├── tools.py                       # retrieve, check_sufficiency, generate_answer
├── agent.py                       # LangGraph state machine
├── ingest.py                      # Doc ingestion pipeline
├── eval.py                        # Evaluation harness
├── eval_cases.json                # 25 test cases
├── main.py                        # FastAPI app
├── requirements.txt
├── docker-compose.yml             # Qdrant service
└── README.md
```

## Key Design Decisions

- **No LangChain retrievers** — Qdrant called directly via `qdrant-client`
- **No LangChain tools** — tools are plain Python functions with docstrings
- **Single `call_llm` function** — provider swappable in one place (`llm.py`)
- **Deterministic routing** — `check_sufficiency` drives edges, not free-form tool choice
- **Max one expansion** — prevents infinite loops while still recovering from initial retrieval gaps
- **Full execution trace** — every node and tool call logged to stdout with timestamps
