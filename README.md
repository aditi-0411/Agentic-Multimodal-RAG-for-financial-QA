# Agentic RAG for Financial Question Answering

A multi-agent Retrieval-Augmented Generation system for answering questions over financial reports — built as a ReAct-style pipeline of specialist agents (Text / Table / Metadata) coordinated by a Planner, Router, Critic, and Synthesizer, with an LLM-as-judge evaluation layer scoring every answer on four RAGAS-style metrics.

> The interesting part isn't "RAG over PDFs" — it's the multi-agent control flow: a Planner that classifies and rewrites the query, a Router that can override the Planner's agent selection, a Critic that can trigger a bounded re-retrieval loop without ever discarding a good answer, and an evaluation layer that scores itself with no ground-truth labels.

---

## Architecture

```
USER QUERY
    │
    ▼
┌──────────────┐
│  PLANNER     │  classifies query type, rewrites query, builds execution plan
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ROUTER      │  LLM validates/overrides Planner's agent selection
└──────┬───────┘
       │
   ┌───┴───────────────────────┐
   ▼                           ▼                          ▼
┌──────────┐          ┌──────────────┐          ┌───────────────────┐
│TEXT AGENT│          │ TABLE AGENT  │          │  METADATA AGENT    │
│HyDE+Dense│          │Dense+Sparse  │          │ Dense+Sparse+RRF   │
│+Sparse   │          │+RRF+Rerank   │          │ +Rerank            │
│+RRF      │          │+Numeric boost│          │ +Structured summary│
│+Rerank   │          └──────┬───────┘          └─────────┬──────────┘
└──────┬───┘                 │                             │
       └─────────────────────┼─────────────────────────────┘
                              ▼
                     ┌─────────────────┐
                     │ CONTEXT MERGER  │
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │  SYNTHESIZER    │  LLM generates answer
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │  CRITIC AGENT   │  checks grounding/completeness
                     │  (ReAct loop)   │  re-retrieves if REFINE verdict
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐
                     │   FORMATTER     │  final structured answer
                     └────────┬────────┘
                              ▼
                        FINAL ANSWER
```

Each specialist agent uses **dense retrieval (FAISS, BGE-small embeddings) + sparse retrieval (BM25) fused with Reciprocal Rank Fusion, then reranked with a cross-encoder** — the Text Agent additionally expands the query with HyDE (Hypothetical Document Embeddings) before the dense pass, and the Table Agent adds a post-rerank numeric/percentage-pattern boost.

## Evaluation results

Ran on 3 held-out queries against the BAF (financial QA) sample dataset, scored by LLM-as-judge on Faithfulness / Answer Relevancy / Context Precision / Completeness (no ground-truth labels used):

| Query | Type | Agents Used | Iterations | Faithfulness | Relevancy | Ctx. Precision | Completeness | Overall |
|---|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Revenue growth of the company? | numeric | TableAgent → +TextAgent (refined) | 1 | 0.00 | 0.80 | 0.40 | 0.60 | 0.45 |
| Which sector does the company belong to? | metadata | MetadataAgent | 0 | 1.00 | 1.00 | 0.75 | 0.75 | **0.88** |
| Net income mentioned in the report? | financial | TextAgent, TableAgent | 1 | 0.50 | 0.60 | 0.40 | 0.50 | 0.50 |
| **Mean** | | | | **0.50** | **0.80** | **0.52** | **0.62** | **0.61** |

**What this actually shows, read honestly:** the metadata-lookup query performs well (0.88) — simple, single-fact retrieval is where this architecture is strongest. The numeric/financial queries score much lower, and the standout number is **Faithfulness = 0.00 on the revenue-growth query** despite the Critic marking that same answer "GROUNDED: Yes" one step earlier. That's not just a low score — it's the two judges (Critic vs. the separate faithfulness evaluator) disagreeing, which is itself a useful diagnostic: the Critic's binary grounded/not-grounded check is looser than the evaluation layer's claim-by-claim scoring. The built-in diagnostic summary (`src/evaluation.py`) flags exactly this kind of pattern automatically (e.g. "LOW FAITHFULNESS → hallucination risk, raise rerank threshold").

This is a small, illustrative 3-query run, not a benchmark — treat the numbers as a demonstration of the evaluation harness working correctly (catching a real discrepancy) rather than a claim about overall system quality. Worth re-running on a larger held-out query set before citing these numbers anywhere more formal.

## Project structure

```
.
├── README.md
├── requirements.txt
├── .env.example                    # copy to .env and fill in your own key
├── notebooks/
│   └── agentic_rag_financial_qa.ipynb   # original notebook, API key redacted
├── src/
│   ├── config.py                   # env-based LLM/embedding/reranker setup
│   ├── data_loading.py             # JSONL loading, cleaning, table parsing
│   ├── indexing.py                 # chunking + FAISS/BM25 index construction
│   ├── retrieval.py                # dense/sparse retrieval, RRF, rerank, HyDE
│   ├── state.py                    # AgentState — shared pipeline state
│   ├── orchestrator.py             # master controller sequencing all agents
│   ├── evaluation.py               # LLM-as-judge metrics + report
│   └── agents/
│       ├── planner.py
│       ├── router.py
│       ├── specialists.py          # Text / Table / Metadata agents
│       ├── merger.py
│       ├── synthesizer.py
│       ├── critic.py               # ReAct refine loop, bug-fixed
│       └── formatter.py
└── results/
    └── (evaluation CSVs / reports land here)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your own GROQ_API_KEY
```

You'll also need a JSONL financial-QA dataset (fields: `question`, `pre_text`, `context`, `post_text`, `table`, `original_answer`, `company_name`, `report_year`, `company_sector`) — this repo doesn't redistribute the dataset itself.

## Usage

```python
import os
from dotenv import load_dotenv
load_dotenv()

from src.config import get_llm, get_embedding_model, get_reranker
from src.data_loading import load_jsonl, prepare_dataset
from src.indexing import build_knowledge_base
from src.orchestrator import orchestrator, print_result

llm = get_llm()
embedding_model = get_embedding_model()
reranker = get_reranker()

data = load_jsonl("your_dataset.jsonl")
processed = prepare_dataset(data, max_samples=500)
kb = build_knowledge_base(processed, embedding_model)

result = orchestrator("What was the revenue growth of the company?", kb, llm, embedding_model, reranker)
print_result(result)
```

Running the evaluation layer:

```python
from src.evaluation import evaluate_pipeline, print_eval_report

queries = [
    "What was the revenue growth of the company?",
    "Which sector does the company belong to?",
    "What is the net income mentioned in the report?",
]
results = evaluate_pipeline(queries, kb, llm, embedding_model, reranker)
eval_df = print_eval_report(results)
```

## Tech stack

LangChain (`langchain-groq`) · Groq (Llama 3.1 8B Instant) · sentence-transformers (BGE-small embeddings, cross-encoder reranker) · FAISS · BM25 (rank_bm25) · pandas

## Future work

- [ ] Run evaluation on a larger query set (the 3-query run here is illustrative, not a benchmark)
- [ ] Investigate the Critic-vs-Faithfulness-evaluator disagreement seen on the numeric query — likely means the Critic's grounding check needs to be claim-level, not answer-level
- [ ] Add citation-level grounding (link each sentence of the final answer back to a specific retrieved chunk)
- [ ] Swap the fixed reranker score boosts (`+0.10`, `+0.15`) for a learned or validated weighting

## Author

Aditi — M.Sc.-M.Tech. Data and Computational Science, IIT Jodhpur
