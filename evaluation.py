"""
evaluation.py — RAGAS-style evaluation layer, computed WITHOUT ground-truth
labels, using the LLM itself as judge.

Four metrics per query:
- Faithfulness      — is every claim in the answer supported by the retrieved context?
- Answer Relevancy  — does the answer actually address the question asked?
- Context Precision — are the retrieved chunks relevant to the query (no noise)?
- Completeness      — does the answer fully resolve the query without gaps?
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.orchestrator import orchestrator


@dataclass
class EvalResult:
    query: str
    final_answer: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    completeness: float = 0.0
    overall_score: float = 0.0
    faithfulness_reason: str = ""
    answer_relevancy_reason: str = ""
    context_precision_reason: str = ""
    completeness_reason: str = ""
    agents_used: List[str] = field(default_factory=list)
    iterations: int = 0
    query_type: str = ""


def _parse_score(text: str, key: str):
    score_m = re.search(rf"{key}_SCORE:\s*([0-9.]+)", text, re.IGNORECASE)
    reason_m = re.search(rf"{key}_REASON:\s*(.+)", text, re.IGNORECASE)
    score = float(score_m.group(1)) if score_m else 0.5
    score = max(0.0, min(1.0, score))
    reason = reason_m.group(1).strip() if reason_m else "N/A"
    return score, reason


def evaluate_faithfulness(query, answer, context, llm):
    prompt = f"""You are evaluating a RAG system's answer for faithfulness.

CONTEXT (what was retrieved):
{context[:2000]}

ANSWER (what the system said):
{answer}

Task: Identify every factual claim in the answer.
Count how many are directly supported by the context vs invented.

Score = supported_claims / total_claims  (between 0.0 and 1.0)

Respond EXACTLY:
FAITHFULNESS_SCORE: <0.0 to 1.0>
FAITHFULNESS_REASON: <one sentence>"""
    try:
        r = llm.invoke(prompt)
        return _parse_score(r.content, "FAITHFULNESS")
    except Exception as e:
        return 0.5, f"Eval error: {e}"


def evaluate_answer_relevancy(query, answer, llm):
    prompt = f"""You are evaluating whether an answer is relevant to its question.

QUESTION: {query}
ANSWER: {answer}

Score how directly the answer addresses the question.
Penalise answers that are vague, off-topic, or only partially responsive.

Score = relevance (between 0.0 and 1.0)

Respond EXACTLY:
ANSWER_RELEVANCY_SCORE: <0.0 to 1.0>
ANSWER_RELEVANCY_REASON: <one sentence>"""
    try:
        r = llm.invoke(prompt)
        return _parse_score(r.content, "ANSWER_RELEVANCY")
    except Exception as e:
        return 0.5, f"Eval error: {e}"


def evaluate_context_precision(query, retrieved_docs: List[Dict], llm):
    if not retrieved_docs:
        return 0.0, "No documents retrieved"

    snippets = ""
    for i, doc in enumerate(retrieved_docs):
        snippets += f"[Chunk {i+1} | {doc.get('type', '?')}]: {doc.get('content', '')[:200]}\n"

    prompt = f"""You are evaluating context precision in a RAG system.

QUESTION: {query}

RETRIEVED CHUNKS:
{snippets}

For each chunk, decide if it is RELEVANT or IRRELEVANT to answering the question.
Score = relevant_chunks / total_chunks  (between 0.0 and 1.0)

Respond EXACTLY:
CONTEXT_PRECISION_SCORE: <0.0 to 1.0>
CONTEXT_PRECISION_REASON: <one sentence>"""
    try:
        r = llm.invoke(prompt)
        return _parse_score(r.content, "CONTEXT_PRECISION")
    except Exception as e:
        return 0.5, f"Eval error: {e}"


def evaluate_completeness(query, answer, context, llm):
    prompt = f"""You are evaluating whether a RAG answer is complete.

QUESTION: {query}

CONTEXT (available information):
{context[:2000]}

ANSWER (what the system gave):
{answer}

Check: given the context, is there important information the answer missed?
Score = 1.0 if nothing was missed, lower if gaps exist.

Respond EXACTLY:
COMPLETENESS_SCORE: <0.0 to 1.0>
COMPLETENESS_REASON: <one sentence>"""
    try:
        r = llm.invoke(prompt)
        return _parse_score(r.content, "COMPLETENESS")
    except Exception as e:
        return 0.5, f"Eval error: {e}"


def evaluate_pipeline(queries: List[str], kb, llm, embedding_model, reranker) -> List[EvalResult]:
    """Runs the full agentic pipeline on each query, then scores it with all 4 metrics."""
    results = []

    for query in queries:
        print(f"\n{'='*55}")
        print(f"Evaluating: {query}")
        print("=" * 55)

        state = orchestrator(query, kb, llm, embedding_model, reranker)
        all_docs = state.text_results + state.table_results + state.metadata_results

        print("\n   Scoring metrics...")
        faith_s, faith_r = evaluate_faithfulness(query, state.final_answer, state.merged_context, llm)
        relev_s, relev_r = evaluate_answer_relevancy(query, state.final_answer, llm)
        prec_s, prec_r = evaluate_context_precision(query, all_docs, llm)
        comp_s, comp_r = evaluate_completeness(query, state.final_answer, state.merged_context, llm)

        overall = round((faith_s + relev_s + prec_s + comp_s) / 4, 4)

        er = EvalResult(
            query=query, final_answer=state.final_answer,
            faithfulness=round(faith_s, 4), answer_relevancy=round(relev_s, 4),
            context_precision=round(prec_s, 4), completeness=round(comp_s, 4),
            overall_score=overall,
            faithfulness_reason=faith_r, answer_relevancy_reason=relev_r,
            context_precision_reason=prec_r, completeness_reason=comp_r,
            agents_used=state.agents_to_call, iterations=state.iterations, query_type=state.query_type,
        )
        results.append(er)

        print(f"   Faithfulness      : {faith_s:.2f}  — {faith_r}")
        print(f"   Answer Relevancy  : {relev_s:.2f}  — {relev_r}")
        print(f"   Context Precision : {prec_s:.2f}  — {prec_r}")
        print(f"   Completeness      : {comp_s:.2f}  — {comp_r}")
        print(f"   Overall           : {overall:.2f}")

    return results


def print_eval_report(results: List[EvalResult]) -> pd.DataFrame:
    """Pretty-prints a full evaluation report with per-query and aggregate
    scores, plus a rule-based diagnostic summary, and returns a DataFrame."""

    METRICS = ["faithfulness", "answer_relevancy", "context_precision", "completeness", "overall_score"]
    LABELS = ["Faithfulness", "Ans. Relevancy", "Ctx. Precision", "Completeness", "OVERALL"]

    def bar(s):
        return ("#" * int(s * 20)).ljust(20) + f" {s:.2f}"

    print("\n" + "=" * 70)
    print("               AGENTIC RAG — EVALUATION REPORT")
    print("=" * 70)

    for i, er in enumerate(results):
        print(f"\n-- Query {i+1}: {er.query[:65]}")
        print(f"   Type: {er.query_type:<12} Agents: {er.agents_used}  Iterations: {er.iterations}")
        for metric, label in zip(METRICS, LABELS):
            score = getattr(er, metric)
            reason = getattr(er, metric + "_reason", "")
            flag = "OK " if score >= 0.7 else ("MID" if score >= 0.4 else "LOW")
            print(f"   [{flag}] {label:<18} {bar(score)}")
            if reason:
                print(f"        -> {reason[:70]}")

    print("\n" + "-" * 70)
    print("  AGGREGATE SCORES (mean across all queries)")
    print("-" * 70)
    for metric, label in zip(METRICS, LABELS):
        avg = sum(getattr(r, metric) for r in results) / len(results)
        flag = "OK " if avg >= 0.7 else ("MID" if avg >= 0.4 else "LOW")
        print(f"  [{flag}] {label:<18} {bar(avg)}")

    print("\n" + "-" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("-" * 70)

    avg_faith = sum(r.faithfulness for r in results) / len(results)
    avg_relev = sum(r.answer_relevancy for r in results) / len(results)
    avg_prec = sum(r.context_precision for r in results) / len(results)
    avg_comp = sum(r.completeness for r in results) / len(results)
    avg_iters = sum(r.iterations for r in results) / len(results)

    diagnostics = []
    if avg_faith < 0.6:
        diagnostics.append("LOW FAITHFULNESS  -> Reranker threshold too low; hallucination risk. "
                            "Fix: raise top_k cutoff, add source-grounding instruction to Synthesizer prompt.")
    if avg_relev < 0.6:
        diagnostics.append("LOW RELEVANCY     -> Planner/Router mismatch; wrong agents called. "
                            "Fix: tighten Router prompt, add query-type examples.")
    if avg_prec < 0.6:
        diagnostics.append("LOW CTX PRECISION -> Too much retrieval noise. "
                            "Fix: reduce top_k in dense/sparse retrieval, raise rerank_score threshold.")
    if avg_comp < 0.6:
        diagnostics.append("LOW COMPLETENESS  -> Critic REFINE loop not helping or MAX_ITER too low. "
                            "Fix: increase MAX_ITER to 3, improve HINT extraction in Critic.")
    if avg_iters > 0.8:
        diagnostics.append("HIGH ITERATION RATE -> Planner query classification may be wrong. "
                            "Fix: add more keyword rules to planner_agent classify block.")

    if not diagnostics:
        print("  All metrics in healthy range.")
    else:
        for d in diagnostics:
            print(f"  - {d}")

    print("\n" + "=" * 70)

    rows = []
    for er in results:
        rows.append({
            "Query": er.query[:50], "Type": er.query_type,
            "Faithfulness": er.faithfulness, "Ans. Relevancy": er.answer_relevancy,
            "Ctx. Precision": er.context_precision, "Completeness": er.completeness,
            "Overall": er.overall_score, "Iterations": er.iterations,
            "Agents": ", ".join(er.agents_used),
        })
    return pd.DataFrame(rows)
