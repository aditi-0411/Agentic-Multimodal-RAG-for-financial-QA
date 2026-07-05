"""
retrieval.py — shared retrieval primitives used by all specialist agents:
dense (FAISS) retrieval, sparse (BM25) retrieval, Reciprocal Rank Fusion,
cross-encoder reranking (with financial-domain score boosts), and HyDE
query expansion for the Text Agent.
"""

import re

import numpy as np
import torch


def dense_retrieve(query, embedding_model, faiss_index, top_k=10):
    q_emb = embedding_model.encode([query], normalize_embeddings=True)
    scores, indices = faiss_index.search(q_emb, top_k)
    return [{"doc_id": int(indices[0][i]), "score": float(scores[0][i])} for i in range(len(indices[0]))]


def sparse_retrieve(query, bm25_index, top_k=10):
    scores = bm25_index.get_scores(query.split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"doc_id": int(idx), "score": float(scores[idx])} for idx in top_indices]


def rrf_fusion(dense_results, sparse_results, k=60):
    """Reciprocal Rank Fusion: combines two ranked lists without needing
    score calibration between dense and sparse retrievers."""
    fusion = {}
    for rank, item in enumerate(dense_results):
        fusion[item["doc_id"]] = fusion.get(item["doc_id"], 0) + 1 / (k + rank)
    for rank, item in enumerate(sparse_results):
        fusion[item["doc_id"]] = fusion.get(item["doc_id"], 0) + 1 / (k + rank)
    return sorted(fusion.items(), key=lambda x: x[1], reverse=True)


def rerank(query, candidate_docs, reranker):
    """Cross-encoder reranking with financial-domain score boosts:
    surfaces numeric/percentage content for growth-type queries, and
    up-weights table chunks slightly (tables carry the actual figures)."""
    if not candidate_docs:
        return []

    pairs = [(query, doc["content"]) for doc in candidate_docs]
    raw_scores = reranker.predict(pairs)
    norm = torch.sigmoid(torch.tensor(raw_scores)).numpy()

    for i, doc in enumerate(candidate_docs):
        s = float(norm[i])
        q = query.lower()
        if any(w in q for w in ["growth", "increase", "decrease", "percentage"]):
            if "%" in doc["content"] or "percent" in doc["content"]:
                s += 0.15
            elif "growth" in doc["content"] and "%" not in doc["content"]:
                s -= 0.05
        if doc.get("type") == "table":
            s += 0.10
        doc["rerank_score"] = s

    return sorted(candidate_docs, key=lambda x: x["rerank_score"], reverse=True)


def numeric_boost(reranked_docs):
    """Extra boost for chunks containing explicit numeric/percentage/dollar patterns.
    Used by the Table Agent after the base rerank pass."""
    numeric_pattern = re.compile(r"\d+\.?\d*\s*%|\$\s*\d+|\d{4,}")
    for doc in reranked_docs:
        if numeric_pattern.search(doc["content"]):
            doc["rerank_score"] += 0.10
    reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_docs


def hyde_generate(query):
    """Hypothetical Document Embeddings: generates a plausible pseudo-answer
    to embed instead of the raw query, since financial-report prose and a
    short user question live in very different regions of embedding space."""
    return (
        f"The financial report discusses detailed numerical values including revenue, "
        f"net income, percentage growth, and year-over-year comparisons. "
        f"Specifically related to: {query}. "
        f"The answer likely involves extracting values from tables and analyzing trends."
    )
