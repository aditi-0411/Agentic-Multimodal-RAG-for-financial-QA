"""specialists.py — the three specialist retrieval agents:

- TextAgent     : narrative/paragraph text, via HyDE + Dense + Sparse + RRF + Rerank
- TableAgent    : numerical tables, with extra numeric-pattern boosting
- MetadataAgent : company/sector/year lookups, plus a structured summary
"""

from src.indexing import KnowledgeBase
from src.retrieval import dense_retrieve, hyde_generate, numeric_boost, rerank, rrf_fusion, sparse_retrieve
from src.state import AgentState


def text_agent(state: AgentState, kb: KnowledgeBase, embedding_model, reranker) -> AgentState:
    print("\n[TEXT AGENT] Retrieving text chunks...")

    hyde_query = hyde_generate(state.query)

    dense_res = dense_retrieve(hyde_query, embedding_model, kb.text_index, top_k=12)
    sparse_res = sparse_retrieve(state.rewritten_query, kb.bm25_text, top_k=12)
    fused = rrf_fusion(dense_res, sparse_res)

    candidates = []
    for doc_id, score in fused[:12]:
        doc = kb.text_chunks[doc_id]
        candidates.append({"content": doc["content"], "type": doc["type"], "metadata": doc["metadata"], "base_score": score})

    state.text_results = rerank(state.query, candidates, reranker)[:5]

    print(f"   Retrieved {len(state.text_results)} text chunks")
    return state


def table_agent(state: AgentState, kb: KnowledgeBase, embedding_model, reranker) -> AgentState:
    print("\n[TABLE AGENT] Retrieving table chunks...")

    if not kb.table_chunks:
        print("   No table chunks available.")
        return state

    dense_res = dense_retrieve(state.rewritten_query, embedding_model, kb.table_index, top_k=10)
    sparse_res = sparse_retrieve(state.rewritten_query, kb.bm25_table, top_k=10)
    fused = rrf_fusion(dense_res, sparse_res)

    candidates = []
    for doc_id, score in fused[:10]:
        doc = kb.table_chunks[doc_id]
        candidates.append({"content": doc["content"], "type": "table", "metadata": doc["metadata"], "base_score": score})

    reranked = rerank(state.query, candidates, reranker)
    reranked = numeric_boost(reranked)

    state.table_results = reranked[:5]
    print(f"   Retrieved {len(state.table_results)} table chunks")
    return state


def metadata_agent(state: AgentState, kb: KnowledgeBase, embedding_model, reranker) -> AgentState:
    print("\n[METADATA AGENT] Retrieving metadata...")

    dense_res = dense_retrieve(state.query, embedding_model, kb.metadata_index, top_k=8)
    sparse_res = sparse_retrieve(state.query, kb.bm25_metadata, top_k=8)
    fused = rrf_fusion(dense_res, sparse_res)

    candidates = []
    for doc_id, score in fused[:8]:
        doc = kb.metadata_chunks[doc_id]
        candidates.append({"content": doc["content"], "type": "metadata", "metadata": doc["metadata"], "base_score": score})

    state.metadata_results = rerank(state.query, candidates, reranker)[:4]

    print(f"   Retrieved {len(state.metadata_results)} metadata chunks")
    return state
