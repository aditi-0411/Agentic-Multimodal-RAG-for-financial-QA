"""planner.py — PLANNER AGENT: classifies query type, rewrites the query
for retrieval, and builds a step-by-step execution plan."""

from src.state import AgentState

REWRITE_MAP = {
    "numeric": " percentage increase year over year revenue growth financial statement",
    "financial": " income statement revenue total revenue financial report",
    "metadata": " company name sector year industry classification",
    "general": " financial data business performance annual report",
}

PLAN_MAP = {
    "numeric": ["TextAgent", "TableAgent", "Critic", "Synthesizer"],
    "financial": ["TextAgent", "TableAgent", "MetadataAgent", "Critic", "Synthesizer"],
    "metadata": ["MetadataAgent", "TextAgent", "Critic", "Synthesizer"],
    "general": ["TextAgent", "MetadataAgent", "Critic", "Synthesizer"],
}


def planner_agent(state: AgentState) -> AgentState:
    print("\n[PLANNER] Analysing query...")

    query = state.query.lower().strip()

    if any(w in query for w in ["growth", "increase", "decrease", "percentage", "change", "ratio"]):
        state.query_type = "numeric"
    elif any(w in query for w in ["revenue", "income", "profit", "loss", "earnings", "sales", "cost"]):
        state.query_type = "financial"
    elif any(w in query for w in ["company", "sector", "year", "industry", "who", "when", "which"]):
        state.query_type = "metadata"
    else:
        state.query_type = "general"

    state.rewritten_query = state.query + REWRITE_MAP.get(state.query_type, "")

    state.plan = PLAN_MAP.get(state.query_type, PLAN_MAP["general"])
    state.agents_to_call = [a for a in state.plan if a not in ("Critic", "Synthesizer")]

    print(f"   Query type    : {state.query_type}")
    print(f"   Rewritten     : {state.rewritten_query[:80]}...")
    print(f"   Execution plan: {state.plan}")

    return state
