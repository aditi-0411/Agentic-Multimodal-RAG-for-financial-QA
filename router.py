"""router.py — ROUTER AGENT: uses the LLM to validate/override the
Planner's specialist-agent selection, then dispatches."""

from src.state import AgentState

VALID_AGENTS = {"TextAgent", "TableAgent", "MetadataAgent"}

ROUTING_PROMPT_TEMPLATE = """You are a routing agent for a financial QA system.
Given the query below, decide which specialist agents to call.

Available agents:
- TextAgent      : handles narrative/paragraph financial text
- TableAgent     : handles numerical tables, figures, and percentages
- MetadataAgent  : handles company name, sector, year, and classification queries

Query: "{query}"
Query type already detected: {query_type}
Planner's suggested agents: {agents_to_call}

Respond with ONLY a comma-separated list of agent names to call.
Example: TextAgent, TableAgent
"""


def router_agent(state: AgentState, llm) -> AgentState:
    print("\n[ROUTER] Deciding agent routing...")

    routing_prompt = ROUTING_PROMPT_TEMPLATE.format(
        query=state.query, query_type=state.query_type, agents_to_call=state.agents_to_call
    )

    try:
        response = llm.invoke(routing_prompt)
        raw = response.content.strip()
        parsed = [a.strip() for a in raw.split(",") if a.strip() in VALID_AGENTS]

        if parsed:
            state.agents_to_call = parsed
            print(f"   Router decision (LLM): {state.agents_to_call}")
        else:
            print(f"   Router kept Planner's plan: {state.agents_to_call}")

    except Exception as e:
        print(f"   Router LLM failed ({e}), keeping Planner plan.")

    return state
