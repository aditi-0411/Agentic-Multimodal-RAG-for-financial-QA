"""
orchestrator.py — ORCHESTRATOR: the master controller that sequences all
agents based on the Planner's plan.

Fixes carried over from the original notebook:
1. Router can only select from the Planner's agent list — it cannot
   invoke an agent the Planner didn't already deem relevant (no
   hallucinated agent names from the LLM router).
2. Uses the best-answer-tracking Critic (see src/agents/critic.py).
3. state.query is never mutated during the refine loop.
"""

from src.agents.critic import critic_agent
from src.agents.formatter import format_final_answer
from src.agents.merger import merge_context
from src.agents.planner import planner_agent
from src.agents.router import router_agent
from src.agents.specialists import metadata_agent, table_agent, text_agent
from src.agents.synthesizer import synthesizer_agent
from src.indexing import KnowledgeBase
from src.state import AgentState

AGENT_MAP = {
    "TextAgent": text_agent,
    "TableAgent": table_agent,
    "MetadataAgent": metadata_agent,
}


def orchestrator(query: str, kb: KnowledgeBase, llm, embedding_model, reranker) -> AgentState:
    print("=" * 60)
    print(f"ORCHESTRATOR — Query: {query}")
    print("=" * 60)

    state = AgentState(query=query)
    state.best_answer = ""

    state = planner_agent(state)
    state = router_agent(state, llm)

    # Enforce: Router can only pick from Planner's list
    planner_agents = [a for a in state.plan if a not in ("Critic", "Synthesizer")]
    state.agents_to_call = [a for a in state.agents_to_call if a in planner_agents]
    if not state.agents_to_call:
        state.agents_to_call = planner_agents
        print(f"   Router returned invalid agents — using Planner's list: {planner_agents}")

    for agent_name in state.agents_to_call:
        agent_fn = AGENT_MAP.get(agent_name)
        if agent_fn:
            state = agent_fn(state, kb, embedding_model, reranker)

    state = merge_context(state)
    state = synthesizer_agent(state, llm)
    state = critic_agent(state, kb, llm, embedding_model, reranker)
    state = format_final_answer(state, llm)

    return state


def print_result(state: AgentState):
    print("\n" + "=" * 60)
    print("AGENTIC RAG — FINAL OUTPUT")
    print("=" * 60)
    print(f"Query       : {state.query}")
    print(f"Query Type  : {state.query_type}")
    print(f"Agents Used : {state.agents_to_call}")
    print(f"Iterations  : {state.iterations}")
    print("\n---- Final Answer ----")
    print(state.final_answer)
    print("\n---- Critic Feedback ----")
    print(state.critic_feedback)
    print("=" * 60)
