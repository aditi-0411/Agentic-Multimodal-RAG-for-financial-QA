"""synthesizer.py — SYNTHESIZER AGENT: calls the LLM with the merged
context to produce the raw answer."""

from src.state import AgentState

SYNTHESIS_PROMPT_TEMPLATE = """You are a financial analyst assistant.
Answer the question STRICTLY based on the provided context.

Instructions:
- Prioritise numerical data and percentages if present.
- Use table data when the question involves figures or growth.
- Use metadata to ground the answer (company, year, sector).
- If the answer is not found in the context, say "Not found in provided context."
- Be concise and precise.

Context:
{context}

Question:
{query}

Answer:"""


def synthesizer_agent(state: AgentState, llm) -> AgentState:
    print("\n[SYNTHESIZER] Generating answer...")

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(context=state.merged_context, query=state.query)

    response = llm.invoke(prompt)
    state.raw_answer = response.content.strip()

    print(f"   Raw answer: {state.raw_answer[:200]}...")
    return state
