"""formatter.py — FINAL ANSWER FORMATTER: polishes the raw answer into a
clean, structured, citation-tagged response."""

from src.state import AgentState

FORMAT_PROMPT_TEMPLATE = """You are a professional financial report writer.

Given the raw answer below, rewrite it into a clean, structured response.

Rules:
- Keep all numbers and percentages exactly as given.
- Add a one-line summary at the top.
- Cite source type (Text / Table / Metadata) at the end.
- Do NOT add any information not present in the raw answer.
- Max 150 words.

Raw answer:
{raw_answer}

Sources used:
- Text chunks: {n_text}
- Table chunks: {n_table}
- Metadata chunks: {n_metadata}

Formatted answer:"""


def format_final_answer(state: AgentState, llm) -> AgentState:
    print("\n[FORMATTER] Polishing final answer...")

    format_prompt = FORMAT_PROMPT_TEMPLATE.format(
        raw_answer=state.raw_answer,
        n_text=len(state.text_results),
        n_table=len(state.table_results),
        n_metadata=len(state.metadata_results),
    )

    try:
        response = llm.invoke(format_prompt)
        state.final_answer = response.content.strip()
    except Exception as e:
        print(f"   Formatter failed ({e}), using raw answer.")
        state.final_answer = state.raw_answer

    return state
