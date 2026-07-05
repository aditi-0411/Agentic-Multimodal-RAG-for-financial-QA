"""merger.py — CONTEXT MERGER: assembles the final labelled context string
from all active specialist agents' results."""

from src.state import AgentState


def merge_context(state: AgentState) -> AgentState:
    print("\n[MERGER] Building merged context...")

    sections = []

    if state.text_results:
        sections.append("=== NARRATIVE TEXT ===")
        for i, doc in enumerate(state.text_results):
            sections.append(f"[Text {i+1}]\n{doc['content']}")

    if state.table_results:
        sections.append("\n=== TABLES & NUMERICAL DATA ===")
        for i, doc in enumerate(state.table_results):
            sections.append(f"[Table {i+1}]\n{doc['content']}")

    if state.metadata_results:
        sections.append("\n=== COMPANY METADATA ===")
        for i, doc in enumerate(state.metadata_results):
            sections.append(f"[Meta {i+1}]\n{doc['content']}")

    state.merged_context = "\n\n".join(sections)

    print(f"   Context length: {len(state.merged_context)} chars")
    return state
