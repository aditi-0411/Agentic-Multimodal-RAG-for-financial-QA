"""critic.py — CRITIC AGENT: reviews the raw answer for hallucination,
completeness, and grounding; triggers a bounded re-retrieval (ReAct-style
REFINE loop) if needed.

Bug fixes carried over from the original notebook (both were real bugs,
not just style preferences):
1. Never mutate `state.query` when refining — a separate `rewritten_query`
   is used for the refine pass, so the original user query is preserved
   for logging/evaluation.
2. Track `best_answer` across iterations so a REFINE pass that returns a
   worse ("not found") answer can't silently discard a good one.
"""

import re

from src.state import AgentState

NOT_FOUND_PHRASES = ["not found", "cannot be determined", "no information"]

CRITIC_PROMPT_TEMPLATE = """You are a strict QA critic for a financial question-answering system.

Question: {query}
Generated Answer: {answer}
Context (excerpt): {context_excerpt}

Evaluate on three criteria:
1. GROUNDED  — Is every claim in the answer supported by the context? (Yes/No)
2. COMPLETE  — Does the answer fully resolve the question? (Yes/No)
3. NUMERICAL — If numbers were available in context, were they used? (Yes/No/NA)

Verdict rules:
- PASS   : answer is grounded AND complete
- REFINE : answer is incomplete but context has more info — give a SHORT (max 8 words) search hint
- FAIL   : answer contradicts context or is hallucinated

Respond EXACTLY in this format:
GROUNDED: Yes/No
COMPLETE: Yes/No
NUMERICAL: Yes/No/NA
VERDICT: PASS/REFINE/FAIL
HINT: <max 8 words, the missing info to search for>"""


def critic_agent(state: AgentState, kb, llm, embedding_model, reranker) -> AgentState:
    # Imported here to avoid a circular import (specialists/synthesizer -> critic -> specialists)
    from src.agents.merger import merge_context
    from src.agents.specialists import metadata_agent, table_agent, text_agent
    from src.agents.synthesizer import synthesizer_agent

    print("\n[CRITIC] Evaluating answer quality...")

    if not state.best_answer:
        state.best_answer = state.raw_answer

    current_is_empty = any(p in state.raw_answer.lower() for p in NOT_FOUND_PHRASES)
    best_is_empty = any(p in state.best_answer.lower() for p in NOT_FOUND_PHRASES)

    if not current_is_empty:
        state.best_answer = state.raw_answer

    critic_prompt = CRITIC_PROMPT_TEMPLATE.format(
        query=state.query, answer=state.raw_answer, context_excerpt=state.merged_context[:1500]
    )

    try:
        resp = llm.invoke(critic_prompt)
        state.critic_feedback = resp.content.strip()
        print(f"   Critic:\n{state.critic_feedback}")

        verdict_m = re.search(r"VERDICT:\s*(PASS|REFINE|FAIL)", state.critic_feedback)
        hint_m = re.search(r"HINT:\s*(.+)", state.critic_feedback)
        verdict = verdict_m.group(1) if verdict_m else "PASS"
        hint = hint_m.group(1).strip()[:80] if hint_m else ""

        if verdict == "REFINE" and state.iterations < state.MAX_ITER:
            print(f"   REFINE (iter {state.iterations + 1}) — hint: '{hint}'")
            state.iterations += 1

            # FIX: use hint as a *separate* refined query — never touch state.query
            refined_query = state.query + " " + hint
            original_rewritten = state.rewritten_query
            state.rewritten_query = refined_query

            state = text_agent(state, kb, embedding_model, reranker)
            state = table_agent(state, kb, embedding_model, reranker)
            state = metadata_agent(state, kb, embedding_model, reranker)
            state = merge_context(state)
            state = synthesizer_agent(state, llm)

            state.rewritten_query = original_rewritten

            # FIX: if new answer is empty/worse, restore best answer
            new_is_empty = any(p in state.raw_answer.lower() for p in NOT_FOUND_PHRASES)
            if new_is_empty and not best_is_empty:
                print("   New answer is worse — restoring best answer.")
                state.raw_answer = state.best_answer
            else:
                state.best_answer = state.raw_answer

        elif verdict == "FAIL":
            print("   FAIL — appending disclaimer.")
            state.raw_answer = state.best_answer
            state.raw_answer += "\n\n[Critic flagged potential reliability issues. Verify against source documents.]"

        else:
            state.raw_answer = state.best_answer

    except Exception as e:
        print(f"   Critic error ({e}), passing through.")

    return state
