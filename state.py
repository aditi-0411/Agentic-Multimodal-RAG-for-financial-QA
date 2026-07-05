"""
state.py — the shared state object threaded through the entire agent
pipeline (Planner -> Router -> specialist agents -> Critic -> Synthesizer
-> Formatter).
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentState:
    query: str = ""
    query_type: str = "general"  # numeric | financial | metadata | general
    rewritten_query: str = ""
    plan: List[str] = field(default_factory=list)
    agents_to_call: List[str] = field(default_factory=list)
    text_results: List[Dict] = field(default_factory=list)
    table_results: List[Dict] = field(default_factory=list)
    metadata_results: List[Dict] = field(default_factory=list)
    merged_context: str = ""
    raw_answer: str = ""
    best_answer: str = ""
    critic_feedback: str = ""
    final_answer: str = ""
    iterations: int = 0
    MAX_ITER: int = 2
