"""Run all agents on one unchanged scenario and prepare reviewable results."""

import json
from dataclasses import asdict
from pathlib import Path

from incd_assist.classical_agents import LearningAgent, create_classical_agents
from incd_assist.contracts import AgentResult, IncidentScenario
from incd_assist.llm_agent import OllamaIncidentAgent

ROOT = Path(__file__).resolve().parent.parent
SEED_HISTORY = ROOT / "data" / "learning_agent_experience.json"
LOCAL_HISTORY = ROOT / "data" / "learning_agent_experience_integration.json"


def local_history_path() -> Path:
    if not LOCAL_HISTORY.exists():
        LOCAL_HISTORY.write_bytes(SEED_HISTORY.read_bytes())
    return LOCAL_HISTORY


def run_comparison(
    scenario: IncidentScenario,
    history_path: Path,
    llm_agent: OllamaIncidentAgent | None = None,
) -> tuple[dict[str, AgentResult], LearningAgent, list[dict]]:
    agents = create_classical_agents(experience_path=history_path)
    results = {name: agent.decide(scenario) for name, agent in agents.items()}
    llm = llm_agent if llm_agent is not None else OllamaIncidentAgent()
    llm_result = llm.decide(scenario)
    results[llm_result.agent_name] = llm_result
    if len(results) != 6 or len({call["tool"] for call in llm.last_trace}) < 2:
        raise ValueError("Comparison requires six agents and two distinct real LLM tool calls")
    return results, agents["Learning Agent"], list(llm.last_trace)


def comparison_rows(scenario: IncidentScenario, results: dict[str, AgentResult]) -> list[dict]:
    if len(results) != 6:
        raise ValueError("Six agent results are required for a comparison")
    return [
        {"scenario": scenario.id, **asdict(result)}
        for result in results.values()
    ]


def load_expected() -> dict:
    return json.loads((ROOT / "data" / "expected_results.json").read_text(encoding="utf-8"))
