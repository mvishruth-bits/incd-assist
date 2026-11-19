import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from incd_assist.classical_agents import LearningAgent, create_classical_agents
from incd_assist.comparison import comparison_rows, run_comparison
from incd_assist.contracts import AgentResult
from incd_assist.scenarios import load_scenarios


class FixtureLLM:
    last_trace = [
        {"tool": "get_metrics", "result": '{"cpu_pct": 92}', "arguments": {}, "scenario_id": "B"},
        {"tool": "search_logs", "result": '["cpu 92 percent"]', "arguments": {"query": "cpu"}, "scenario_id": "B"},
    ]

    def decide(self, scenario):
        assert scenario.id == "B"
        return AgentResult(
            agent_name="LangChain Ollama (fixture)", incident_summary="High CPU",
            evidence=("CPU 92%", "DB 30/100"), likely_cause="Traffic-driven CPU pressure",
            recommended_action="Consider controlled scaling", verification_steps=("Check CPU",),
            limitations=("Synthetic data",), reason="High traffic with healthy DB pool",
        )


@pytest.mark.parametrize(
    ("scenario_id", "cause_terms", "avoid_action"),
    [
        ("A", ("database", "pool"), "restart all"),
        ("B", ("traffic", "cpu"), "rollback"),
        ("C", ("startup", "auth_issuer"), "restart application without"),
    ],
)
def test_five_classical_agents_match_incident_evidence(tmp_path, scenario_id, cause_terms, avoid_action):
    scenario = load_scenarios()[scenario_id]
    agents = create_classical_agents(experience_path=tmp_path / "feedback.json")
    assert len(agents) == 5
    for agent in agents.values():
        result = agent.decide(scenario)
        cause = result.likely_cause.lower()
        assert all(term in cause for term in cause_terms), result.agent_name
        assert avoid_action not in result.recommended_action.lower(), result.agent_name
        assert result.evidence and result.verification_steps and result.limitations


def test_six_agents_use_same_scenario_and_feedback_changes_score(tmp_path):
    scenario = load_scenarios()["B"]
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps({"schema_version": 1, "feedback": {}}), encoding="utf-8")
    results, learning, trace = run_comparison(scenario, path, FixtureLLM())
    assert len(comparison_rows(scenario, results)) == 6
    assert [item["tool"] for item in trace] == ["get_metrics", "search_logs"]
    before = results["Learning Agent"]
    action = learning.available_actions(scenario)[0]
    learning.record_feedback(scenario, action, successful=False, notes="SIMULATED")
    after = LearningAgent(path).decide(scenario)
    assert before.evidence[-1] != after.evidence[-1]
    assert before.recommended_action != after.recommended_action
    assert scenario.metrics["cpu_pct"] == 92


def test_streamlit_scenario_selector_renders_without_ollama():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=120
    ).run()
    assert not app.exception
    assert len(app.selectbox) == 1
    assert [option[0] for option in app.selectbox[0].options] == ["A", "B", "C"]


def test_streamlit_displays_six_results_and_feedback(tmp_path, monkeypatch):
    history = tmp_path / "feedback.json"
    history.write_text(json.dumps({"schema_version": 1, "feedback": {}}), encoding="utf-8")
    monkeypatch.setattr("incd_assist.comparison.local_history_path", lambda: history)
    monkeypatch.setattr(
        "incd_assist.comparison.run_comparison",
        lambda scenario, path: run_comparison(scenario, path, FixtureLLM()),
    )
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=120
    ).run()
    app.selectbox[0].set_value("B").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert len(app.expander) == 6
    assert len(app.selectbox) == 2
    app.button[1].click().run()
    assert not app.exception
    assert "traffic" in json.loads(history.read_text(encoding="utf-8"))["feedback"]
