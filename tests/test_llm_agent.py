
"""Automated tests for Tanweer's Ollama incident analysis agent."""

import pytest

from incd_assist.llm_agent import OllamaIncidentAgent
from incd_assist.scenarios import load_scenarios


@pytest.fixture
def agent():
    """Create an agent without making a live Ollama connection."""
    return OllamaIncidentAgent()


@pytest.fixture
def scenarios():
    """Load the repository's synthetic incident scenarios."""
    return load_scenarios()


def test_scenario_a_collects_db_exhaustion_evidence(agent, scenarios):
    scenario = scenarios["A"]
    agent._active_scenario = scenario
    agent.last_trace = []

    agent._collect_required_evidence(scenario)

    assert [entry["tool"] for entry in agent.last_trace] == [
        "get_metrics",
        "search_logs",
    ]

    assert agent.last_trace[1]["arguments"] == {"query": "DB"}
    assert "100/100" in agent.last_trace[1]["result"]


def test_scenario_b_collects_cpu_and_db_evidence(agent, scenarios):
    scenario = scenarios["B"]
    agent._active_scenario = scenario
    agent.last_trace = []

    agent._collect_required_evidence(scenario)

    assert [entry["tool"] for entry in agent.last_trace] == [
        "get_metrics",
        "search_logs",
        "search_logs",
    ]

    queries = [
        entry["arguments"]["query"]
        for entry in agent.last_trace
        if entry["tool"] == "search_logs"
    ]

    assert queries == ["CPU", "DB"]
    assert "30/100" in agent.last_trace[2]["result"]


def test_scenario_c_collects_config_and_historical_db_evidence(
    agent,
    scenarios,
):
    scenario = scenarios["C"]
    agent._active_scenario = scenario
    agent.last_trace = []

    agent._collect_required_evidence(scenario)

    assert [entry["tool"] for entry in agent.last_trace] == [
        "get_deployment_history",
        "check_previous_version",
        "search_logs",
        "search_logs",
    ]

    queries = [
        entry["arguments"]["query"]
        for entry in agent.last_trace
        if entry["tool"] == "search_logs"
    ]

    assert queries == ["AUTH_ISSUER", "DB"]
    assert "missing AUTH_ISSUER" in agent.last_trace[2]["result"]
    assert "resolved" in agent.last_trace[3]["result"]


def test_same_tool_can_be_called_with_different_arguments(
    agent,
    scenarios,
):
    scenario = scenarios["B"]
    agent._active_scenario = scenario
    agent.last_trace = []

    agent._collect_required_evidence(scenario)

    search_calls = [
        entry
        for entry in agent.last_trace
        if entry["tool"] == "search_logs"
    ]

    assert len(search_calls) == 2
    assert search_calls[0]["arguments"] != search_calls[1]["arguments"]


def test_numbered_verification_steps_are_parsed_separately():
    response = (
        "1. Check the configuration. "
        "2. Review the logs. "
        "3. Verify service health."
    )

    result = OllamaIncidentAgent._parse_list(response)

    assert result == (
        "Check the configuration.",
        "Review the logs.",
        "Verify service health.",
    )


def test_bulleted_verification_steps_are_parsed_separately():
    response = (
        "- Check the configuration.\n"
        "- Review the logs.\n"
        "- Verify service health."
    )

    result = OllamaIncidentAgent._parse_list(response)

    assert result == (
        "Check the configuration.",
        "Review the logs.",
        "Verify service health.",
    )


def test_empty_verification_steps_return_empty_tuple():
    assert OllamaIncidentAgent._parse_list("") == ()


def test_ollama_failure_is_reported_clearly(
    agent,
    scenarios,
    monkeypatch,
):
    scenario = scenarios["A"]

    def fail_invoke(self, messages, *args, **kwargs):
        raise ConnectionError("Ollama is unavailable")

    # Patch the ChatOllama class method, not the Pydantic model instance.
    monkeypatch.setattr(
        "incd_assist.llm_agent.ChatOllama.invoke",
        fail_invoke,
    )

    with pytest.raises(
        RuntimeError,
        match="Ollama incident analysis failed for scenario A",
    ):
        agent.decide(scenario)
