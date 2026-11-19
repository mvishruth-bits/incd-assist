"""Verify that tool traces come from Ollama requests, not preloaded evidence."""

import pytest
from langchain_core.messages import AIMessage

from incd_assist.llm_agent import OllamaIncidentAgent
from incd_assist.scenarios import load_scenarios


@pytest.fixture
def scenarios():
    return load_scenarios()


FINAL = (
    "INCIDENT_SUMMARY: Elevated latency and errors.\n"
    "LIKELY_CAUSE: DB connection pool exhaustion (100/100).\n"
    "RECOMMENDED_ACTION: Investigate long-running connections before changing capacity.\n"
    "VERIFICATION_STEPS: Check queue length; verify errors fall.\n"
    "LIMITATIONS: This is simulated evidence only.\n"
    "REASON: Pool 100/100 and DB timeout logs indicate saturation."
)


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def bind_tools(self, tools):
        assert {tool.name for tool in tools} <= {
            "get_metrics", "search_logs", "get_deployment_history", "check_previous_version"
        }
        return self

    def invoke(self, messages):
        self.messages.append(list(messages))
        return next(self.responses)


def tool_call(name, args, index):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{index}"}])


def test_model_calls_two_distinct_real_tools_before_answer(scenarios):
    agent = OllamaIncidentAgent()
    model = FakeModel([
        tool_call("get_metrics", {}, 1),
        tool_call("search_logs", {"query": "DB"}, 2),
        AIMessage(content=FINAL),
    ])
    agent.llm = model
    result = agent.decide(scenarios["A"])

    assert [entry["tool"] for entry in agent.last_trace] == ["get_metrics", "search_logs"]
    assert all(entry["scenario_id"] == "A" for entry in agent.last_trace)
    assert "100/100" in agent.last_trace[1]["result"]
    assert "100/100" in result.likely_cause
    assert model.messages[0][-1].type == "human"
    assert model.messages[-1][-2].type == "tool"
    assert model.messages[-1][-1].type == "human"


def test_model_must_call_two_distinct_tools(scenarios):
    agent = OllamaIncidentAgent()
    agent.llm = FakeModel([tool_call("get_metrics", {}, 1), tool_call("get_metrics", {}, 2)])
    with pytest.raises(RuntimeError, match="Unexpected evidence tool call"):
        agent.decide(scenarios["B"])


def test_unknown_tool_fails_instead_of_faking_evidence(scenarios):
    agent = OllamaIncidentAgent()
    agent.llm = FakeModel([tool_call("invented_tool", {}, 1)])
    with pytest.raises(RuntimeError, match="Unexpected evidence tool call"):
        agent.decide(scenarios["C"])
    assert not agent.last_trace


def test_unexpected_tool_arguments_fail_instead_of_being_ignored(scenarios):
    agent = OllamaIncidentAgent()
    agent.llm = FakeModel([
        tool_call("get_deployment_history", {"version": "invented"}, 1)
    ])
    with pytest.raises(RuntimeError, match="Unexpected arguments for get_deployment_history"):
        agent.decide(scenarios["C"])
    assert not agent.last_trace


def test_missing_sections_fail_even_after_tool_calls(scenarios):
    agent = OllamaIncidentAgent()
    agent.llm = FakeModel([
        tool_call("get_deployment_history", {}, 1),
        tool_call("search_logs", {"query": "AUTH_ISSUER"}, 2),
        AIMessage(content="No structured answer"),
    ])
    with pytest.raises(RuntimeError, match="missing one or more required sections"):
        agent.decide(scenarios["C"])


def test_unsafe_pool_increase_is_revised_using_existing_tool_evidence(scenarios):
    unsafe = FINAL.replace(
        "Investigate long-running connections before changing capacity.",
        "Increase DB pool size immediately.",
    )
    agent = OllamaIncidentAgent()
    model = FakeModel([
        tool_call("get_metrics", {}, 1),
        tool_call("search_logs", {"query": "DB"}, 2),
        AIMessage(content=unsafe),
        AIMessage(content=FINAL),
    ])
    agent.llm = model
    result = agent.decide(scenarios["A"])
    assert result.recommended_action.startswith("Investigate")
    assert len(model.messages) == 4
    assert [entry["tool"] for entry in agent.last_trace] == ["get_metrics", "search_logs"]


def test_numbered_verification_steps_are_parsed_separately():
    response = "1. Check config. 2. Review logs. 3. Verify service health."
    assert OllamaIncidentAgent._parse_list(response) == (
        "Check config.", "Review logs.", "Verify service health."
    )


def test_bulleted_verification_steps_are_parsed_separately():
    response = "- Check config.\n- Review logs.\n- Verify service health."
    assert OllamaIncidentAgent._parse_list(response) == (
        "Check config.", "Review logs.", "Verify service health."
    )


def test_ollama_failure_is_reported_clearly(scenarios):
    class OfflineModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise ConnectionError("Ollama is unavailable")

    agent = OllamaIncidentAgent()
    agent.llm = OfflineModel()
    with pytest.raises(RuntimeError, match="Ollama incident analysis failed for scenario A"):
        agent.decide(scenarios["A"])
