
"""LangChain + Ollama incident analysis agent."""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from incd_assist.contracts import AgentResult, IncidentScenario
from incd_assist import tools as incident_tools


class OllamaIncidentAgent:
    """Analyze synthetic incidents using local Ollama and project evidence tools."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0,
    ) -> None:
        self.model = model
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )
        self.last_trace: list[dict[str, Any]] = []
        self.langchain_tools = self._build_tools()

    def _build_tools(self):
        """Create LangChain tool wrappers for the project evidence tools."""

        @tool
        def get_metrics() -> str:
            """Retrieve metrics for the current incident scenario."""
            return json.dumps(
                incident_tools.get_metrics(self._active_scenario),
                ensure_ascii=False,
            )

        @tool
        def search_logs(query: str = "") -> str:
            """Search incident logs using a keyword or phrase."""
            return json.dumps(
                incident_tools.search_logs(
                    self._active_scenario,
                    query=query,
                ),
                ensure_ascii=False,
            )

        @tool
        def get_deployment_history() -> str:
            """Retrieve deployment history for the current scenario."""
            return json.dumps(
                incident_tools.get_deployment_history(
                    self._active_scenario
                ),
                ensure_ascii=False,
            )

        @tool
        def check_previous_version() -> str:
            """Check the health and configuration of the previous version."""
            return json.dumps(
                incident_tools.check_previous_version(
                    self._active_scenario
                ),
                ensure_ascii=False,
            )

        return [
            get_metrics,
            search_logs,
            get_deployment_history,
            check_previous_version,
        ]

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a real project evidence tool and record its output."""

        tool_functions = {
            "get_metrics": lambda: incident_tools.get_metrics(
                self._active_scenario
            ),
            "search_logs": lambda: incident_tools.search_logs(
                self._active_scenario,
                query=arguments.get("query", ""),
            ),
            "get_deployment_history": lambda: (
                incident_tools.get_deployment_history(
                    self._active_scenario
                )
            ),
            "check_previous_version": lambda: (
                incident_tools.check_previous_version(
                    self._active_scenario
                )
            ),
        }

        if name not in tool_functions:
            raise ValueError(f"Unknown evidence tool: {name}")

        result = tool_functions[name]()

        self.last_trace.append(
            {
                "scenario_id": self._active_scenario.id,
                "tool": name,
                "arguments": arguments.copy(),
                "result": json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                ),
            }
        )

        return result

    def _collect_required_evidence(
        self,
        scenario: IncidentScenario,
    ) -> None:
        """Collect relevant evidence for each known incident scenario."""

        def execute_if_missing(
            name: str,
            arguments: dict[str, Any],
        ) -> None:
            already_called = any(
                entry["tool"] == name
                and entry["arguments"] == arguments
                for entry in self.last_trace
            )

            if not already_called:
                self._execute_tool(name, arguments)

        if scenario.id == "A":
            execute_if_missing("get_metrics", {})
            execute_if_missing(
                "search_logs",
                {"query": "DB"},
            )

        elif scenario.id == "B":
            execute_if_missing("get_metrics", {})
            execute_if_missing(
                "search_logs",
                {"query": "CPU"},
            )
            execute_if_missing(
                "search_logs",
                {"query": "DB"},
            )

        elif scenario.id == "C":
            execute_if_missing(
                "get_deployment_history",
                {},
            )
            execute_if_missing(
                "check_previous_version",
                {},
            )
            execute_if_missing(
                "search_logs",
                {"query": "AUTH_ISSUER"},
            )
            execute_if_missing(
                "search_logs",
                {"query": "DB"},
            )

        else:
            execute_if_missing("get_metrics", {})
            execute_if_missing(
                "search_logs",
                {"query": ""},
            )

    @staticmethod
    def _extract_section(
        response: str,
        section_name: str,
        next_sections: tuple[str, ...],
    ) -> str:
        """Extract one named section from the LLM response."""

        pattern = rf"{re.escape(section_name)}\s*:\s*(.*?)(?="

        if next_sections:
            following = "|".join(
                re.escape(name) for name in next_sections
            )
            pattern += rf"\s*(?:{following})\s*:|$)"
        else:
            pattern += r"$)"

        match = re.search(
            pattern,
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not match:
            return ""

        return match.group(1).strip()

    @staticmethod
    def _parse_list(value: str) -> tuple[str, ...]:
        """Parse numbered, bulleted, or semicolon-separated LLM lists."""

        if not value:
            return ()

        normalized = value.strip()

        # First handle numbered or bulleted steps on separate lines.
        lines = [
            line.strip()
            for line in normalized.splitlines()
            if line.strip()
        ]

        if len(lines) > 1:
            items = []
            for line in lines:
                cleaned = re.sub(
                    r"^(?:[-*•]\s*|\d+[.)]\s*)",
                    "",
                    line,
                ).strip()
                if cleaned:
                    items.append(cleaned)

            return tuple(items)

        # Also handle models that return numbered steps on one line.
        numbered = re.split(
            r"\s+\d+[.)]\s+",
            normalized,
        )

        if len(numbered) > 1:
            items = [
                re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", item).strip()
                for item in numbered
            ]
            return tuple(item for item in items if item)

        # Finally support semicolon-separated responses.
        items = [
            item.strip(" -*•\t")
            for item in normalized.split(";")
        ]

        return tuple(item for item in items if item)

    def _parse_result(
        self,
        scenario: IncidentScenario,
        response: str,
    ) -> AgentResult:
        """Convert the LLM response into the project's AgentResult contract."""

        sections = {
            "INCIDENT_SUMMARY": (
                "LIKELY_CAUSE",
                "RECOMMENDED_ACTION",
                "VERIFICATION_STEPS",
                "LIMITATIONS",
                "REASON",
            ),
            "LIKELY_CAUSE": (
                "RECOMMENDED_ACTION",
                "VERIFICATION_STEPS",
                "LIMITATIONS",
                "REASON",
            ),
            "RECOMMENDED_ACTION": (
                "VERIFICATION_STEPS",
                "LIMITATIONS",
                "REASON",
            ),
            "VERIFICATION_STEPS": (
                "LIMITATIONS",
                "REASON",
            ),
            "LIMITATIONS": ("REASON",),
            "REASON": (),
        }

        parsed = {
            name: self._extract_section(
                response,
                name,
                next_sections,
            )
            for name, next_sections in sections.items()
        }

        evidence = tuple(
            (
                f"{entry['tool']}("
                f"{json.dumps(entry['arguments'], ensure_ascii=False)}"
                f") -> {entry['result']}"
            )
            for entry in self.last_trace
        )

        return AgentResult(
            agent_name=f"LangChain Ollama ({self.model})",
            incident_summary=parsed["INCIDENT_SUMMARY"],
            evidence=evidence,
            likely_cause=parsed["LIKELY_CAUSE"],
            recommended_action=parsed["RECOMMENDED_ACTION"],
            verification_steps=self._parse_list(
                parsed["VERIFICATION_STEPS"]
            ),
            limitations=self._parse_list(
                parsed["LIMITATIONS"]
            ),
            reason=parsed["REASON"],
        )

    def decide(
        self,
        scenario: IncidentScenario,
    ) -> AgentResult:
        """Collect evidence and ask Ollama to analyze the incident."""

        self.last_trace = []
        self._active_scenario = scenario

        try:
            self._collect_required_evidence(scenario)

            evidence_json = json.dumps(
                self.last_trace,
                indent=2,
                ensure_ascii=False,
            )

            system_prompt = """
You are an incident analysis assistant.

Analyze the supplied synthetic incident evidence carefully.

Rules:
- Use only the supplied evidence.
- Distinguish symptoms from root causes.
- Respect timestamps and event chronology.
- Do not treat an earlier resolved warning as the current root cause.
- Compare the current deployment with the previous healthy version.
- Do not invent metrics, logs, configuration values, or events.
- Recommendations must be advisory only.
- Never claim that you executed a recovery action.
- For database pool exhaustion, recommend investigating query duration,
  connection usage, concurrency, and database capacity before changing
  the pool limit.
- For high CPU and high traffic, distinguish traffic pressure from
  database connection exhaustion.
- Explain uncertainty and limitations.

Return exactly these sections:

INCIDENT_SUMMARY:
A concise summary.

LIKELY_CAUSE:
The most likely cause, supported by evidence.

RECOMMENDED_ACTION:
A safe advisory recommendation. Do not execute it.

VERIFICATION_STEPS:
Provide three separate numbered verification steps.

LIMITATIONS:
State the important limitations of the analysis.

REASON:
Explain why the evidence supports your conclusion.
"""

            human_prompt = f"""
Analyze this incident.

Scenario ID: {scenario.id}
Scenario label: {scenario.label}

Actual evidence tool calls and their results:
{evidence_json}

Pay particular attention to the event chronology.
For scenario C, explicitly explain whether the earlier DB warning
is related to the later AUTH_ISSUER startup failure.
For scenario B, explain what the DB connection metrics and DB logs
indicate about database pool health.
"""

            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )

            response_text = str(response.content).strip()

            if not response_text:
                raise ValueError("Ollama returned an empty response.")

            result = self._parse_result(
                scenario,
                response_text,
            )

            required_fields = (
                result.incident_summary,
                result.likely_cause,
                result.recommended_action,
                result.reason,
            )

            if any(not field.strip() for field in required_fields):
                raise ValueError(
                    "Ollama response is missing one or more required sections."
                )

            return result

        except Exception as exc:
            raise RuntimeError(
                "Ollama incident analysis failed for scenario "
                f"{scenario.id}: {exc}"
            ) from exc
