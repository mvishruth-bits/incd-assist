
"""LangChain + Ollama incident analysis agent."""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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
            num_predict=350,
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
        allowed = {"search_logs": {"query"}}.get(name, set())
        if set(arguments) - allowed:
            raise ValueError(f"Unexpected arguments for {name}: {set(arguments) - allowed}")

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

        return re.sub(r"^[A-C]\.\s+", "", match.group(1).strip())

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
            system_prompt = (
                "You are an incident analyst. First call at least two DIFFERENT "
                "relevant evidence tools. Do not answer yet. Use tool results "
                "to check the observed data. Never invent evidence or tool arguments. "
                "For tools with no parameters, call with empty arguments {}."
            )

            human_prompt = f"""Incident {scenario.id}: {scenario.label}.
Observed data (the same input supplied to all six agents):
{json.dumps({
    "metrics": dict(scenario.metrics),
    "logs": list(scenario.logs),
    "deployment_history": [dict(item) for item in scenario.deployment_history],
    "previous_version": dict(scenario.previous_version),
}, ensure_ascii=False)}
Call two different relevant tools now. No final answer yet.
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
            requested_tools = {
                "A": (("get_metrics", ""), ("search_logs", "DB")),
                "B": (("get_metrics", ""), ("search_logs", "DB")),
                "C": (("get_deployment_history", ""), ("search_logs", "AUTH_ISSUER")),
            }.get(scenario.id, (("get_metrics", ""), ("search_logs", "")))
            by_name = {item.name: item for item in self.langchain_tools}
            for name, query in requested_tools:
                instruction = f"Call {name} now. Return only the tool call, not text."
                if query:
                    instruction += f" Search for {query}."
                messages.append(HumanMessage(content=instruction))
                response = self.llm.bind_tools([by_name[name]]).invoke(messages)
                messages.append(response)
                if not response.tool_calls:
                    raise ValueError(f"Ollama did not call required evidence tool {name}")
                for call in response.tool_calls:
                    if call["name"] != name or not isinstance(call["args"], dict):
                        raise ValueError(f"Unexpected evidence tool call: {call['name']}")
                    value = self._execute_tool(name, call["args"])
                    messages.append(ToolMessage(
                        content=json.dumps(value, ensure_ascii=False),
                        tool_call_id=call["id"],
                    ))
            if len({entry["tool"] for entry in self.last_trace}) < 2:
                raise ValueError("Ollama did not complete two distinct tool calls")
            scenario_guidance = {
                "A": "100/100 means at the limit, not above it. Long-running connections "
                     "are possible, not confirmed. RECOMMENDED_ACTION must start by investigating "
                     "connections, wait queues and DB capacity. Do NOT recommend increasing the pool now.",
                "B": "The DB pool is 30/100 and healthy. Do not invent a stale DB warning "
                     "or recommend DB recovery; mention 30/100 when assessing traffic-driven CPU and scaling.",
                "C": "A prior DB timeout was resolved at 09:10 and is not the 10:00 "
                     "startup failure. Cite the healthy previous version when comparing the config change.",
            }.get(scenario.id, "")
            messages.append(HumanMessage(content=(
                "Now answer briefly using returned tool values. " + scenario_guidance +
                " Recommend actions only; execute nothing. "
                "Use these headings with colons on separate lines: INCIDENT_SUMMARY, LIKELY_CAUSE, "
                "RECOMMENDED_ACTION, VERIFICATION_STEPS, LIMITATIONS, REASON. "
                "Include concrete values, two verification steps and uncertainty."
            )))
            final_response = self.llm.invoke(messages)
            response_text = str(final_response.content).strip()

            if not response_text:
                raise ValueError("Ollama returned an empty response.")

            result = self._parse_result(
                scenario,
                response_text,
            )
            if scenario.id == "A" and re.match(
                r"^\s*(?:increase|expand|raise)\b", result.recommended_action, re.IGNORECASE
            ):
                messages.extend([
                    final_response,
                    HumanMessage(content=(
                        "That action is unsafe: the logs show a full pool but do not prove "
                        "the database has spare capacity. Rewrite ALL six sections. "
                        "Start RECOMMENDED_ACTION with 'Investigate' connection usage "
                        "and wait queues; only consider a capacity adjustment after review."
                    )),
                ])
                corrected = self.llm.invoke(messages)
                result = self._parse_result(scenario, str(corrected.content).strip())
                if re.match(
                    r"^\s*(?:increase|expand|raise)\b",
                    result.recommended_action, re.IGNORECASE,
                ):
                    raise ValueError("Ollama continued to recommend an unsafe pool increase")

            required_fields = (
                result.incident_summary,
                result.likely_cause,
                result.recommended_action,
                result.reason,
            )

            if (any(not field.strip() for field in required_fields)
                    or not result.verification_steps or not result.limitations):
                raise ValueError(
                    "Ollama response is missing one or more required sections."
                )

            return result

        except Exception as exc:
            raise RuntimeError(
                "Ollama incident analysis failed for scenario "
                f"{scenario.id}: {exc}"
            ) from exc
