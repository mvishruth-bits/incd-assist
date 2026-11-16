from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class IncidentScenario:
    id: str
    label: str
    metrics: Mapping[str, int | float]
    logs: tuple[str, ...]
    deployment_history: tuple[Mapping[str, str], ...]
    previous_version: Mapping[str, str]


@dataclass(frozen=True)
class AgentResult:
    agent_name: str
    incident_summary: str
    evidence: tuple[str, ...]
    likely_cause: str
    recommended_action: str
    verification_steps: tuple[str, ...]
    limitations: tuple[str, ...]
    reason: str


class Agent(Protocol):
    def decide(self, scenario: IncidentScenario) -> AgentResult: ...
