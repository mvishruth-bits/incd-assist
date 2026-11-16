import json
from pathlib import Path
from types import MappingProxyType

from incd_assist.contracts import IncidentScenario

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "scenarios.json"
METRIC_KEYS = frozenset(
    {
        "latency_p95_ms",
        "error_rate_pct",
        "cpu_pct",
        "requests_per_min",
        "db_connections_in_use",
        "db_pool_limit",
        "availability_pct",
    }
)


def load_scenarios(path: Path = DATA_PATH) -> dict[str, IncidentScenario]:
    """Load only agent-visible incident evidence, never evaluation answers."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Scenario data must be a list")
    scenarios: dict[str, IncidentScenario] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "id", "label", "metrics", "logs", "deployment_history", "previous_version"
        }:
            raise ValueError("Scenario has invalid fields")
        scenario_id = record["id"]
        metrics = record["metrics"]
        logs = record["logs"]
        history = record["deployment_history"]
        previous = record["previous_version"]
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in scenarios:
            raise ValueError(f"Invalid or duplicate scenario ID: {scenario_id!r}")
        if not isinstance(record["label"], str) or not record["label"]:
            raise ValueError(f"Scenario {scenario_id} needs a label")
        if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in metrics.values()
        ):
            raise ValueError(f"Scenario {scenario_id} has invalid metrics")
        if not isinstance(logs, list) or not all(isinstance(line, str) for line in logs):
            raise ValueError(f"Scenario {scenario_id} has invalid logs")
        if not isinstance(history, list) or not all(
            isinstance(deployment, dict)
            and set(deployment) == {"version", "status", "change"}
            and all(isinstance(value, str) for value in deployment.values())
            for deployment in history
        ):
            raise ValueError(f"Scenario {scenario_id} has invalid deployment history")
        if not isinstance(previous, dict) or set(previous) != {"version", "health", "config"} or not all(
            isinstance(value, str) for value in previous.values()
        ):
            raise ValueError(f"Scenario {scenario_id} has invalid previous version")
        scenarios[scenario_id] = IncidentScenario(
            id=scenario_id,
            label=record["label"],
            metrics=MappingProxyType(metrics),
            logs=tuple(logs),
            deployment_history=tuple(MappingProxyType(item) for item in history),
            previous_version=MappingProxyType(previous),
        )
    return scenarios
