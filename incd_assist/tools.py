from incd_assist.contracts import IncidentScenario


def get_metrics(scenario: IncidentScenario) -> dict[str, int | float]:
    return dict(scenario.metrics)


def search_logs(scenario: IncidentScenario, query: str = "") -> list[str]:
    if not isinstance(query, str):
        raise TypeError("Log query must be a string")
    return [line for line in scenario.logs if query.casefold() in line.casefold()]


def get_deployment_history(scenario: IncidentScenario) -> list[dict[str, str]]:
    return [dict(deployment) for deployment in scenario.deployment_history]


def check_previous_version(scenario: IncidentScenario) -> dict[str, str]:
    return dict(scenario.previous_version)
