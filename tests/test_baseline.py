import ast
import json
from pathlib import Path

import pytest

from incd_assist.scenarios import load_scenarios
from incd_assist.tools import (
    check_previous_version,
    get_deployment_history,
    get_metrics,
    search_logs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_export_matches_committed_data(tmp_path: Path) -> None:
    notebook = json.loads((ROOT / "notebooks" / "incident_scenarios.ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        tree.body = [
            node for node in tree.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "export")
        ]
        exec(compile(tree, "incident_scenarios.ipynb", "exec"), namespace)
    namespace["export"](tmp_path)
    for filename in ("scenarios.json", "expected_results.json"):
        assert (tmp_path / "data" / filename).read_bytes() == (ROOT / "data" / filename).read_bytes()


def test_three_scenarios_share_complete_visible_evidence_without_answers() -> None:
    scenarios = load_scenarios()
    expected = json.loads((ROOT / "data" / "expected_results.json").read_text(encoding="utf-8"))
    visible = (ROOT / "data" / "scenarios.json").read_text(encoding="utf-8")
    assert set(scenarios) == set(expected) == {"A", "B", "C"}
    for scenario in scenarios.values():
        assert scenario.logs and scenario.deployment_history and scenario.previous_version
        assert scenario.metrics["db_connections_in_use"] <= scenario.metrics["db_pool_limit"]
    assert scenarios["A"].metrics["db_connections_in_use"] == scenarios["A"].metrics["db_pool_limit"]
    assert scenarios["B"].metrics["db_connections_in_use"] < scenarios["B"].metrics["db_pool_limit"]
    assert scenarios["B"].metrics["availability_pct"] > 99
    assert scenarios["C"].metrics["availability_pct"] == 0
    for answer in expected.values():
        assert answer["safe_action"] not in visible
        assert answer["likely_cause"] not in visible


def test_tools_return_selected_evidence_without_mutating_scenario() -> None:
    scenarios = load_scenarios()
    assert "pool 100/100" in search_logs(scenarios["A"], "timeout")[0]
    assert not search_logs(scenarios["B"], "pool 100/100")
    assert len(search_logs(scenarios["C"], "db transient")) == 1
    assert get_deployment_history(scenarios["C"])[0]["status"] == "failed startup"
    assert check_previous_version(scenarios["C"])["health"] == "healthy before config change"
    metrics = get_metrics(scenarios["A"])
    metrics["db_connections_in_use"] = 0
    assert get_metrics(scenarios["A"])["db_connections_in_use"] == 100
    with pytest.raises(TypeError):
        scenarios["A"].metrics["db_connections_in_use"] = 0


def test_loader_rejects_duplicate_or_malformed_scenarios(tmp_path: Path) -> None:
    records = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([records[0], records[0]]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_scenarios(path)
    records[0]["metrics"]["db_pool_limit"] = "unknown"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics"):
        load_scenarios(path)
