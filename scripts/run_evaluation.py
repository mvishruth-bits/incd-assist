"""Reproduce the 18-row comparison using a live local Ollama model."""

import csv
import json
import tempfile
from pathlib import Path

from incd_assist.classical_agents import LearningAgent
from incd_assist.comparison import comparison_rows, load_expected, local_history_path, run_comparison
from incd_assist.scenarios import load_scenarios


def main() -> None:
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    rows = []
    traces = {}
    for scenario in load_scenarios().values():
        results, _, trace = run_comparison(scenario, local_history_path())
        rows.extend(comparison_rows(scenario, results))
        traces[scenario.id] = trace
        print(f"{scenario.id}: {len(results)} agents, {len(trace)} actual Ollama calls", flush=True)
    if len(rows) != 18:
        raise ValueError(f"Expected 18 rows, received {len(rows)}")
    report = {
        "results": rows, "tool_traces": traces,
        "evaluation_reference_not_sent_to_agents": load_expected(),
    }
    (output / "comparison_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "comparison_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "scenario", "agent_name", "likely_cause", "recommended_action",
            "reason", "evidence", "verification_steps", "limitations",
        ])
        writer.writeheader()
        for result in rows:
            writer.writerow({key: " | ".join(value) if isinstance(value, list) else value
                             for key, value in result.items() if key in writer.fieldnames})
    with tempfile.TemporaryDirectory() as directory:
        history = Path(directory) / "feedback.json"
        history.write_bytes((Path("data") / "learning_agent_experience.json").read_bytes())
        learning = LearningAgent(history)
        scenario = load_scenarios()["A"]
        before = learning.decide(scenario)
        action = next(name for name in learning.available_actions(scenario)
                      if before.recommended_action.startswith(name))
        learning.record_feedback(scenario, action, successful=False, notes="SIMULATED evaluation feedback")
        after = learning.decide(scenario)
        demo = {
            "scenario": "A", "feedback": {"action": action, "outcome": "simulated failure"},
            "before_action": before.recommended_action, "before_score": before.evidence[-1],
            "after_action": after.recommended_action, "after_score": after.evidence[-1],
        }
    (output / "learning_feedback_demo.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
    print("Saved 18-row comparison, real tool traces and learning feedback demo in outputs/", flush=True)


if __name__ == "__main__":
    main()
