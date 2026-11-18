
"""Run Tanweer's Ollama agent against all three incident scenarios."""

import json
from pathlib import Path

from incd_assist.llm_agent import OllamaIncidentAgent
from incd_assist.scenarios import load_scenarios


def main() -> None:
    scenarios = load_scenarios()
    agent = OllamaIncidentAgent()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    report = {}

    for scenario_id in ("A", "B", "C"):
        scenario = scenarios[scenario_id]

        print(f"\n{'=' * 65}")
        print(f"SCENARIO {scenario_id}: {scenario.label}")
        print("=" * 65)

        result = agent.decide(scenario)

        report[scenario_id] = {
            "agent_name": result.agent_name,
            "incident_summary": result.incident_summary,
            "evidence": list(result.evidence),
            "likely_cause": result.likely_cause,
            "recommended_action": result.recommended_action,
            "verification_steps": list(result.verification_steps),
            "limitations": list(result.limitations),
            "reason": result.reason,
            "tool_trace": agent.last_trace.copy(),
        }

        print("\nIncident summary:", result.incident_summary)
        print("Likely cause:", result.likely_cause)
        print("Recommended action:", result.recommended_action)

        print("\nActual tool calls:")
        for index, call in enumerate(agent.last_trace, start=1):
            print(f"{index}. {call['tool']}({call['arguments']})")
            print("   Result:", call["result"])

        print("\nVerification steps:")
        for step in result.verification_steps:
            print("-", step)

        print("\nLimitations:")
        for limitation in result.limitations:
            print("-", limitation)

    report_path = output_dir / "tanweer_llm_agent_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
