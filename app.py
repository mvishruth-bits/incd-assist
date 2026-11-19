"""Interactive, recommendation-only comparison of six incident agents."""

import json

import streamlit as st

from incd_assist.classical_agents import LearningAgent
from incd_assist.comparison import comparison_rows, load_expected, local_history_path, run_comparison
from incd_assist.scenarios import load_scenarios


def main() -> None:
    st.set_page_config(page_title="Incident agent comparison", layout="wide")
    st.title("Production incident assistant")
    st.caption("Synthetic evidence only. Recommendations are advisory; no recovery action is executed.")
    scenarios = load_scenarios()
    scenario_id = st.selectbox("Incident scenario", list(scenarios), format_func=lambda key: f"{key}: {scenarios[key].label}")
    scenario = scenarios[scenario_id]
    st.json({
        "metrics": dict(scenario.metrics), "logs": list(scenario.logs),
        "deployment_history": [dict(item) for item in scenario.deployment_history],
        "previous_version": dict(scenario.previous_version),
    })

    if st.button("Compare all six agents", type="primary"):
        with st.spinner("Running five classical agents and local Ollama..."):
            try:
                results, _, trace = run_comparison(scenario, local_history_path())
            except (RuntimeError, ValueError, OSError) as exc:
                st.session_state.pop("comparison", None)
                st.error(f"Comparison incomplete: {exc}")
            else:
                st.session_state["comparison"] = (scenario_id, results, trace)

    comparison = st.session_state.get("comparison")
    if comparison is None or comparison[0] != scenario_id:
        return
    _, results, trace = comparison
    st.subheader("Side-by-side comparison")
    st.dataframe(
        [
            {"Agent": result.agent_name, "Likely cause": result.likely_cause,
             "Recommended action": result.recommended_action, "Decision method": result.reason}
            for result in results.values()
        ],
        width="stretch", hide_index=True,
    )
    for result in results.values():
        with st.expander(result.agent_name, expanded=result.agent_name == "Learning Agent"):
            st.write("**Incident summary:**", result.incident_summary)
            st.write("**Evidence:**")
            for item in result.evidence:
                st.write("- ", item)
            st.write("**Likely cause:**", result.likely_cause)
            st.write("**Recommended action:**", result.recommended_action)
            st.write("**Verification:**")
            for step in result.verification_steps:
                st.write("- ", step)
            st.write("**Limitations:**")
            for limitation in result.limitations:
                st.write("- ", limitation)
            st.write("**Why this decision:**", result.reason)
    st.subheader("Actual Ollama tool calls")
    st.json(trace)
    st.subheader("Evaluation reference (not supplied to agents)")
    st.json(load_expected()[scenario_id])
    st.download_button(
        "Download comparison JSON",
        data=json.dumps(comparison_rows(scenario, results), indent=2),
        file_name=f"comparison_{scenario_id}.json", mime="application/json",
    )

    st.subheader("Learning feedback (after comparison)")
    learning = LearningAgent(local_history_path())
    actions = learning.available_actions(scenario)
    proposed = results["Learning Agent"].recommended_action
    selected_index = next((i for i, action in enumerate(actions) if proposed.startswith(action)), 0)
    action = st.selectbox("Action whose observed outcome you are recording", actions, index=selected_index)
    outcome = st.radio("Observed outcome", ("Success", "Failure"), horizontal=True)
    if st.button("Record simulated outcome"):
        before = learning.decide(scenario)
        learning.record_feedback(scenario, action, successful=outcome == "Success", notes="SIMULATED UI feedback")
        after = learning.decide(scenario)
        st.success("Feedback saved. The original six-agent comparison remains frozen.")
        st.write("Before:", before.recommended_action, before.evidence[-1])
        st.write("After:", after.recommended_action, after.evidence[-1])


if __name__ == "__main__":
    main()
