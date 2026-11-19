# Incident assistant

This repository implements the Agentic AI assignment using one reproducible source of
synthetic incident data. **A** is critical DB-pool exhaustion; **B** is a normal
high-traffic/CPU path without DB exhaustion; **C** is an ambiguous startup
failure after a config change (a stale DB warning is deliberately misleading).
No module connects to a real service or performs recovery actions.

## Prerequisites and common environment

- Python **3.11**, `pip`, and a local Ollama installation with enough memory to
  run **`llama3.1:8b`**. The first model pull needs internet access. Ollama is
  installed separately, not through `requirements.txt`.
- `requirements.txt` pins the direct Python dependencies: classic Jupyter
  Notebook (notebook/export), Streamlit (UI), `langchain-ollama` and
  `langchain-core` (LLM tool calling), and pytest (tests). Transitive
  dependencies are installed by pip.
- Each teammate uses the **same** notebook export, requirements file, and
  model tag. Do not copy fixture data into agent files.

From the repository root, create a Python 3.11 virtual environment:

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

```sh
# macOS/Linux shell
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Install/start Ollama from its official distribution, then in a separate
terminal run `ollama pull llama3.1:8b` and `ollama serve` if your installation
does not already run a local server. The default local endpoint is
`http://127.0.0.1:11434`. With the server running, check a **real** tool call:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py  # Windows
```

```sh
.venv/bin/python scripts/check_environment.py  # macOS/Linux
```

The check fails explicitly if Ollama is offline, the model is missing, or it
does not call the offered tool. It does not claim that the assignment's
two-tool-per-scenario LLM requirement is satisfied; that belongs to the LLM
agent implementation.

## Notebook and immutable comparison data

Open `notebooks/incident_scenarios.ipynb` with Jupyter Notebook from the
repository root (`.\.venv\Scripts\jupyter.exe notebook` on Windows or
`.venv/bin/jupyter notebook` on macOS/Linux), then **Run All**. The last cell
exports `data/scenarios.json` (agent-visible evidence) and
`data/expected_results.json` (evaluation answers only). Re-exporting the
unchanged notebook produces byte-for-byte identical files. Keep these exports
committed and use the scenario loader rather than maintaining a second copy.
`pytest` verifies both exports against the notebook. Never import or pass
`expected_results.json` to an agent, its prompt, or its tools.

```python
from incd_assist.scenarios import load_scenarios
from incd_assist.tools import get_metrics, search_logs, get_deployment_history, check_previous_version

scenario = load_scenarios()["A"]
print(get_metrics(scenario))
print(search_logs(scenario, "timeout"))
print(get_deployment_history(scenario))
print(check_previous_version(scenario))
```

`load_scenarios()` returns `{id: IncidentScenario}` with read-only metrics,
deployment records, and previous-version details. The scenario has `id`,
`label`, `metrics`, `logs`, `deployment_history`, and `previous_version`.
Each `Agent.decide(scenario)` returns `AgentResult` with `agent_name`,
`incident_summary`, `evidence`, `likely_cause`, `recommended_action`,
`verification_steps`, `limitations`, and `reason`; see
`incd_assist/contracts.py`. Tool functions accept the selected scenario as
their first argument and return plain JSON-compatible copies. A LangChain
adapter can bind closures around these functions for one selected scenario
and log the tool names, arguments, and actual returns. For a fair comparison,
pass the same read-only scenario to all six agents; record learning feedback
only after collecting their original decisions.

## Team contributions

The notebook, JSON, tools, contracts and environment above form the common
baseline. The three independently built components are now integrated:

| Owner | Contribution | Integration point |
| --- | --- | --- |
| Chakritha | Five distinct classical agents and Learning feedback hook | `incd_assist/classical_agents.py` |
| Tanweer | Local Ollama/LangChain agent | `incd_assist/llm_agent.py` with actual model-requested tool calls |
| Vishruth | Streamlit selector, six-agent comparison, evaluation and submission artifacts | `app.py`, `scripts/run_evaluation.py`, `outputs/` |

The five classical agents, Ollama agent, and Streamlit app are integrated with
plain Python and LangChain tool binding; **LangGraph is not required**. Run the app with
`.\.venv\Scripts\streamlit.exe run app.py` (or
`.venv/bin/streamlit run app.py`). Select a case, click **Compare all six
agents**, review the evidence and comparison, then optionally record a
**simulated** success/failure in the Learning feedback section. Its history
starts from `data/learning_agent_experience.json` and is saved locally in
ignored `data/learning_agent_experience_integration.json`; the original
comparison stays frozen until you click Compare again. Do not treat simulated
feedback as a real recovery outcome.

To reproduce the three-case experiment, start Ollama and run:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_evaluation  # Windows
```

```sh
.venv/bin/python -m scripts.run_evaluation  # macOS/Linux
```

This writes `outputs/comparison_results.json` (18 agent rows, real LLM tool
traces, and separate reviewer-only expected outcomes),
`outputs/comparison_results.csv`, and
`outputs/learning_feedback_demo.json` (a before/after score from **simulated**
feedback; the committed seed history remains unchanged). The scenario
selector and live six-agent comparison screenshots are in
`outputs/screenshots/`; `outputs/test_results.txt` records the baseline and
integration test run. Review each cause and recommendation against
the expected outcomes manually; scores and wording are not reliable exact-match
classifiers. Verify two different relevant LLM tools per scenario, compare
their returned data with the recommendation, and demonstrate Learning's score
before and after a simulated feedback entry. For C, check timestamps and the
previous healthy version before recommending recovery. If Ollama is down or
the model fails to call tools, the run **fails explicitly**; it does not
fabricate a sixth result.
