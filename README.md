# Incident assistant: shared baseline

This repository starts the Agentic AI assignment with one reproducible source of
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

## Three independent teammate handoffs

The notebook, JSON, tools, contracts and environment above are the shared
baseline, **not** anyone's assigned data task. Once every machine passes
setup, work in parallel:

| Owner | Independent deliverable | Handoff |
| --- | --- | --- |
| Chakritha | Five distinct classical agents, tests, registry and Learning feedback hook | Importable registry, A-C results and before/after learning example to Vishruth |
| Tanweer | Local Ollama/LangChain agent, tests and actual relevant tool-call traces | Callable agent, A-C trace/results and explicit failure behavior to Vishruth |
| Vishruth | Streamlit selector, six-agent comparison, integration, evaluation, screenshots and submission | Runnable app, 18-row three-case table, README finalization and test/screenshots to team |

Use the common `AgentResult` rather than editing teammates' modules. The
Streamlit app and five/LLM agents are **assigned work, not implemented in the
baseline**; after integration, run the app with
`.\.venv\Scripts\streamlit.exe run app.py` (or
`.venv/bin/streamlit run app.py`). Evaluate three cases × six agents with
identical visible inputs, verify at least two real relevant LLM tool calls per
case, and demonstrate Learning before/after a manual outcome update. For C,
check the timestamps and previous healthy version before recommending a
recovery; keep the recommendation advisory.
