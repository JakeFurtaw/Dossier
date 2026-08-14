# Agentic Workflow Test Demo

A minimal multi-agent ReAct demo. A **planner** decomposes a goal and can spawn **several researchers in parallel**. Each researcher’s report is checked by an **evaluator** (PASS / WEAK / FAIL) before it returns. The planner then calculates and calls `final_answer`.

## Setup

Create the project conda environment (name is `test_muli-agent_workflow`):

```bash
conda create -n test_muli-agent_workflow python=3.12 -y
conda activate test_muli-agent_workflow
pip install -r requirements.txt
playwright install chromium
```

Ollama must be running locally with a tool-capable model. Default: `qwen3.6`.

```bash
ollama serve          # if it is not already running
ollama list           # confirm the model is pulled
```

## Run

```bash
python agentic_workflow_test.py
```

Custom goal:

```bash
python agentic_workflow_test.py "What is the population of Porto, Portugal vs a town of 5,000?"
```

The terminal shows a **live agent tree** (spinner while Ollama or a tool is running) and colored Thought / Action / Observation panels. Nested researchers are indented under the planner. The final answer is rendered as markdown.

Each run also writes a report you can reopen later:

```
runs/YYYYMMDD-HHMMSS.md      # full trace + final answer
```

```bash
python agentic_workflow_test.py --verbose              # do not truncate thoughts/observations
python agentic_workflow_test.py --no-save              # skip writing runs/
python agentic_workflow_test.py --report-dir ./out     # write reports somewhere else
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.6` | Chat model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_TEMPERATURE` | `0.2` | Sampling temperature |
| `OLLAMA_NUM_PREDICT` | `2048` | Max generated tokens |
| `OLLAMA_REASONING` | `1` | Use the model's thinking stream as Thought |
| `PLANNER_MAX_ITERS` | `10` | Planner ReAct steps |
| `RESEARCHER_MAX_ITERS` | `6` | Researcher ReAct steps |
| `VERBOSE` | `0` | Same as `--verbose` |
| `REPORT_DIR` | `runs` | Default report directory |
| `MAX_PARALLEL_RESEARCHERS` | `3` | Cap on concurrent researcher tasks |
| `EVALUATOR_ENABLED` | `1` | Run an evaluator after each researcher |
| `EVALUATOR_RETRY` | `1` | One extra researcher pass if the evaluator says FAIL |

## Why this is an agentic workflow

The system is given a **goal**, not a fixed pipeline. The planner writes a plan, then chooses tools at runtime based on what it still needs. Researcher sub-agents run their own ReAct loops: they search, read pages, and adapt if a query or URL fails. Observations are written back into conversation memory, so later thoughts can change the plan (retry a different query, browse a better source, or spawn another researcher). Arithmetic is a tool call, not a guessed number. The loop stops only when `final_answer` is called or an iteration cap is hit. That is the opposite of a single LLM call or a hard-coded search → summarize chain: the path is produced by the agents, not by the programmer.
