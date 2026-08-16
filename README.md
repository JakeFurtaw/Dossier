# Mutli-Agent Workflow Demo

A minimal multi-agent ReAct demo. A **planner** decomposes a goal and can spawn **several researchers in parallel**. Each researcher’s report is checked by an **evaluator** (PASS / WEAK / FAIL) before it returns. The planner then calculates and calls `final_answer`.

## Layout

```
workflow/
  agents/          planner, researcher, evaluator
  runtime/         ReAct loop, TraceBus, salvage chain, markdown reports
  tools/           web_search, browse_page, calculator, stop tools
  config.py
  prompts.py
  util.py
tests/             pytest for citations, calc, recovery, replay, …
```

## Setup

Create the project conda environment (name is `test_muli-agent_workflow`):

```bash
conda create -n test_muli-agent_workflow python=3.12 -y
conda activate test_muli-agent_workflow
pip install -r requirements.txt
playwright install chromium
```

Ollama must be running locally with a tool-capable model. Default: `qwen3.8:latest`.

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

The terminal stays compact by default: a **live agent tree** plus one-line tool actions (and evaluator verdicts). Thoughts, page extracts, and full tool JSON are **not** printed. The markdown report still has the complete trace. Use `--verbose` for the old full panels.

Each run also writes a report you can reopen later:

```
runs/YYYYMMDD-HHMMSS.md      # full trace + final answer
```

```bash
python agentic_workflow_test.py --verbose              # full Thought / Action / Observation panels
python agentic_workflow_test.py --no-save              # skip writing runs/
python agentic_workflow_test.py --report-dir ./out     # write reports somewhere else
python agentic_workflow_test.py --replay runs/foo.md   # re-render a saved run (no LLM)
python agentic_workflow_test.py --replay runs/foo.md --reaudit
```

`--replay` reconstructs the event tree from a saved markdown report and
re-renders it. `--reaudit` also runs the current citation checker against
the saved final answer (useful after changing `citations.py`).

```bash
pytest
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.8:latest` | Default chat model (all roles) |
| `OLLAMA_MODEL_PLANNER` | `$OLLAMA_MODEL` | Planner / final-answer synthesis |
| `OLLAMA_MODEL_RESEARCHER` | `$OLLAMA_MODEL` | Researcher sub-agents |
| `OLLAMA_MODEL_EVALUATOR` | `$OLLAMA_MODEL` | PASS / WEAK / FAIL judge |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_TEMPERATURE` | `0.2` | Sampling temperature |
| `OLLAMA_NUM_PREDICT` | `2048` | Max generated tokens |
| `OLLAMA_REASONING` | `1` | Use the model's thinking stream as Thought |
| `PLANNER_MAX_ITERS` | `8` | Planner ReAct steps |
| `RESEARCHER_MAX_ITERS` | `6` | Researcher ReAct steps |
| `VERBOSE` | `0` | Same as `--verbose` |
| `REPORT_DIR` | `runs` | Default report directory |
| `MAX_PARALLEL_RESEARCHERS` | `3` | Cap on concurrent researcher tasks |
| `EVALUATOR_ENABLED` | `1` | Run an evaluator after each researcher |
| `EVALUATOR_RETRY` | `1` | One extra researcher pass if the evaluator says FAIL |
| `CITATION_CHECK` | `1` | Verify cited URLs against tool output (researcher reports + final answer) |
| `CITATION_STRICT` | `0` | Exit non-zero if any URL in the final answer is unverified |

## Shared context

Researchers in the same run share a compact ledger (queries already issued,
URLs already opened, and a short summary of each finished report). Later
researchers — retries and a second spawn — see that digest in their prompt
and are told not to repeat that work. `web_search` is also cached per run
(same idea as the page cache), so two parallel researchers who pick the same
query only hit the network once.

The ledger is written into the run report as `## Shared context`.

## Citation verification

Every run audits where its sources came from — pure string matching, no extra
model calls, no GPU load:

- each **researcher report** is checked against that researcher's own
  `web_search` / `browse_page` output. A `**Citation check:** …` line is
  appended to the report, so the planner and the evaluator both see it.
- the **final answer** is checked against the researcher reports the planner
  received. The audit table lands in the run report as
  `## Citation audit (final answer)`.
- numbers cited on the same line as a URL are also matched against the
  observed page text for that URL (researcher level only).

URLs are compared canonically (no `www.`, `old.reddit.com` vs `reddit.com`,
query/fragment/trailing-slash differences), so formatting drift is not read as
fabrication. `CITATION_STRICT=1` is meant for CI/evals: a run with unverified
URLs exits non-zero.

## Why this is an agentic workflow

The system is given a **goal**, not a fixed pipeline. The planner writes a plan, then chooses tools at runtime based on what it still needs. Researcher sub-agents run their own ReAct loops: they search, read pages, and adapt if a query or URL fails. Observations are written back into conversation memory, so later thoughts can change the plan (retry a different query, browse a better source, or spawn another researcher). Arithmetic is a tool call, not a guessed number. The loop stops only when `final_answer` is called or an iteration cap is hit. That is the opposite of a single LLM call or a hard-coded search → summarize chain: the path is produced by the agents, not by the programmer.
