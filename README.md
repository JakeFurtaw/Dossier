# Dossier

A local multi-agent research runtime. A planner fans out parallel specialists, an evaluator retries weak reports, and every cited URL is checked against tool output — no extra model call.

The loop is shared. A **recipe** is just the prompts and the specialist agents the planner is allowed to spawn. `research` is general-purpose; `apartments` is the same runtime aimed at a Reston / Herndon hunt.

Runs entirely on your machine through Ollama. The path is produced by the agents, not a hard-coded search-then-summarize chain.

## What it does

Give Dossier a goal and a recipe (`--workflow`). A **planner** writes a short plan and delegates independent questions in one turn. Each **specialist** (a researcher, or listing / geo / amenities in the apartment recipe) runs its own ReAct loop (`web_search`, `browse_page`, then `report_findings`). An **evaluator** scores that report `PASS` / `WEAK` / `FAIL` and a `FAIL` gets one extra pass. The planner then does any arithmetic and must call `final_answer` — writing the answer in a thought does not finish the run.

What is unusual is the reliability layer around that loop:

- **Parallel supervisor** — `spawn_researchers` (or several `spawn_researcher` calls in one turn) runs workers at the same time, capped by `MAX_PARALLEL_RESEARCHERS`.
- **Shared ledger** — researchers publish queries, URLs, and short report summaries so later siblings and retries do not repeat that work. Search and page fetches are cached and single-flighted per run.
- **Hybrid browse** — `browse_page` tries a 20 MB-capped httpx GET first. Playwright only runs for thin, JS-heavy, or anti-bot pages, in an isolated browser context, and Chromium starts only on that first fallback.
- **Deterministic citation audit** — every cited URL is checked against tool output (canonical host, no extra GPU). Numbers next to a URL are matched against the browsed page text. `CITATION_STRICT=1` fails the process if the final answer cites anything unverified.
- **Salvage and replay** — if an agent hits the iteration cap without its stop tool, a fallback chain still returns the evidence it gathered. `--replay` re-renders a saved `runs/*.md` file without calling the model; `--reaudit` re-runs the current citation checker on that answer.

The terminal stays compact: a live agent tree plus one-line tool actions and evaluator verdicts. Thoughts, page extracts, and full tool JSON stay in the markdown report unless you pass `--verbose`.

## How a run is structured

```
goal
 └─ planner                         (prompts come from the recipe)
     ├─ spawn_*                     (researchers, or listing + geo + amenities)
     │   ├─ specialist  (search → browse → report_findings)
     │   │   └─ evaluator  PASS | WEAK | FAIL  (+ one retry on FAIL)
     │   └─ specialist  …
     ├─ calculator                  (only if the goal needs arithmetic)
     └─ final_answer                → runs/YYYYMMDD-HHMMSS.md
```

## Layout

```
dossier.py             CLI entry point
workflow/
  recipes/             named workflows (prompts + specialist roster)
    research.py        general research (default)
    apartments.py      Reston / Herndon listing + geo + amenities
  agents/              planner, specialist runner, evaluator
  runtime/             ReAct loop, TraceBus, salvage, citations, replay, reports
  tools/               web_search, browse_page, fetch_raw, calculator, stop tools
  config.py
  prompts.py           re-exports the research prompts
tests/                 pytest for citations, calc, recovery, replay, ledger, …
```

The loop lives in `workflow/runtime/react.py` so Thought / Action / Observation stay explicit instead of disappearing into a graph runtime. Add a recipe by dropping a module next to `research.py` that builds a `Recipe` (planner prompt, specialist prompts, spawn-tool descriptions) and registering it in `workflow/recipes/__init__.py`.

## Setup

Activate the Dossier environment (recommended):
```bash
conda activate Dossier
```

```bash
pip install -r requirements.txt
playwright install chromium
```

Ollama must be running locally with a tool-capable model. Default: `qwen3.8:latest`.

```bash
ollama serve          # if it is not already running
ollama list           # confirm the model is pulled
```

## Usage

```bash
python dossier.py
python dossier.py "What is the population of Porto, Portugal vs a town of 5,000?"
python dossier.py --workflow apartments
python dossier.py --workflow apartments "2 bed under $2800 near Wiehle, pets ok"
python dossier.py --list-workflows
```

```bash
python dossier.py --verbose              # full Thought / Action / Observation panels
python dossier.py --no-save              # skip writing runs/
python dossier.py --report-dir ./out     # write reports somewhere else
python dossier.py --replay runs/foo.md   # re-render a saved run (no LLM)
python dossier.py --replay runs/foo.md --reaudit
```

`--replay` rebuilds the event tree from a saved markdown report and re-renders it. `--reaudit` also runs the current citation checker against the saved final answer (useful after changing `citations.py`).

`--workflow apartments` keeps the same tools and loop. The orchestrator spawns a **listing** agent (public listing sites → a normalized table; there is no RentCast/Apify API in this runtime), a **geo** agent (Reston Town Center, Silver Line stations, Dulles Toll Road), then an **amenities** agent that scores units and estimates total monthly cost. Pass your own constraints as the goal; the default is a Reston / Herndon 1–2 bed hunt under $3,000.

```bash
pytest
```

Swap models without editing code:

```bash
OLLAMA_MODEL=gemma4:31b python dossier.py
OLLAMA_MODEL_EVALUATOR=qwen2.5:3b python dossier.py
OLLAMA_HOST=http://host:11434 python dossier.py
```

## Citation verification

Every run audits where its sources came from — string matching only, no extra model calls:

- each **researcher report** is checked against that researcher's own `web_search` / `browse_page` output. A `**Citation check:** …` line is appended so the planner and the evaluator both see it.
- the **final answer** is checked against the researcher reports the planner received. The audit table lands in the run report as `## Citation audit (final answer)`.
- numbers cited on the same line as a URL are also matched against the observed page text for that URL (researcher level only).

URLs are compared canonically (no `www.`, `old.reddit.com` vs `reddit.com`, query/fragment/trailing-slash differences), so formatting drift is not treated as fabrication.

## Shared context

Researchers in the same run share a compact ledger: queries already issued, URLs already opened, and a short summary of each finished report. Later researchers — retries and a second spawn — see that digest in their prompt and are told not to repeat that work. `web_search` is cached per run the same way pages are, so two parallel researchers who pick the same query only hit the network once.

The ledger is written into the run report as `## Shared context`.

## Configuration

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
| `CITATION_CHECK` | `1` | Verify cited URLs against tool output |
| `CITATION_STRICT` | `0` | Exit non-zero if any URL in the final answer is unverified |
