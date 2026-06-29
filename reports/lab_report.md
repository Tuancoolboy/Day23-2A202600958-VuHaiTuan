# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit:
- Date:

## 2. Architecture

The workflow is a LangGraph support-ticket agent. It normalizes the query, classifies it into
one route, runs the route-specific nodes, and sends every path through `finalize`.

### File responsibilities

| File | Responsibility |
|---|---|
| `state.py` | Defines `Scenario`, `AgentState`, event schema, and initial state |
| `llm.py` | Creates the Gemini/OpenAI/Anthropic chat model from environment variables |
| `nodes.py` | Implements the workflow steps: classify, tool, evaluate, answer, approval, retry |
| `routing.py` | Maps state values to the next node name |
| `graph.py` | Registers nodes and connects fixed/conditional LangGraph edges |
| `cli.py` | Loads config/scenarios, invokes the graph, and writes output artifacts |
| `metrics.py` | Converts final states into grading metrics |
| `report.py` | Produces Markdown, JSON run details, and an HTML UI report |

Core paths:

- `simple`: classify -> answer -> finalize
- `tool`: classify -> tool -> evaluate -> answer or retry/dead_letter -> finalize
- `missing_info`: classify -> clarify -> finalize
- `risky`: classify -> risky_action -> approval -> tool or clarify -> finalize
- `error`: classify -> retry -> tool/evaluate or dead_letter -> finalize

### End-to-end data flow

1. `data/sample/scenarios.jsonl` provides one query per scenario.
2. `scenarios.py` validates each JSONL row as a `Scenario`.
3. `state.py` converts the scenario into an `AgentState`.
4. `graph.py` starts at `intake`, then moves through nodes based on edges.
5. `classify_node` uses structured LLM output to choose one of five routes.
6. `routing.py` turns that route into the next node name.
7. Route-specific nodes update state with answers, tool results, approvals, or errors.
8. `metrics.py` validates final state against the expected route.
9. The CLI writes `metrics.json`, `run_details.json`, `ui_report.html`, and this report.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| query | overwrite | normalized user request |
| route | overwrite | selected workflow route |
| risk_level | overwrite | marks risky actions |
| attempt | overwrite | bounded retry counter |
| final_answer | overwrite | final customer-facing output |
| pending_question | overwrite | clarification output |
| proposed_action | overwrite | risky action awaiting approval |
| approval | overwrite | human/mock approval decision |
| evaluation_result | overwrite | retry gate after tool execution |
| messages | append | lightweight trace |
| tool_results | append | preserve each tool attempt |
| errors | append | retry/failure audit trail |
| events | append | grading and debugging audit trail |

## 4. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | yes | 0 | 0 |
| S02_tool | tool | tool | yes | 0 | 0 |
| S03_missing | missing_info | missing_info | yes | 0 | 0 |
| S04_risky | risky | risky | yes | 0 | 1 |
| S05_error | error | error | yes | 2 | 0 |
| S06_delete | risky | risky | yes | 0 | 1 |
| S07_dead_letter | error | error | yes | 1 | 0 |

Summary:

- Total scenarios: 7
- Success rate: 100.00%
- Average nodes visited: 6.43
- Total retries: 3
- Total interrupts: 2

## 5. Failure analysis

1. Retry or tool failure: tool results containing `ERROR` are routed to retry until
   `attempt >= max_attempts`, then dead-lettered.
2. Risky action without approval: risky requests are routed through approval before the
   tool/action step. Rejections go to clarification instead of execution.
3. Missing information: vague requests avoid hallucinated answers and return a clarification
   question through `pending_question`.
4. LLM/provider failure: nodes can fall back to deterministic offline behavior unless
   `LANGGRAPH_AGENT_STRICT_LLM=true` is set for real-provider validation.

## 6. Persistence / recovery evidence

Each run uses a stable `thread_id` from the scenario id. Memory checkpointer is wired by
default, and SQLite can be enabled with `checkpointer: sqlite` plus sqlite extras installed.

## 7. Extension work

Implemented extensions:

- HTML UI report: `outputs/ui_report.html`
- Full final-state trace: `outputs/run_details.json`
- Strict LLM mode: `LANGGRAPH_AGENT_STRICT_LLM=true`
- `.env` loading in `llm.py` for local API keys
- SQLite checkpointer support when `.[sqlite]` extras are installed
- Offline fallback runner for environments without LangGraph installed, while still using
  real `StateGraph` when the dependency exists.

## 8. Improvement plan

Productionization priorities:

1. Replace the mock tool with typed tools for order lookup, refunds, email actions, and
   customer account operations.
2. Add true human-in-the-loop review UI for approval/rejection.
3. Add rate limiting and retry/backoff for provider quota errors.
4. Persist checkpoints in SQLite/Postgres and demonstrate crash recovery.
5. Add hidden-scenario regression tests for routing priority and prompt robustness.
