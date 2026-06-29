"""Report generation helper.

TODO(student): implement report rendering using MetricsReport data
and the template in reports/lab_report_template.md.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    TODO(student): Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    scenario_rows = "\n".join(
        (
            "| {scenario_id} | {expected_route} | {actual_route} | "
            "{success} | {retry_count} | {interrupt_count} |"
        ).format(
            scenario_id=item.scenario_id,
            expected_route=item.expected_route,
            actual_route=item.actual_route or "",
            success="yes" if item.success else "no",
            retry_count=item.retry_count,
            interrupt_count=item.interrupt_count,
        )
        for item in metrics.scenario_metrics
    )
    return f"""# Day 08 Lab Report

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
{scenario_rows}

Summary:

- Total scenarios: {metrics.total_scenarios}
- Success rate: {metrics.success_rate:.2%}
- Average nodes visited: {metrics.avg_nodes_visited:.2f}
- Total retries: {metrics.total_retries}
- Total interrupts: {metrics.total_interrupts}

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
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")


def _event_label(event: dict[str, Any]) -> str:
    node = html.escape(str(event.get("node", "unknown")))
    event_type = html.escape(str(event.get("event_type", "")))
    message = html.escape(str(event.get("message", "")))
    return f"<strong>{node}</strong><span>{event_type}</span><p>{message}</p>"


def _metadata_value(state: dict[str, Any], node: str, key: str) -> str:
    for event in state.get("events", []) or []:
        if event.get("node") == node:
            metadata = event.get("metadata", {}) or {}
            value = metadata.get(key)
            if value:
                return str(value)
    return ""


def render_ui_report(metrics: MetricsReport, final_states: list[dict[str, Any]]) -> str:
    """Render a self-contained HTML report for demo review."""
    state_by_id = {str(item.get("scenario_id", "")): item for item in final_states}
    cards = []
    for item in metrics.scenario_metrics:
        state = state_by_id.get(item.scenario_id, {})
        events = state.get("events", []) or []
        timeline = "\n".join(f"<li>{_event_label(event)}</li>" for event in events)
        tool_results = "\n".join(
            f"<li>{html.escape(str(result))}</li>" for result in state.get("tool_results", []) or []
        )
        errors = "\n".join(
            f"<li>{html.escape(str(error))}</li>" for error in state.get("errors", []) or []
        )
        classify_source = _metadata_value(state, "classify", "source") or "n/a"
        answer_source = _metadata_value(state, "answer", "source") or "n/a"
        approval_text = "yes" if item.approval_observed else "no"
        llm_text = f"{html.escape(classify_source)} / {html.escape(answer_source)}"
        answer = html.escape(str(state.get("final_answer") or state.get("pending_question") or ""))
        query = html.escape(str(state.get("query", "")))
        success_class = "ok" if item.success else "bad"
        card = f"""
            <article class="scenario">
              <header>
                <div>
                  <h2>{html.escape(item.scenario_id)}</h2>
                  <p>{query}</p>
                </div>
                <span class="badge {success_class}">{'PASS' if item.success else 'FAIL'}</span>
              </header>
              <div class="grid">
                <div><span>Expected</span><strong>{html.escape(item.expected_route)}</strong></div>
                <div><span>Actual</span><strong>{html.escape(str(item.actual_route))}</strong></div>
                <div><span>Nodes</span><strong>{item.nodes_visited}</strong></div>
                <div><span>Retries</span><strong>{item.retry_count}</strong></div>
                <div><span>Approval</span><strong>{approval_text}</strong></div>
                <div><span>LLM</span><strong>{llm_text}</strong></div>
              </div>
              <section>
                <h3>Final Output</h3>
                <p class="answer">{answer}</p>
              </section>
              <section>
                <h3>Tool Results</h3>
                <ul>{tool_results or '<li>none</li>'}</ul>
              </section>
              <section>
                <h3>Errors</h3>
                <ul>{errors or '<li>none</li>'}</ul>
              </section>
              <section>
                <h3>Node Timeline</h3>
                <ol class="timeline">{timeline}</ol>
              </section>
            </article>
            """
        cards.append("\n".join(line.rstrip() for line in card.splitlines()).strip())

    cards_markup = "\n".join(cards)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LangGraph Agent Lab Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2433;
      --muted: #667085;
      --line: #d9dee8;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --accent: #0f766e;
      --warn: #b42318;
      --soft: #e8f3f1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }}
    h2, h3 {{ letter-spacing: 0; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .summary div, .scenario {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgb(16 24 40 / 0.04);
    }}
    .summary div {{ padding: 16px; }}
    .summary span, .grid span {{ color: var(--muted); font-size: 13px; display: block; }}
    .summary strong {{ display: block; margin-top: 6px; font-size: 24px; }}
    .scenario {{ padding: 18px; margin-top: 16px; }}
    .scenario header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
    }}
    .scenario h2 {{ margin: 0 0 6px; font-size: 20px; }}
    .scenario p {{ margin: 0; color: var(--muted); }}
    .badge {{
      border-radius: 999px;
      padding: 5px 10px;
      font-weight: 700;
      font-size: 12px;
      white-space: nowrap;
    }}
    .badge.ok {{ background: var(--soft); color: var(--accent); }}
    .badge.bad {{ background: #fee4e2; color: var(--warn); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin: 16px 0;
    }}
    .grid div {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; }}
    .grid strong {{ display: block; margin-top: 4px; }}
    section {{ margin-top: 16px; }}
    section h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .answer {{
      background: #f9fafb;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      color: var(--ink);
    }}
    ul, ol {{ margin: 0; padding-left: 22px; }}
    .timeline li {{ margin-bottom: 10px; }}
    .timeline span {{ margin-left: 8px; color: var(--muted); font-size: 12px; }}
    .timeline p {{ margin: 3px 0 0; }}
    @media (max-width: 640px) {{
      main {{ padding: 24px 12px 36px; }}
      .scenario header {{ display: block; }}
      .badge {{ display: inline-block; margin-top: 10px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>LangGraph Agent Lab Report</h1>
    <p>
      Scenario execution dashboard with route quality, retries, approval, tool results,
      and node timeline.
    </p>
    <section class="summary">
      <div><span>Total scenarios</span><strong>{metrics.total_scenarios}</strong></div>
      <div><span>Success rate</span><strong>{metrics.success_rate:.2%}</strong></div>
      <div><span>Average nodes</span><strong>{metrics.avg_nodes_visited:.2f}</strong></div>
      <div><span>Total retries</span><strong>{metrics.total_retries}</strong></div>
      <div><span>Total approvals</span><strong>{metrics.total_interrupts}</strong></div>
    </section>
    {cards_markup}
  </main>
</body>
</html>
"""


def write_ui_report(
    metrics: MetricsReport, final_states: list[dict[str, Any]], output_path: str | Path
) -> None:
    """Write the HTML report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ui_report(metrics, final_states), encoding="utf-8")


def write_run_details(final_states: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write full final states for debugging and demos."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(final_states, indent=2, ensure_ascii=False, default=str)
    path.write_text(payload, encoding="utf-8")
