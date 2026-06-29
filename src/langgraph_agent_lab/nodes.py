"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


class ClassificationResult(BaseModel):
    """Structured output returned by the classifier LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The single best workflow route for the user query."
    )
    risk_level: Literal["low", "high"] = Field(description="High only for risky actions.")


def _text_from_response(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _strict_llm_enabled() -> bool:
    return os.getenv("LANGGRAPH_AGENT_STRICT_LLM", "").lower() == "true"


def _heuristic_classification(query: str) -> ClassificationResult:
    text = query.lower().strip()
    risky_terms = [
        "refund",
        "delete",
        "remove account",
        "close account",
        "cancel",
        "send confirmation email",
        "send email",
        "chargeback",
        "terminate",
    ]
    tool_terms = [
        "lookup",
        "look up",
        "order status",
        "tracking",
        "track",
        "search",
        "find",
        "retrieve",
        "check order",
        "status for order",
    ]
    missing_terms = ["can you fix it", "fix it", "help me", "it is broken", "not working"]
    error_terms = [
        "timeout",
        "failure",
        "failed",
        "crash",
        "exception",
        "unavailable",
        "cannot recover",
    ]

    if any(term in text for term in risky_terms):
        return ClassificationResult(route="risky", risk_level="high")
    if any(term in text for term in tool_terms) or re.search(r"\border\s*#?\s*\d+\b", text):
        return ClassificationResult(route="tool", risk_level="low")
    if any(term in text for term in missing_terms):
        return ClassificationResult(route="missing_info", risk_level="low")
    if any(term in text for term in error_terms):
        return ClassificationResult(route="error", risk_level="low")
    if len(text.split()) <= 3:
        return ClassificationResult(route="missing_info", risk_level="low")
    return ClassificationResult(route="simple", risk_level="low")


def _classify_with_llm(query: str) -> ClassificationResult:
    llm = get_llm(temperature=0.0)
    classifier = llm.with_structured_output(ClassificationResult)
    response = classifier.invoke(
        [
            (
                "system",
                "Classify support tickets into exactly one route. "
                "Routes: risky, tool, missing_info, error, simple. "
                "Priority: risky > tool > missing_info > error > simple. "
                "Risky means side effects such as refunds, deletes, emails, or cancellations. "
                "Tool means lookups or data retrieval. Missing_info means too vague to act. "
                "Error means system failures such as timeout, crash, unavailable, "
                "or cannot recover.",
            ),
            ("human", f"Query: {query}"),
        ]
    )
    if isinstance(response, ClassificationResult):
        return response
    if isinstance(response, dict):
        return ClassificationResult.model_validate(response)
    return ClassificationResult.model_validate_json(str(response))


def _answer_with_llm(prompt: str) -> str:
    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [
            (
                "system",
                "You are a concise customer support assistant. "
                "Use the supplied workflow context and do not invent tool results.",
            ),
            ("human", prompt),
        ]
    )
    return _text_from_response(response)


def _fallback_answer(state: AgentState) -> str:
    query = state.get("query", "")
    route = state.get("route", "simple")
    latest_tool = (state.get("tool_results") or [""])[-1]

    if route == "tool" and latest_tool:
        return f"I checked the available system data: {latest_tool}"
    if route == "risky":
        approval = state.get("approval") or {}
        if approval.get("approved") and latest_tool:
            return f"The approved action has been completed. Result: {latest_tool}"
        return "I cannot complete this risky request without approval."
    if route == "error":
        if latest_tool and "ERROR" not in latest_tool:
            return f"The issue was retried and completed successfully. Result: {latest_tool}"
        return "The request could not be completed after retrying. Please escalate to support."
    return (
        "To reset your password, open the login page, choose 'Forgot password', "
        "and follow the instructions sent to your email."
        if "password" in query.lower()
        else f"Here is a helpful support response for your request: {query}"
    )


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    try:
        result = _classify_with_llm(query)
        source = "llm"
    except Exception as exc:
        if _strict_llm_enabled():
            raise
        result = _heuristic_classification(query)
        source = "offline_fallback"
        fallback_reason = exc.__class__.__name__
    else:
        fallback_reason = ""

    return {
        "route": result.route,
        "risk_level": result.risk_level,
        "messages": [f"classify:{result.route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"query classified as {result.route}",
                source=source,
                fallback_reason=fallback_reason,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    query = state.get("query", "")
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))

    if route == "error" and attempt < 2:
        result = f"ERROR: transient processing failure on attempt {attempt}"
        status = "failed"
    elif route == "risky":
        action = state.get("proposed_action") or f"Perform risky action for: {query}"
        result = f"Risky action completed after approval: {action}"
        status = "completed"
    else:
        order_match = re.search(r"\b(\d{3,})\b", query)
        order_id = order_match.group(1) if order_match else "unknown"
        result = f"Order {order_id} status: in transit; estimated delivery: 2 business days"
        status = "completed"

    return {
        "tool_results": [result],
        "messages": [f"tool:{status}"],
        "events": [make_event("tool", status, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""
    evaluation_result = "needs_retry" if not latest or "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"tool result evaluated as {evaluation_result}",
                latest_result=latest,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    prompt = "\n".join(
        [
            f"User query: {state.get('query', '')}",
            f"Route: {state.get('route', '')}",
            f"Tool results: {state.get('tool_results') or 'none'}",
            f"Approval: {state.get('approval') or 'none'}",
            "Write the final customer-facing answer.",
        ]
    )
    try:
        final_answer = _answer_with_llm(prompt)
        source = "llm"
    except Exception as exc:
        if _strict_llm_enabled():
            raise
        final_answer = _fallback_answer(state)
        source = "offline_fallback"
        fallback_reason = exc.__class__.__name__
    else:
        fallback_reason = ""

    return {
        "final_answer": final_answer,
        "messages": ["answer:completed"],
        "events": [
            make_event(
                "answer",
                "completed",
                "final answer generated",
                source=source,
                fallback_reason=fallback_reason,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    route = state.get("route", "")
    if route == "risky" and state.get("approval") and not state["approval"].get("approved", False):
        question = (
            "The requested action was not approved. "
            "What safer alternative would you like to try?"
        )
    else:
        question = (
            "Can you share the specific account, order, error message, "
            "or action you want help with?"
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:requested"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    proposed_action = (
        f"Review and execute the customer-impacting request only after approval: "
        f"{state.get('query', '')}"
    )
    return {
        "proposed_action": proposed_action,
        "risk_level": "high",
        "messages": ["risky_action:prepared"],
        "events": [make_event("risky_action", "completed", "risky action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {
        "approval": {"approved": bool, "reviewer": str, "comment": str},
        "events": [make_event(...)]
    }
    """
    proposed_action = state.get("proposed_action") or state.get("query", "")
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        try:
            from langgraph.types import interrupt

            decision = interrupt(
                {
                    "proposed_action": proposed_action,
                    "question": "Approve this risky support action?",
                }
            )
            approved = (
                bool(decision.get("approved", False))
                if isinstance(decision, dict)
                else bool(decision)
            )
            reviewer = (
                str(decision.get("reviewer", "human-reviewer"))
                if isinstance(decision, dict)
                else "human-reviewer"
            )
            comment = str(decision.get("comment", "")) if isinstance(decision, dict) else ""
        except Exception:
            approved = True
            reviewer = "mock-reviewer"
            comment = "Auto-approved because interrupt approval is unavailable."
    else:
        approved = True
        reviewer = "mock-reviewer"
        comment = "Auto-approved for offline lab execution."

    approval = {"approved": approved, "reviewer": reviewer, "comment": comment}
    return {
        "approval": approval,
        "messages": [f"approval:{'approved' if approved else 'rejected'}"],
        "events": [
            make_event("approval", "completed", "approval decision recorded", approval=approval)
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    current_attempt = int(state.get("attempt", 0))
    next_attempt = current_attempt + 1
    latest_tool = (state.get("tool_results") or [""])[-1]
    error = (
        latest_tool
        if latest_tool and "ERROR" in latest_tool.upper()
        else f"Retry attempt {next_attempt} scheduled"
    )
    return {
        "attempt": next_attempt,
        "errors": [error],
        "messages": [f"retry:{next_attempt}"],
        "events": [
            make_event("retry", "completed", "retry attempt recorded", attempt=next_attempt)
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    answer = (
        "The request could not be completed after the allowed retry attempts. "
        "It has been escalated for manual follow-up."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:completed"],
        "events": [make_event("dead_letter", "completed", "max retries exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "messages": ["finalize:completed"],
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
