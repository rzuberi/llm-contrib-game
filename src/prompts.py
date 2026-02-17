
import json
import math
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple


def _json_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def recent_contributions_for_agent(
    *,
    agent_id: str,
    contribution_history: List[Dict[str, float]],
    window: int,
) -> Dict[str, List[float]]:
    recent = contribution_history[-window:] if window > 0 else []
    if not recent:
        return {}

    all_agents = sorted(recent[-1].keys())
    result: Dict[str, List[float]] = {}
    for other in all_agents:
        if other == agent_id:
            continue
        result[other] = [float(round_data.get(other, 0.0)) for round_data in recent]
    return result


def build_message_prompt(
    *,
    agent_id: str,
    wealth: float,
    round_index: int,
    recent_other_contributions: Dict[str, List[float]],
    chat_history: List[Dict[str, Any]],
) -> str:
    return (
        f"You are {agent_id} in round {round_index} of a repeated public-goods game.\\n"
        f"Your current wealth is {wealth:.6f}.\\n"
        "You are in the message sub-round. Post exactly one short strategic message to the whole group.\\n"
        "Recent contributions from OTHER agents (last up to 3 rounds):\\n"
        f"{_json_compact(recent_other_contributions)}\\n\\n"
        "Chat history so far:\\n"
        f"{_json_compact(chat_history)}\\n\\n"
        "Output STRICT JSON only with this schema:"
        " {\"message\": \"<your message>\"}. No extra keys."
    )


def build_contribution_prompt(
    *,
    agent_id: str,
    wealth: float,
    round_index: int,
    round_messages: Dict[str, str],
    recent_other_contributions: Dict[str, List[float]],
) -> str:
    return (
        f"You are {agent_id} in round {round_index} of a repeated public-goods game.\\n"
        f"Your current wealth is {wealth:.6f}.\\n"
        "Choose your contribution c for this round. Constraints: 0 <= c <= wealth.\\n"
        "Messages posted by all agents this round:\\n"
        f"{_json_compact(round_messages)}\\n\\n"
        "Recent contributions from OTHER agents (last up to 3 rounds):\\n"
        f"{_json_compact(recent_other_contributions)}\\n\\n"
        "Output STRICT JSON only with this schema:"
        " {\"contribution\": <number>, \"reason\": \"<very short reason>\"}."
        " Do not output any extra text."
    )


def build_contribution_repair_prompt(
    *,
    wealth: float,
    raw_response: str,
) -> str:
    return (
        "Rewrite the prior model output into valid JSON for contribution parsing.\\n"
        f"Constraint: contribution must be a finite number in [0, {wealth:.6f}].\\n"
        "Output STRICT JSON only with schema:"
        " {\"contribution\": <number>, \"reason\": \"<short reason>\"}.\\n"
        "Prior raw output:\\n"
        f"{raw_response}"
    )


def build_round_summarizer_prompt(
    *,
    round_index: int,
    messages: Dict[str, str],
    contributions: Dict[str, float],
    wealth_before: Dict[str, float],
    wealth_after: Dict[str, float],
    pot: float,
    redistributed_per_agent: float,
    gini: float,
    mean_contribution: float,
) -> str:
    return (
        f"Summarize round {round_index} of a repeated public-goods game.\\n"
        f"Messages: {_json_compact(messages)}\\n"
        f"Contributions: {_json_compact(contributions)}\\n"
        f"Wealth before: {_json_compact(wealth_before)}\\n"
        f"Wealth after: {_json_compact(wealth_after)}\\n"
        f"Total pot: {pot:.6f}. Redistribution per agent: {redistributed_per_agent:.6f}.\\n"
        f"Computed gini: {gini:.6f}. Mean contribution: {mean_contribution:.6f}.\\n"
        "Output format requirements:\\n"
        "- 3 to 6 bullet points only\\n"
        "- Then one final line starting with 'Stats:' containing top wealth, bottom wealth, gini, mean contribution."
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_message_response(raw_response: str) -> Tuple[Optional[str], Optional[str]]:
    obj = _extract_json_object(raw_response)
    if obj is not None and "message" in obj:
        message = str(obj.get("message", "")).strip()
        if message:
            return message, None

    fallback = raw_response.strip()
    if fallback:
        short = " ".join(fallback.split())
        return short[:1000], "message_json_parse_failed_used_text_fallback"

    return None, "empty_message_response"


def parse_contribution_response(raw_response: str, wealth: float) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    obj = _extract_json_object(raw_response)
    if obj is None:
        return None, None, "no_json_object_found"

    if "contribution" not in obj:
        return None, None, "json_missing_contribution_field"

    contribution_raw = obj.get("contribution")
    try:
        contribution = float(contribution_raw)
    except (TypeError, ValueError):
        return None, None, f"invalid_contribution_type:{type(contribution_raw).__name__}"

    if not math.isfinite(contribution):
        return None, None, "contribution_not_finite"

    contribution = max(0.0, min(float(wealth), contribution))
    reason = str(obj.get("reason", "")).strip() if obj.get("reason") is not None else ""
    return contribution, reason, None
