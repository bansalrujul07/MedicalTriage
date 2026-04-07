import json
import logging
import re
from typing import Any

from triage_env.models import TriageAction

LOGGER = logging.getLogger(__name__)
VALID_ACTIONS = {"treat", "allocate_ventilator", "wait"}


def _extract_json_block(text: str) -> str | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return None


def _coerce_patient_id(raw: Any) -> int:
    if raw is None:
        return -1
    if isinstance(raw, bool):
        return -1
    if isinstance(raw, int):
        return raw
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def parse_llm_action(payload: str | dict[str, Any]) -> TriageAction:
    data: dict[str, Any] = {}

    if isinstance(payload, dict):
        data = payload
    elif isinstance(payload, str):
        source = payload.strip()
        if source:
            try:
                data = json.loads(source)
            except json.JSONDecodeError:
                maybe_json = _extract_json_block(source)
                if maybe_json is not None:
                    try:
                        data = json.loads(maybe_json)
                    except json.JSONDecodeError:
                        LOGGER.warning("LLM response JSON parsing failed; using fallback action")
                else:
                    LOGGER.warning("LLM response contained no JSON object; using fallback action")

    action_type = data.get("action_type", "wait")
    patient_id = _coerce_patient_id(data.get("patient_id", -1))

    if action_type not in VALID_ACTIONS:
        LOGGER.warning("LLM produced invalid action_type '%s'; using wait", action_type)
        return TriageAction(action_type="wait", patient_id=-1)

    if action_type == "wait":
        return TriageAction(action_type="wait", patient_id=-1)

    if patient_id < 0:
        LOGGER.warning("LLM produced invalid patient_id '%s'; using wait", patient_id)
        return TriageAction(action_type="wait", patient_id=-1)

    return TriageAction(action_type=action_type, patient_id=patient_id)
