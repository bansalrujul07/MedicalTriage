import json

try:
    from triage_env.models import TriageAction
except ImportError:
    from models import TriageAction


VALID_ACTIONS = {"treat", "allocate_ventilator", "wait"}


def parse_llm_action(text: str) -> TriageAction:
    try:
        data = json.loads(text)
        action_type = data.get("action_type", "wait")
        patient_id = data.get("patient_id", -1)

        if action_type not in VALID_ACTIONS:
            return TriageAction(action_type="wait", patient_id=-1)

        if not isinstance(patient_id, int):
            patient_id = -1

        if action_type == "wait":
            patient_id = -1

        return TriageAction(action_type=action_type, patient_id=patient_id)

    except Exception:
        return TriageAction(action_type="wait", patient_id=-1)