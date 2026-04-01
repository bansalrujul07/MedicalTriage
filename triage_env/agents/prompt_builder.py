try:
    from triage_env.models import TriageObservation
except ImportError:
    from models import TriageObservation


def observation_to_prompt(observation: TriageObservation) -> str:
    patient_lines = []
    for p in observation.patients:
        patient_lines.append(
            f"Patient {p.id}: severity={p.severity}, health={p.health}, "
            f"waiting_time={p.waiting_time}, alive={p.alive}, ventilated={p.ventilated}"
        )

    return f"""
You are a medical triage decision agent.

Current step: {observation.step_count}

Resources:
- medics_available: {observation.resources.medics_available}
- ventilators_available: {observation.resources.ventilators_available}

Patients:
{chr(10).join(patient_lines)}

Choose exactly one action in JSON format:
{{
  "action_type": "treat" | "allocate_ventilator" | "wait",
  "patient_id": integer
}}

Rules:
- Use patient_id = -1 for wait
- Prefer saving critical patients first
- Avoid invalid actions
- Return JSON only
""".strip()