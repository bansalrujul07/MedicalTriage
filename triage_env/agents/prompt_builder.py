try:
    from triage_env.models import TriageObservation
except ImportError:
    from models import TriageObservation


def observation_to_prompt(observation: TriageObservation) -> str:
    severity_rank = {"mild": 1, "moderate": 2, "severe": 3, "critical": 4}

    alive_patients = [p for p in observation.patients if p.alive]

    def urgency_score(p):
        return (severity_rank[p.severity] * 100) - p.health + (p.waiting_time * 5)

    sorted_alive = sorted(alive_patients, key=urgency_score, reverse=True)

    patient_lines = []
    for p in observation.patients:
        score = urgency_score(p) if p.alive else -1
        patient_lines.append(
            f"Patient {p.id}: severity={p.severity}, health={p.health}, "
            f"waiting_time={p.waiting_time}, alive={p.alive}, "
            f"ventilated={p.ventilated}, urgency_score={score}"
        )

    most_urgent_text = "None"
    if sorted_alive:
        top = sorted_alive[0]
        most_urgent_text = (
            f"Patient {top.id} "
            f"(severity={top.severity}, health={top.health}, "
            f"waiting_time={top.waiting_time}, urgency_score={urgency_score(top)})"
        )

        return f"""
You are a medical triage decision agent.

Goal:
Maximize total survival across ALL patients.
Do not keep treating one patient if another patient is closer to death.
Re-evaluate urgency at every step.

Current step: {observation.step_count}

Resources:
- medics_available: {observation.resources.medics_available}
- ventilators_available: {observation.resources.ventilators_available}

Most urgent patient right now:
- {most_urgent_text}

Patients:
{chr(10).join(patient_lines)}

Critical action validity rules:
- You may only choose a patient_id for a patient who is alive=true
- Never choose a dead patient
- If a patient has alive=false, they cannot be treated or ventilated
- If choosing treat or allocate_ventilator, patient_id must belong to a currently alive patient
- Never allocate a ventilator to a patient who already has ventilated=true
- Never choose allocate_ventilator if ventilators_available is 0
- If no ventilator is available, choose treat or wait instead
- If no valid patient exists, return wait with patient_id = -1

Decision rules:
- Prefer the patient most at risk of dying soon
- If a critical or severe patient is unventilated and ventilators are available, consider allocate_ventilator
- Do not repeatedly treat a stable patient while another patient is collapsing
- Avoid invalid actions
- After assigning one ventilator, do not repeat allocate_ventilator for the same patient

Return exactly one valid JSON object only.

Output format:
{{
  "action_type": "treat" | "allocate_ventilator" | "wait",
  "patient_id": integer
}}
""".strip()