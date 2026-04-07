from triage_env.models import TriageObservation


def build_system_prompt() -> str:
    return (
        "You are a medical triage decision system. "
        "Return exactly one JSON object and no other text. "
        "You must choose only legal actions for the current state. "
        "The JSON schema is: "
        "{\"action_type\": \"treat\"|\"allocate_ventilator\"|\"wait\", "
        "\"patient_id\": int|null, \"reasoning\": string}."
    )


def build_user_prompt(observation: TriageObservation) -> str:
    severity_rank = {"mild": 1, "moderate": 2, "severe": 3, "critical": 4}

    def urgency_score(severity: str, health: float, waiting_time: int) -> float:
        return (severity_rank[severity] * 100) - health + (waiting_time * 5)

    task_name = str(observation.metadata.get("task", "task2"))
    patient_lines = []
    alive_ids = []
    for patient in observation.patients:
        if patient.alive:
            alive_ids.append(patient.id)
            score = urgency_score(patient.severity, patient.health, patient.waiting_time)
        else:
            score = -1.0
        patient_lines.append(
            f"- id={patient.id}, severity={patient.severity}, health={patient.health:.1f}, "
            f"waiting_time={patient.waiting_time}, alive={patient.alive}, "
            f"ventilated={patient.ventilated}, urgency_score={score:.1f}"
        )

    if not patient_lines:
        patient_lines.append("- no patients available")

    return (
        f"Task: {task_name}\n"
        f"Step: {observation.step_count}\n"
        f"Resources: medics_available={observation.resources.medics_available}, "
        f"ventilators_available={observation.resources.ventilators_available}\n"
        f"Alive patient ids: {alive_ids if alive_ids else 'none'}\n"
        f"Legal actions: treat(patient_id), allocate_ventilator(patient_id), wait(patient_id=null/-1)\n"
        "Rules: choose only alive patients for treat/allocate_ventilator; wait when no valid patient exists.\n"
        "Output: JSON only, no markdown, no explanation outside JSON.\n"
        "Patients:\n"
        + "\n".join(patient_lines)
    )


def observation_to_prompt(observation: TriageObservation) -> str:
    return build_user_prompt(observation)