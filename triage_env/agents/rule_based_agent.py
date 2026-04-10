from triage_env.agents.base_agent import BaseAgent
from triage_env.models import TriageAction

_SEVERITY_PRIORITY = {"critical": 0, "severe": 1, "moderate": 2, "mild": 3}


class RuleBasedAgent(BaseAgent):
    """
    Ventilator-aware triage agent.

    Priority order:
    1. Allocate a free ventilator to the most critical unventilated patient.
    2. Treat the alive patient with the lowest health.
    3. Wait if no patients are alive.
    """

    def act(self, observation) -> TriageAction:
        alive_patients = [p for p in observation.patients if p.alive]

        if not alive_patients:
            return TriageAction(action_type="wait", patient_id=-1)

        # Priority 1: ventilate critical/severe unventilated patients if a ventilator is free
        if observation.resources.ventilators_available > 0:
            vent_candidates = [
                p for p in alive_patients
                if not p.ventilated and p.severity in ("critical", "severe")
            ]
            if vent_candidates:
                target = min(
                    vent_candidates,
                    key=lambda p: (_SEVERITY_PRIORITY.get(p.severity, 99), p.health),
                )
                return TriageAction(action_type="allocate_ventilator", patient_id=target.id)

        # Priority 2: treat the most critical lowest-health patient
        target = min(
            alive_patients,
            key=lambda p: (_SEVERITY_PRIORITY.get(p.severity, 99), p.health),
        )
        return TriageAction(action_type="treat", patient_id=target.id)