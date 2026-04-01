try:
    from triage_env.models import TriageAction, TriageObservation
except ImportError:
    from models import TriageAction, TriageObservation


class RuleBasedAgent:
    def act(self, observation: TriageObservation) -> TriageAction:
        alive = [p for p in observation.patients if p.alive]

        if not alive:
            return TriageAction(action_type="wait", patient_id=-1)

        # 1. If any critical patient is not ventilated and ventilator exists, assign it
        critical_unventilated = sorted(
            [
                p for p in alive
                if p.severity == "critical" and not p.ventilated
            ],
            key=lambda p: (p.health, -p.waiting_time),
        )
        if critical_unventilated and observation.resources.ventilators_available > 0:
            return TriageAction(
                action_type="allocate_ventilator",
                patient_id=critical_unventilated[0].id,
            )

        # 2. Prioritize the patient in greatest danger, not just by severity
        # lower health first, then larger waiting time, then severity rank
        severity_rank = {"critical": 0, "severe": 1, "moderate": 2, "mild": 3}

        target = sorted(
            alive,
            key=lambda p: (
                p.health > 35,                 # immediate danger first
                severity_rank[p.severity],
                p.health,
                -p.waiting_time,
            ),
        )[0]

        return TriageAction(action_type="treat", patient_id=target.id)