from triage_env.agents.base_agent import BaseAgent
from triage_env.models import TriageAction


class RuleBasedAgent(BaseAgent):
    """
    Treat the alive patient with the lowest health.
    If no valid patient exists, wait.
    """

    def act(self, observation) -> TriageAction:
        alive_patients = [p for p in observation.patients if p.alive]

        if not alive_patients:
            return TriageAction(action_type="wait", patient_id=-1)

        target = min(alive_patients, key=lambda p: p.health)
        return TriageAction(action_type="treat", patient_id=target.id)