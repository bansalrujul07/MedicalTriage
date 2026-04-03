import random

try:
    from triage_env.agents.base_agent import BaseAgent
    from triage_env.models import TriageAction, TriageObservation
except ImportError:
    from agents.base_agent import BaseAgent
    from models import TriageAction, TriageObservation


class RandomAgent(BaseAgent):
    def act(self, observation: TriageObservation) -> TriageAction:
        alive = [p for p in observation.patients if p.alive]

        if not alive:
            return TriageAction(action_type="wait", patient_id=-1)

        actions = [TriageAction(action_type="wait", patient_id=-1)]

        for p in alive:
            actions.append(TriageAction(action_type="treat", patient_id=p.id))
            actions.append(TriageAction(action_type="allocate_ventilator", patient_id=p.id))

        return random.choice(actions)