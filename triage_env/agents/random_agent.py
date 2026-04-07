import random

from triage_env.agents.base_agent import BaseAgent
from triage_env.models import TriageAction


class RandomAgent(BaseAgent):
    """
    Picks a random valid action.
    """

    def act(self, observation) -> TriageAction:
        alive_patients = [p for p in observation.patients if p.alive]

        possible_actions = [TriageAction(action_type="wait", patient_id=-1)]

        for patient in alive_patients:
            possible_actions.append(
                TriageAction(action_type="treat", patient_id=patient.id)
            )

        return random.choice(possible_actions)