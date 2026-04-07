import random
from collections import defaultdict

from triage_env.agents.base_agent import BaseAgent
from triage_env.models import TriageAction
from triage_env.training.state_encoder import encode_observation


class QLearningAgent(BaseAgent):
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.checkpoint_metadata = {}
        self.checkpoint_warning = None

    def get_valid_actions(self, observation):
        actions = [("wait", -1)]

        for patient in observation.patients:
            if patient.alive:
                actions.append(("treat", patient.id))
                if (
                    observation.resources.ventilators_available > 0
                    and not patient.ventilated
                ):
                    actions.append(("allocate_ventilator", patient.id))

        return actions

    def act(self, observation):
        state = encode_observation(observation)
        valid_actions = self.get_valid_actions(observation)

        if random.random() < self.epsilon:
            action_type, patient_id = random.choice(valid_actions)
        else:
            q_values = [self.q_table[(state, action)] for action in valid_actions]
            max_q = max(q_values)
            best_actions = [
                action for action, q in zip(valid_actions, q_values) if q == max_q
            ]
            action_type, patient_id = random.choice(best_actions)

        return TriageAction(action_type=action_type, patient_id=patient_id)

    def update(self, state, action, reward, next_state, done, next_valid_actions):
        current_q = self.q_table[(state, action)]

        if done:
            target = reward
        else:
            next_q = max(self.q_table[(next_state, a)] for a in next_valid_actions)
            target = reward + self.gamma * next_q

        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)

    def save(self, path, metadata: dict | None = None):
        import pickle
        import json
        from pathlib import Path

        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

        sidecar = Path(f"{path}.meta.json")
        with sidecar.open("w", encoding="utf-8") as handle:
            json.dump(metadata or {}, handle, indent=2)

    def load(self, path):
        import pickle
        import json
        from pathlib import Path

        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data)

        sidecar = Path(f"{path}.meta.json")
        if sidecar.exists():
            with sidecar.open("r", encoding="utf-8") as handle:
                self.checkpoint_metadata = json.load(handle)
        else:
            self.checkpoint_metadata = {}
        self.checkpoint_warning = None